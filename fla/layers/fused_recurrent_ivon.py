# New: IVON-based delta-rule forward kernel + wrapper (forward-only)
import torch
import triton
import triton.language as tl
from fla.utils import input_guard
from fla.modules.l2norm import l2norm_bwd, l2norm_fwd


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_ivon_delta_fwd_kernel(
    q, k, v,                    
    o,                         
    h0, ht,                     
    h_state, g_state,   
    beta,
    u,        
    cu_seqlens,                
    scale,                      
    lr, beta1, beta2,   
    weight_decay, lam,  # additional IVON parameters          
    seed: tl.constexpr,         
    T,                          
    B: tl.constexpr,
    Hh: tl.constexpr,
    Kd: tl.constexpr,
    Vd: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
    USE_BIAS_CORR: tl.constexpr  # if True, expect bc1_inv provided per step outside (here we keep False)
):
    # program ids
    i_v, i_k, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // Hh, i_nh % Hh

    # varlen handling
    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        allT = T
        T = eos - bos
    else:
        bos = i_n * T
        eos = bos + T
        allT = B * T

    # block pointers
    p_q = q + (bos * Hh + i_h) * Kd + i_k * BK + tl.arange(0, BK)
    p_k = k + (bos * Hh + i_h) * Kd + i_k * BK + tl.arange(0, BK)
    p_v = v + (bos * Hh + i_h) * Vd + i_v * BV + tl.arange(0, BV)
    p_o = o + (bos * Hh + i_h) * Vd + i_v * BV + tl.arange(0, BV)
    
    p_u = u + (bos * Hh + i_h) * Vd + i_v * BV + tl.arange(0, BV)
    if IS_BETA_HEADWISE:
        # beta is [B,T,H,V]
        p_beta = beta + (bos * Hh + i_h) * Vd + i_v * BV + tl.arange(0, BV)
    else:
        # beta is [B,T,H]
        p_beta = beta + bos * Hh + i_h

    # masks
    mask_k = (i_k * BK + tl.arange(0, BK)) < Kd
    mask_v = (i_v * BV + tl.arange(0, BV)) < Vd
    mask_hblk = mask_v[:, None] & mask_k[None, :]

    # local tiles
    # m is the memory matrix block [BV, BK]
    m = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_m0 = h0 + i_nh * Kd * Vd + (i_k * BK + tl.arange(0, BK)[None, :]) * Vd + (i_v * BV + tl.arange(0, BV)[:, None])
        m += tl.load(p_m0, mask=mask_hblk, other=0).to(tl.float32)

    # IVON states: h (diag precond) and g (grad mom), shape [BV,BK]
    p_hs = h_state + i_nh * Kd * Vd + (i_k * BK + tl.arange(0, BK)[None, :]) * Vd + (i_v * BV + tl.arange(0, BV)[:, None])
    p_gs = g_state + i_nh * Kd * Vd + (i_k * BK + tl.arange(0, BK)[None, :]) * Vd + (i_v * BV + tl.arange(0, BV)[:, None])

    hdiag = tl.load(p_hs, mask=mask_hblk, other=0).to(tl.float32)
    gmom  = tl.load(p_gs, mask=mask_hblk, other=0).to(tl.float32)

    # constants
    one = tl.full([], 1.0, tl.float32)
    eps_denom = tl.full([], 1e-8, tl.float32)

    # time loop
    for t in range(0, T):
        # load current q,k,v slices
        bk = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        bq = tl.load(p_q, mask=mask_k, other=0).to(tl.float32) * scale
        bv_true = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

        # sigma = 1 / sqrt(lam * (h + delta))  elementwise
        sigma = tl.rsqrt( tl.maximum(lam * (hdiag + weight_decay), eps_denom) )

        # sample epsilon ~ N(0,1) for this block and time
        # create a unique offset per (seq pos, head, tile, element)
        # base offset uses absolute time (bos + t)
        row_ids = (i_v * BV + tl.arange(0, BV))[:, None]
        col_ids = (i_k * BK + tl.arange(0, BK))[None, :]
        offs2d = (bos + t) * (Hh * Kd * Vd) + i_h * (Kd * Vd) + row_ids * Kd + col_ids
        eps = tl.randn(seed, offs2d)

        # theta (params) = m + sigma * eps
        theta_minus_m = sigma * eps
        # predict with theta: (V) = (VxK) * (K)
        pred_theta = tl.sum((m + theta_minus_m) * bk[None, :], axis=1)
        resid = bv_true - pred_theta

        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)

        u_gated = resid * b_beta                   # β ⊙ r
        tl.store(p_u, u_gated.to(p_u.dtype.element_ty), mask=mask_v)

        # TODO: need grad to compute hessian estimate. might be incorrect to have here
        g_hat = -(resid[:, None] * bk[None, :])

        # elementwise: h_hat = g_hat * ((theta - m)/sigma^2) = g_hat * (eps / sigma)  
        h_hat = g_hat * (eps / tl.maximum(sigma, eps_denom))

        # g \gets beta_1* g + (1-beta_1) * g_hat
        gmom = beta1 * gmom + (one - beta1) * g_hat

        # RGD update for h
        # h \gets beta_2 * h + (1-beta_2) * h_hat + 0.5 * (1-beta_2)^2 * (h - h_hat)^2 / (h + delta)
        one_minus_b2 = (one - beta2)
        diff = (hdiag - h_hat)
        corr = 0.5 * (one_minus_b2 * one_minus_b2) * (diff * diff) / tl.maximum(hdiag + weight_decay, eps_denom)
        hdiag = beta2 * hdiag + one_minus_b2 * h_hat + corr

        # TODO: bias-correction of g. if USE_BIAS_CORR: supply bc1_inv and multiply here
        gbar = gmom

        # m \gets m - lr * (gbar + delta * m)/(h + delta)
        step = (gbar + weight_decay * m) / tl.maximum(hdiag + weight_decay, eps_denom)
        m = m - lr * step

        # output with updated m: o_t = (m * q)^sum_K
        bo = tl.sum(m * bq[None, :], axis=1)
        tl.store(p_o, bo.to(p_o.dtype.element_ty), mask=mask_v)

        p_q += Hh * Kd
        p_k += Hh * Kd
        p_v += Hh * Vd
        p_o += Hh * Vd
        p_beta += Hh * (Vd if IS_BETA_HEADWISE else 1) 

    # write back states
    tl.store(p_hs, hdiag.to(p_hs.dtype.element_ty), mask=mask_hblk)
    tl.store(p_gs, gmom.to(p_gs.dtype.element_ty), mask=mask_hblk)

    if STORE_FINAL_STATE:
        p_mt = ht + i_nh * Kd * Vd + (i_k * BK + tl.arange(0, BK)[None, :]) * Vd + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_mt, m.to(p_mt.dtype.element_ty), mask=mask_hblk)


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'WRITE_DH0': lambda args: args['dh0'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_ivon_delta_rule_bwd_kernel(
    q,                
    k,                
    u,                
    beta,             
    h0, dh0, dht,     
    do,               
    dq, dk, dv, db,   
    Av_buf,           
    cu_seqlens,
    scale,
    # IVON hyperparams used only to propagate state gradient across time (NOT differentiated):
    lr, beta1, weight_decay,
    # We DO NOT backprop through the preconditioner; approximate with a constant denom per tile:
    inv_denom_scalar,   # float32 scalar  ~ 1.0/(delta + mean(h_tile))  (host-computed; optional tuning knob)
    #
    B: tl.constexpr, T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr, NK: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    WRITE_DH0: tl.constexpr,
):
    i_v, i_k, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H

    # time span
    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        allT = T
        Tloc = eos - bos
    else:
        bos = i_n * T
        eos = bos + T
        allT = B * T
        Tloc = T

    # masks
    mask_k = (i_k * BK + tl.arange(0, BK)) < K
    mask_v = (i_v * BV + tl.arange(0, BV)) < V
    mask_blk = mask_k[None, :] & mask_v[:, None]

    # set pointers to LAST timestep for reverse sweep
    p_q  = q  + (bos * H + i_h) * K + i_k * BK + tl.arange(0, BK) + (Tloc - 1) * H * K
    p_k  = k  + (bos * H + i_h) * K + i_k * BK + tl.arange(0, BK) + (Tloc - 1) * H * K
    p_u  = u  + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV) + (Tloc - 1) * H * V
    p_do = do + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV) + (Tloc - 1) * H * V
    p_dk = dk + ((i_v * allT + bos) * H + i_h) * K + i_k * BK + tl.arange(0, BK) + (Tloc - 1) * H * K
    p_dv = dv + ((i_k * allT + bos) * H + i_h) * V + i_v * BV + tl.arange(0, BV) + (Tloc - 1) * H * V
    
    if IS_BETA_HEADWISE:
        p_beta  = beta + (bos + Tloc - 1) * H * V + i_h * V + i_v * BV + tl.arange(0, BV)
        p_dbeta = db   + ((i_v * NK + i_k) * allT + bos + Tloc - 1) * H * V + i_h * V + tl.arange(0, BV)
    else:
        p_beta  = beta + (bos + Tloc - 1) * H + i_h
        p_dbeta = db   + (i_v * allT + bos + Tloc - 1) * H + i_h

    # reverse-sweep state adjoint for memory (same role as your b_dh)
    b_Gm = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        p_dht = dht + i_nh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        b_Gm += tl.load(p_dht, mask=mask_blk.T, other=0).to(tl.float32)

    # PRECONDITIONER APPROX: scalar inverse denom (no grad)
    one = tl.full((), 1.0, tl.float32)
    lr_t = tl.full((), lr, tl.float32)
    beta1_t = tl.full((), beta1, tl.float32)
    wd_t = tl.full((), weight_decay, tl.float32)
    inv_den = tl.full((), inv_denom_scalar, tl.float32)   # scalar preconditioner approx
    scale_t = tl.full((), scale, tl.float32) 

    # reverse sweep over time
    for _ in range(Tloc):
        b_q  = tl.load(p_q,  mask=mask_k, other=0).to(tl.float32) * scale
        b_k  = tl.load(p_k,  mask=mask_k, other=0).to(tl.float32)              
        b_u  = tl.load(p_u,  mask=mask_v, other=0).to(tl.float32)              
        b_do = tl.load(p_do, mask=mask_v, other=0).to(tl.float32)              

        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)      
        else:
            b_beta = tl.load(p_beta).to(tl.float32)                             

        # readout contribution: G_{m'} += q ⊗ do^T
        b_Gm += b_q[:, None] * b_do[None, :]

        # map through IVON param step (no grad through denom): m' = m - lr * (g' + wd m) * inv_den
        #    => G_{g'} += -lr * (G_{m'} \odot inv_den)
        #       G_m   += (G_{m'} \odot (1 - lr*wd*inv_den))
        b_Gm += b_q[:, None] * b_do[None, :]

        # map through IVON param step (no grad through denom)
        G_gprime = -lr_t * (b_Gm * inv_den)
        b_Gm     =  b_Gm * (one - lr_t * wd_t * inv_den)

        # through momentum
        G_ghat = (one - beta1_t) * tl.trans(G_gprime)

        # through g' = beta_1* g + (1-beta_1) g_hat  => G_{g_hat} += (1-beta_1) G_{g'}
        G_ghat = (1.0 - beta1) * G_gprime                                      
        G_ghat = tl.trans(G_ghat)                                              

        # helper: A_v = sum_j G_ghat[v,j] * k_j
        A_v = tl.sum(G_ghat * b_k[None, :], axis=1)                            
        # stash for forward sweep (theta-term for dk)
        p_Av = Av_buf + (i_v * allT + (bos + Tloc - 1)) * H * V + i_h * V + tl.arange(0, BV)
        tl.store(p_Av, A_v.to((Av_buf + 0).dtype.element_ty), mask=mask_v)

        # grads wrt v and beta through u = β ⊙ r
        # dv += - beta * A_v
        b_dv = -(A_v * b_beta)                                                 
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=mask_v)
        # dbeta:
        #  headwise: dbeta[v] += - r_v * A_v  (we only have u= beta * r; recover r = u / beta, clamped)
        #  scalar:   dbeta += sum_v (- r_v * A_v)
        # NOTE: use safe division; where beta≈0, treat r≈0 so term = 0.
        r_est = tl.where(b_beta != 0, b_u / b_beta, 0.0)
        if IS_BETA_HEADWISE:
            tl.store(p_dbeta, (-r_est * A_v).to(p_dbeta.dtype.element_ty), mask=mask_v)
        else:
            db_scalar = tl.sum(-r_est * A_v)
            tl.store(p_dbeta, db_scalar.to(p_dbeta.dtype.element_ty))

        # part of dk from the “− u \otimes k” structure:  dk += - sum_v G_ghat[v,:] * u_v
        dk_term1 = -tl.sum(G_ghat * b_u[:, None], axis=0)                       
        tl.store(p_dk, dk_term1.to(p_dk.dtype.element_ty), mask=mask_k)

        # residual→memory path
        b_Gm += ( (b_beta * A_v)[:, None] * b_k[None, :] ).T                    

        p_q -= H * K
        p_k -= H * K
        p_u -= H * V
        p_do -= H * V
        p_dk -= H * K
        p_dv -= H * V
        if IS_BETA_HEADWISE:
            p_dbeta -= H * V
            p_beta  -= H * V
        else:
            p_dbeta -= H
            p_beta  -= H

    # write dh0 if requested and allocated
    if WRITE_DH0:
        p_dh0 = dh0 + i_nh * K * V \
              + (i_k * BK + tl.arange(0, BK)[:, None]) * V \
              + (i_v * BV + tl.arange(0, BV)[None, :])
        tl.store(p_dh0, b_Gm.to(p_dh0.dtype.element_ty), mask=(mask_k[:, None] & mask_v[None, :]))
    tl.debug_barrier()

    # rebuild memory forward to compute dq with updated memory and add dk_term2 = (beta * A_v) * m
    b_m = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_m0 = h0 + i_nh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        b_m += tl.load(p_m0, mask=mask_blk, other=0).to(tl.float32)

    # reset forward pointers to t=0
    p_q = q + (bos * H + i_h) * K + i_k * BK + tl.arange(0, BK)
    p_k = k + (bos * H + i_h) * K + i_k * BK + tl.arange(0, BK)
    p_u = u + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV)
    p_do = do + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV)
    if IS_BETA_HEADWISE:
        p_beta = beta + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV)
    else:
        p_beta = beta + bos * H + i_h
    p_dq = dq + ((i_v * allT + bos) * H + i_h) * K + i_k * BK + tl.arange(0, BK)
    p_dk = dk + ((i_v * allT + bos) * H + i_h) * K + i_k * BK + tl.arange(0, BK)

    for _ in range(Tloc):
        b_q  = tl.load(p_q,  mask=mask_k, other=0).to(tl.float32) * scale
        b_k  = tl.load(p_k,  mask=mask_k, other=0).to(tl.float32)
        b_u  = tl.load(p_u,  mask=mask_v, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_v, other=0).to(tl.float32)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)

        # add dk term that needs the current memory (θ≈m):
        p_Av = Av_buf + (i_v * allT + bos) * H * V + i_h * V + tl.arange(0, BV)
        A_v  = tl.load(p_Av, mask=mask_v, other=0).to(tl.float32)             
        dk_term2 = tl.sum( (b_beta * A_v)[:, None] * b_m, axis=0 )             
        # accumulate into existing dk
        prev_dk = tl.load(p_dk, mask=mask_k, other=0).to(tl.float32)
        tl.store(p_dk, (prev_dk + dk_term2).to(p_dk.dtype.element_ty), mask=mask_k)

        # update memory forward using IVON but with u precomputed:
        # here we **only** need m' for dq, so we can emulate param step with g' from u:
        #   g_hat = - u ⊗ k
        #   g' = beta_1* g + (1-beta_1) g_hat  (we ignore g state; absorb into scale via (1-beta_1))
        #   m' = m - lr * ((1-beta_1) g_hat + wd m) * inv_den
        g_hatT = -(b_k[None, :] * b_u[:, None]) 
        b_m    =  b_m - lr_t * ((one - beta1_t) * g_hatT + wd_t * b_m) * inv_den

        b_dq = tl.sum(b_m * b_do[:, None], axis=0) * scale_t 
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=mask_k)

        p_q  += H * K
        p_k  += H * K
        p_u  += H * V
        p_do += H * V
        p_dq += H * K
        p_dk += H * K
        if IS_BETA_HEADWISE:
            p_beta += H * V
            p_Av   += H * V
        else:
            p_beta += H
            p_Av   += H * V



def fused_recurrent_ivon_delta_fwd(
    q,                 
    k,                 
    v,                 
    beta,              
    *,
    scale,
    initial_state,     
    h_state,           
    g_state,           
    lr,
    beta1,
    beta2,
    weight_decay,
    lam,
    rng_seed = 12345,
    output_final_state = False,
    cu_seqlens = None,
):
    B, T, H, K = k.shape
    V = v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 8)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"

    num_stages = 1
    num_warps = 1

    # outputs
    o = q.new_empty(B, T, H, V)
    # for beta \otimes residual
    u = torch.empty_like(v)                        
    final_state = q.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else o  

    grid = (NV, NK, N * H)

    fused_recurrent_ivon_delta_fwd_kernel[grid](
        q, k, v,
        o,
        initial_state, final_state,
        h_state, g_state,
        beta, u,                             
        cu_seqlens,
        scale,
        lr, beta1, beta2, weight_decay, lam,
        rng_seed,
        T=T, B=B, Hh=H, Kd=K, Vd=V,
        BK=BK, BV=BV,
        USE_INITIAL_STATE=initial_state is not None,
        STORE_FINAL_STATE=output_final_state,
        IS_VARLEN=cu_seqlens is not None,
        # True if [B,T,H,V], else scalar per head
        IS_BETA_HEADWISE=(beta.ndim == v.ndim),   
        USE_BIAS_CORR=False,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return o, u, (final_state if output_final_state else None)



def fused_recurrent_ivon_delta_rule_bwd(
    q, k, u, beta, dht, do,
    *,
    scale,
    initial_state,
    lr,
    beta1,
    weight_decay,
    inv_denom_scalar,
    cu_seqlens=None,
):
    B, T, H, K = k.shape
    V = u.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"

    num_stages = 1
    num_warps = 2

    beta_vector = (beta.ndim == u.ndim)

    dq = q.new_empty(NV, *q.shape)
    dk = q.new_empty(NV, *k.shape)
    dv = q.new_empty(NK, B, T, H, V)
    if beta_vector:
        db = q.new_empty(NV, NK, B, T, H, V)
    else:
        db = q.new_empty(NV, B, T, H)

    Av_buf = q.new_empty(NV, B, T, H, V)

    grid = (NV, NK, N * H)

    if (initial_state is not None) and initial_state.requires_grad:
        dh0 = torch.empty_like(initial_state, dtype=torch.float32)
    else:
        dh0 = None

    fused_recurrent_ivon_delta_rule_bwd_kernel[grid](
        q, k, u, beta,
        initial_state,
        dh0, dht,
        do,
        dq, dk, dv, db,
        Av_buf,
        cu_seqlens,
        scale,
        lr, beta1, weight_decay,
        inv_denom_scalar,
        B=B, T=T, H=H, K=K, V=V,
        BK=BK, BV=BV, NK=NK,
        IS_BETA_HEADWISE=beta_vector,
        USE_INITIAL_STATE=initial_state is not None,
        USE_FINAL_STATE_GRADIENT=dht is not None,
        IS_VARLEN=cu_seqlens is not None,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    dq = dq.sum(0)
    dk = dk.sum(0)
    dv = dv.sum(0)
    db = db.sum((0, 1)) if beta_vector else db.sum(0)

    return dq, dk, dv, db, dh0


class FusedRecurrentIVONFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        q, k, v, beta,
        scale,
        initial_state,
        h_state, g_state,
        lr, beta1, beta2, weight_decay, lam,
        rng_seed=12345,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=None,
    ):
        # optional in-kernel L2 norm for q/k
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        else:
            q_rstd, k_rstd = None, None

        # run IVON forward: returns o, u (= beta \odot residual), final_state (optional)
        o, u, final_state = fused_recurrent_ivon_delta_fwd(
            q=q, k=k, v=v, beta=beta,
            scale=scale,
            initial_state=initial_state,
            h_state=h_state,
            g_state=g_state,
            lr=lr, beta1=beta1, beta2=beta2,
            weight_decay=weight_decay, lam=lam,
            rng_seed=rng_seed,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )

        # partial bptt
        ctx.save_for_backward(q, q_rstd, k, k_rstd, u, beta, initial_state)
        # scalar/context fields
        ctx.scale = scale
        ctx.cu_seqlens = cu_seqlens
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel

        # backward needs lr, beta1, weight_decay, and an inverse denom approx
        # 1 / (wd + mean(h_state))
        with torch.no_grad():
            h_mean = float(h_state.mean()) if h_state is not None else 0.0
        ctx.lr = lr
        ctx.beta1 = beta1
        ctx.weight_decay = weight_decay
        ctx.inv_denom_scalar = 1.0 / max(1e-6, weight_decay + h_mean)

        return o, final_state

    @staticmethod
    @input_guard
    def backward(ctx, do, dht):
        q, q_rstd, k, k_rstd, u, beta, initial_state = ctx.saved_tensors

        dq, dk, dv, db, dh0 = fused_recurrent_ivon_delta_rule_bwd(
            q=q,
            k=k,
            u=u,                  # NOTE: u = beta \odot(v_true - \theta* k) saved in forward
            beta=beta,
            dht=dht,
            do=do,
            scale=ctx.scale,
            initial_state=initial_state,
            lr=ctx.lr,
            beta1=ctx.beta1,
            weight_decay=ctx.weight_decay,
            inv_denom_scalar=ctx.inv_denom_scalar,
            cu_seqlens=ctx.cu_seqlens,
        )

        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)

        # return grads aligned to forward signature:
        # q, k, v, beta, scale, initial_state, h_state, g_state, lr, beta1, beta2, weight_decay, lam, rng_seed, output_final_state, use_qk_l2norm_in_kernel, cu_seqlens
        return (
            dq.to(q),                 # q
            dk.to(k),                 # k
            dv.to(u),                 # v_true
            db.to(beta),              # beta
            None,                     # scale (scalar)
            dh0,                      # initial_state (tensor or None)
            None,                     # h_state (no grad)
            None,                     # g_state (no grad)
            None, None, None,         # lr, beta1, beta2
            None, None,               # weight_decay, lam
            None,                     # rng_seed
            None, None, None          # output_final_state, use_qk..., cu_seqlens
        )

@torch.compiler.disable
def fused_recurrent_ivon_delta_rule(
    q, k, v, beta=None,
    scale=None,
    initial_state=None,
    h_state=None,
    g_state=None,
    lr=1e-3,
    beta1=0.9,
    beta2=0.999,
    weight_decay=0.0,
    lam=1.0,
    rng_seed=12345,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    cu_seqlens=None,
):
    """
    Same contract as fused_recurrent_delta_rule, but memory updates use IVON
    (with momentum g_state and diagonal preconditioner h_state).
    """
    # varlen guards (same as your original)
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )

    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"

    if beta is None:
        beta = torch.ones_like(q[..., 0])   # [B,T,H] (scalar per head). Pass [B,T,H,V] if you want headwise.

    # Sanity for state tensors
    if initial_state is None or h_state is None or g_state is None:
        raise ValueError("IVON requires `initial_state`, `h_state`, and `g_state` tensors (shape [N,H,K,V]).")

    o, final_state = FusedRecurrentIVONFunction.apply(
        q, k, v, beta,
        scale,
        initial_state,
        h_state, g_state,
        lr, beta1, beta2, weight_decay, lam,
        rng_seed,
        output_final_state,
        use_qk_l2norm_in_kernel,
        cu_seqlens,
    )
    return o, final_state
