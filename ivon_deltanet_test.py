#!/usr/bin/env python3
# test_ivon_deltanet.py
import os, math, argparse, random
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_float32_matmul_precision('high')

from fla.layers.delta_net_ivon import DeltaNetIVON

# simple cache shim matching the .update() contract used by your layer
class SimpleCache(list):
    def update(self, recurrent_state=None, ivon_h_state=None, ivon_g_state=None,
               conv_state=None, layer_idx=None, offset=None, **kwargs):
        assert layer_idx is not None
        if len(self) <= layer_idx:
            self.extend([None] * (layer_idx - len(self) + 1))
        self[layer_idx] = dict(
            recurrent_state=recurrent_state,
            ivon_h_state=ivon_h_state,
            ivon_g_state=ivon_g_state,
            conv_state=conv_state,
            offset=offset,
        )

def make_varlen_mask(B, T, min_len=8):
    """Create a [B,T] 0/1 mask with random valid lengths (1s are valid tokens)."""
    lens = [random.randint(min_len, T) for _ in range(B)]
    mask = torch.zeros(B, T, dtype=torch.bool)
    for i, L in enumerate(lens):
        mask[i, :L] = 1
    return mask

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--B', type=int, default=2)
    p.add_argument('--T', type=int, default=64)         # keep <=64 to hit fused path in the layer
    p.add_argument('--d_model', type=int, default=256)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--expand_k', type=float, default=1.0)
    p.add_argument('--expand_v', type=float, default=1.0)
    p.add_argument('--varlen', action='store_true')
    p.add_argument('--stream_chunks', type=int, default=4, help='split T into chunks for streaming test')
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--beta1', type=float, default=0.9)
    p.add_argument('--beta2', type=float, default=0.999)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--lam', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=123)
    args = p.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed)

    B, T, d_model = args.B, args.T, args.d_model
    H = args.heads

    layer = DeltaNetIVON(
        d_model=d_model,
        hidden_size=d_model,
        num_heads=H,
        expand_k=args.expand_k,
        expand_v=args.expand_v,
        use_beta=True,
        use_gate=False,
        use_short_conv=True,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        weight_decay=args.wd,
        lam=args.lam,
        rng_seed=args.seed,
        layer_idx=0,
        qk_activation='silu',
        qk_norm='l2',
    ).to(args.device)

    def rand_inputs(B, T, D):
        return torch.randn(B, T, D, device=args.device, dtype=torch.float32, requires_grad=False)

    # -------- Test A: equal-length forward + backward --------
    print("\n=== Test A: equal-length ===")
    x = rand_inputs(B, T, d_model)
    cache = SimpleCache()
    x.requires_grad_(True)  # to propagate into projections

    o, _, cache = layer(hidden_states=x, attention_mask=None, past_key_values=cache, use_cache=True)
    print("Output:", tuple(o.shape))
    # simple loss: mean squared vs random target
    tgt = torch.randn_like(o)
    loss = F.mse_loss(o, tgt)
    loss.backward()

    # check some grad norms
    grads = {
        'q_proj': layer.q_proj.weight.grad,
        'k_proj': layer.k_proj.weight.grad,
        'v_proj': layer.v_proj.weight.grad,
    }
    for name, g in grads.items():
        gn = float(g.norm().detach()) if g is not None else 0.0
        print(f"grad {name}: {gn:.4e}")

    # reset grads
    layer.zero_grad(set_to_none=True)

    # -------- Test B: streaming with cache (chunks) --------
    print("\n=== Test B: streaming with cache ===")
    assert T % args.stream_chunks == 0, "T must be divisible by stream_chunks"
    chunk = T // args.stream_chunks
    cache = SimpleCache()
    xs = x.detach().clone()  # reuse same content for determinism
    outs = []
    for i in range(args.stream_chunks):
        sl = slice(i*chunk, (i+1)*chunk)
        xo = xs[:, sl, :].contiguous().requires_grad_(True)
        o_step, _, cache = layer(hidden_states=xo, attention_mask=None, past_key_values=cache, use_cache=True)
        outs.append(o_step.detach())
    o_stream = torch.cat(outs, dim=1)
    print("Streaming output:", tuple(o_stream.shape))
    # quick sanity: different from a fresh pass due to IVON adaptation (not a strict equality test)
    with torch.no_grad():
        o_fresh, _, _ = layer(hidden_states=x, attention_mask=None, past_key_values=SimpleCache(), use_cache=True)
        diff = (o_stream - o_fresh).abs().mean().item()
        print(f"Mean |stream - fresh|: {diff:.4e}")

    # -------- Test C: variable-length (ragged) --------
    print("\n=== Test C: variable-length ===")
    # Build a batch with padding, let the layer unpad via attention_mask
    mask = make_varlen_mask(B, T, min_len=max(8, T//4)).to(args.device)
    x_var = rand_inputs(B, T, d_model)
    cache = SimpleCache()
    o_var, _, cache = layer(hidden_states=x_var, attention_mask=mask, past_key_values=cache, use_cache=True)
    print("Varlen output:", tuple(o_var.shape))

    print("\nAll tests done.")

if __name__ == "__main__":
    main()
