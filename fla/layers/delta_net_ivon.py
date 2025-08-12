# New IVON DeltaNet that uses the IVON fused recurrent kernels

import warnings
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.layers.fused_recurrent_ivon import fused_recurrent_ivon_delta_rule

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack
    from fla.models.utils import Cache


def elu_p1(x):
    return (F.elu(x, 1., False) + 1.).to(x)


def sum_norm(x):
    return (x / x.sum(-1, keepdim=True)).to(x)


class DeltaNetIVON(nn.Module):
    """
    Same interface/spirit as DeltaNet, but memory updates use IVON (optimizer-style) instead of the delta rule.

    Differences vs. DeltaNet:
      • Keeps the same projections/activations/short-conv/gating.
      • Uses fused recurrent IVON kernel only (no chunk mode here).
      • Maintains additional per-layer, per-sequence states: ivon_h_state (diag precond) and ivon_g_state (grad momentum).
    """

    def __init__(
        self,
        mode: str = 'fused_recurrent',
        d_model: Optional[int] = None,
        hidden_size: int = 1024,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        num_heads: int = 4,
        use_beta: bool = True,
        use_gate: bool = False,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        allow_neg_eigval: bool = False,
        layer_idx: Optional[int] = None,
        qk_activation: str = 'silu',
        qk_norm: str = 'l2',
        norm_eps: float = 1e-5,
        # IVON hyperparameters
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        weight_decay: float = 0.0,   # delta in the write rule
        lam: float = 1.0,            # lambda for sigma: 1/sqrt(lam(h+delta))
        rng_seed: int = 12345,
    ):
        super().__init__()

        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        assert self.qk_activation in ['silu', 'relu', 'elu', 'identity']
        assert self.qk_norm in ['l2', 'sum']

        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)

        if mode == 'fused_chunk':
            raise NotImplementedError("fused_chunk_delta_rule is now deprecated. Please use `chunk_delta_rule` instead.")
        assert mode in ['fused_recurrent'], f"Not supported mode `{mode}`. Chunk has not yet been implemented."
        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0

        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        self.use_beta = use_beta
        if self.use_beta:
            # per head scalar by default
            self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)

        if use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu' if qk_activation == 'silu' else None
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu' if qk_activation == 'silu' else None
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu'
            )
        else:
            warnings.warn(
                "ShortConvolution is crucial to performance. "
                "Avoid setting `use_short_conv=False` unless you know what you're doing."
            )

        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)

        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        # IVON hparams
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.lam = lam
        self.rng_seed = rng_seed

    def _proj_qkv(self, hidden_states, cu_seqlens):
        if self.use_short_conv:
            conv_state_q = conv_state_k = conv_state_v = None
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states), cache=None, output_final_state=False, cu_seqlens=cu_seqlens
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states), cache=None, output_final_state=False, cu_seqlens=cu_seqlens
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states), cache=None, output_final_state=False, cu_seqlens=cu_seqlens
            )
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            if self.qk_activation == 'silu':
                q, k = F.silu(q), F.silu(k)
            v = F.silu(self.v_proj(hidden_states))

        q, k = map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_k_dim), (q, k))
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)

        if self.qk_activation != 'silu':
            if self.qk_activation == 'relu':
                q, k = q.relu(), k.relu()
            elif self.qk_activation == 'elu':
                q, k = elu_p1(q), elu_p1(k)
            elif self.qk_activation != 'identity':
                raise NotImplementedError

        if self.qk_norm == 'sum':
            q = sum_norm(q).to(q)
            k = sum_norm(k).to(k)

        return q, k, v

    def _make_beta(self, hidden_states, q):
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()  
        else:
            beta = torch.ones_like(q[..., 0])            
        if self.allow_neg_eigval:
            beta = beta * 2.
        return beta

    def _alloc_states_if_needed(self, N, H, K, V, device, dtype, last_state):
        # Pull from cache if present
        if last_state is not None:
            m0 = last_state.get('recurrent_state', None)
            h_state = last_state.get('ivon_h_state', None)
            g_state = last_state.get('ivon_g_state', None)
        else:
            m0 = h_state = g_state = None

        # Allocate zeros on first use (IVON requires all three)
        shape = (N, H, K, V)
        if m0 is None:
            m0 = torch.zeros(shape, device=device, dtype=dtype)
        if h_state is None:
            h_state = torch.zeros(shape, device=device, dtype=dtype)
        if g_state is None:
            g_state = torch.zeros(shape, device=device, dtype=dtype)
        return m0, h_state, g_state

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional["Cache"] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs: "Unpack[Dict]"
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional["Cache"]]:

        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len]. "
                "Arbitrary [B, T, T] masks are not supported."
            )

        batch_size, q_len, _ = hidden_states.shape
        cu_seqlens = kwargs.get('cu_seqlens', None)

        # Unpad for varlen
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        # QKV
        q, k, v = self._proj_qkv(hidden_states, cu_seqlens)

        # Non-silu activations already applied above; apply q/k normalization if requested
        if self.qk_norm == 'l2':
            use_qk_l2norm_in_kernel = True
        else:
            use_qk_l2norm_in_kernel = False

        # Beta (per head scalar by default)
        beta = self._make_beta(hidden_states, q)

        # Shapes for state allocation
        B, T, H, K = k.shape
        V = v.shape[-1]
        N = B if cu_seqlens is None else (len(cu_seqlens) - 1)

        # Fetch last cached states (if any) and allocate if needed
        last_state = None
        if past_key_values is not None and len(past_key_values) > (self.layer_idx or 0):
            last_state = past_key_values[self.layer_idx or 0]

        m0, h_state, g_state = self._alloc_states_if_needed(
            N=N, H=H, K=K, V=V,
            device=hidden_states.device,
            dtype=torch.float32,  # states are float32 for stability
            last_state=last_state
        )

        # scale default (1/sqrt(K))
        scale = (K ** -0.5)

        # run fused recurrent IVON kernel
        o, final_state = fused_recurrent_ivon_delta_rule(
            q=q, k=k, v=v, beta=beta,
            scale=scale,
            initial_state=m0,
            h_state=h_state,
            g_state=g_state,
            lr=self.lr, beta1=self.beta1, beta2=self.beta2,
            weight_decay=self.weight_decay, lam=self.lam,
            rng_seed=self.rng_seed,
            output_final_state=use_cache,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
        )

        # update cache
        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=final_state if use_cache else m0,  # if not caching, m0 has been updated in-place
                ivon_h_state=h_state,                              # updated in-place
                ivon_g_state=g_state,                              # updated in-place
                conv_state=None,                                   # add conv state if you also want to cache it
                layer_idx=self.layer_idx,
                offset=q_len
            )

        # output gating + norm + projection (same as original)
        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)

        # re-pad if varlen
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values
