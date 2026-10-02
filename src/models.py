"""Small fla-based language models for passkey_swar.

`gdn_fixed`  : GatedDeltaNet whose decay is a constant alpha (Exp 1). alpha=1.0 is the plain delta rule.
`gdn`        : stock fla GatedDeltaNet with learned, data-dependent decay (Exp 2/3);
               `allow_neg_eigval=True` gives the [-1, 1] eigenvalue variant of Grazzi et al. (2025).
Everything except the mixer (embeddings, norms, MLP, head) is stock fla `GatedDeltaNetForCausalLM`.
`swa`        : sliding-window attention baseline, stock fla `TransformerForCausalLM` with `window_size=W`. Information
               travels at most `n_layers * (W - 1)` tokens, so anything further back is unreachable by construction.
"""
from __future__ import annotations

import math

import torch
from einops import rearrange
from fla.layers import GatedDeltaNet
from fla.models import GatedDeltaNetConfig, GatedDeltaNetForCausalLM, TransformerConfig, TransformerForCausalLM
from fla.ops.gated_delta_rule import chunk_gated_delta_rule


class FixedDecayGDN(GatedDeltaNet):
    """GatedDeltaNet with S_t = alpha * S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T, alpha a constant.

    Training / full-sequence evaluation only (no cache, no padding mask). The learned-decay parameters of the
    parent (`a_proj`, `A_log`, `dt_bias`) are removed so nothing unused is left in the optimizer.
    """

    def __init__(self, *args, alpha: float, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = float(alpha)
        del self.a_proj, self.A_log, self.dt_bias

    def forward(self, hidden_states, attention_mask=None, past_key_values=None, use_cache=False,
                output_attentions=False, **kwargs):
        if attention_mask is not None or use_cache or past_key_values is not None:
            raise NotImplementedError("FixedDecayGDN supports only full-sequence forward without cache/padding")
        x = hidden_states
        if self.use_short_conv:
            q = self.q_conv1d(x=self.q_proj(x))[0]
            k = self.k_conv1d(x=self.k_proj(x))[0]
            v = self.v_conv1d(x=self.v_proj(x))[0]
        else:
            q, k, v = (torch.nn.functional.silu(p(x)) for p in (self.q_proj, self.k_proj, self.v_proj))
        q, k = (rearrange(t, "... (h d) -> ... h d", d=self.head_k_dim) for t in (q, k))
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)
        g = torch.full(x.shape[:2] + (self.num_v_heads,), math.log(self.alpha), dtype=torch.float32, device=x.device)
        o, _ = chunk_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=self.b_proj(x),
            use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=True,
            allow_neg_eigval=self.allow_neg_eigval, state_v_first=True,
        )
        if self.use_gate:
            o = self.o_norm(o, rearrange(self.g_proj(x), "... (h d) -> ... h d", d=self.head_v_dim))
        else:
            o = self.o_norm(o)
        return self.o_proj(rearrange(o, "b t h d -> b t (h d)")), None, past_key_values


def build_model(kind: str, *, vocab_size: int, d_model: int = 128, n_layers: int = 2, num_heads: int = 1,
                head_dim: int = 32, expand_v: float = 1.0, alpha: float | None = None,
                allow_neg_eigval: bool = False, norm_eps: float = 1e-5) -> GatedDeltaNetForCausalLM:
    """`norm_eps` is the eps of GDN's output RMSNorm (fla default 1e-5); see the Exp 1 plan for why it matters."""
    cfg = GatedDeltaNetConfig(
        hidden_size=d_model, num_hidden_layers=n_layers, num_heads=num_heads, head_dim=head_dim, expand_v=expand_v,
        vocab_size=vocab_size, allow_neg_eigval=allow_neg_eigval, norm_eps=norm_eps, tie_word_embeddings=True,
        fuse_cross_entropy=False, use_cache=False,
    )
    model = GatedDeltaNetForCausalLM(cfg)
    if (kind == "gdn_fixed") != (alpha is not None):
        raise ValueError("alpha must be given for (and only for) kind='gdn_fixed'")
    if kind == "gdn_fixed":
        for i, layer in enumerate(model.model.layers):
            layer.attn = FixedDecayGDN(
                mode="chunk", hidden_size=d_model, expand_v=expand_v, head_dim=head_dim, num_heads=num_heads,
                allow_neg_eigval=allow_neg_eigval, layer_idx=i, norm_eps=norm_eps, alpha=alpha)
            for m in layer.attn.modules():  # same init fla applies to the children of a stock GatedDeltaNet
                if m is not layer.attn:
                    model._init_weights(m)
    elif kind != "gdn":
        raise ValueError(f"unknown model kind {kind!r}")
    return model


def swa_reach(window: int, n_layers: int) -> int:
    """Longest distance (in tokens) over which a sliding-window transformer can pass information."""
    return n_layers * (window - 1)


def build_swa(*, vocab_size: int, d_model: int = 128, n_layers: int = 2, num_heads: int = 4, window: int) -> TransformerForCausalLM:
    """Sliding-window attention: every layer attends to the current token and the `window - 1` tokens before it (RoPE)."""
    if d_model % num_heads:
        raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    cfg = TransformerConfig(
        hidden_size=d_model, num_hidden_layers=n_layers, num_heads=num_heads, window_size=window, vocab_size=vocab_size,
        max_position_embeddings=1 << 17, tie_word_embeddings=True, fuse_cross_entropy=False, use_cache=False,
    )
    return TransformerForCausalLM(cfg)


def build_from_config(mc: dict, vocab_size: int):
    """Build the model described by the `model:` block of a run config (see configs/*.yaml)."""
    kind = mc["kind"]
    if kind == "swa":
        return build_swa(vocab_size=vocab_size, d_model=mc["d_model"], n_layers=mc["n_layers"],
                         num_heads=mc["num_heads"], window=mc["window"])
    return build_model(kind, vocab_size=vocab_size, d_model=mc["d_model"], n_layers=mc["n_layers"],
                       num_heads=mc["num_heads"], head_dim=mc["head_dim"], expand_v=mc["expand_v"],
                       alpha=mc["alpha"] if kind == "gdn_fixed" else None,
                       allow_neg_eigval=mc["allow_neg_eigval"], norm_eps=mc["norm_eps"])
