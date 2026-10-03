"""Small fla-based language models for passkey_swar.

`gdn_fixed`  : GatedDeltaNet whose decay is a constant alpha (Exp 1). alpha=1.0 is the plain delta rule.
`gdn`        : stock fla GatedDeltaNet with learned, data-dependent decay (Exp 2/3);
               `allow_neg_eigval=True` gives the [-1, 1] eigenvalue variant of Grazzi et al. (2025).
Everything except the mixer (embeddings, norms, MLP, head) is stock fla `GatedDeltaNetForCausalLM`.
`swa`        : sliding-window attention baseline, stock fla `TransformerForCausalLM` with `window_size=W`. Information
               travels at most `n_layers * (W - 1)` tokens, so anything further back is unreachable by construction.
`swa_fb`     : the same sliding-window transformer plus a *segment feedback loop* ("memory amplification"): once per
               segment every layer turns its output into one vector, which is added to that layer's input at every
               token of the next segment. See `FeedbackSWA`.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
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
    if kind == "swa_fb":
        return build_swa_fb(vocab_size=vocab_size, d_model=mc["d_model"], n_layers=mc["n_layers"], num_heads=mc["num_heads"],
                            window=mc["window"], segment=mc.get("segment", 0), nonlin=mc.get("nonlin", "tanh"),
                            noise=mc.get("noise", 0.0), gain_init=mc.get("gain_init", 0.0), proj_std=mc.get("proj_std", 0.02))
    if kind == "swa":
        return build_swa(vocab_size=vocab_size, d_model=mc["d_model"], n_layers=mc["n_layers"],
                         num_heads=mc["num_heads"], window=mc["window"])
    return build_model(kind, vocab_size=vocab_size, d_model=mc["d_model"], n_layers=mc["n_layers"],
                       num_heads=mc["num_heads"], head_dim=mc["head_dim"], expand_v=mc["expand_v"],
                       alpha=mc["alpha"] if kind == "gdn_fixed" else None,
                       allow_neg_eigval=mc["allow_neg_eigval"], norm_eps=mc["norm_eps"])


class SegmentFeedback(nn.Module):
    """One feedback loop: m = mean over the segment of nonlin(proj(norm(y))), added to the next segment's layer input.

    y are the layer's outputs for the segment (the residual stream after the block, so the memory can ride along the
    residual path). `tanh` keeps m bounded (fixed points, no blow-up over hundreds of hops); `none` is the linear
    control; `gelu` is unbounded. The injection gain is a learned per-channel scale. With `gain_init` = 0 (the constructor
    default) the model starts as exactly plain SWA. The training recipe (configs/swa_fb.yaml) uses 0.1, which the loop needs
    to start learning at all, and it costs time: the injected memory is a few times the token signal, so the AR task is learned
    later the smaller the segment (AR valid accuracy at step 2000: 0.62 / 0.17 / 0.00 for segments 64 / 32 / 16). Gain 1 stalls
    AR learning for good (loss stuck at 2.1 vs 1.2 after 600 steps)."""

    def __init__(self, d_model: int, nonlin: str = "tanh", gain_init: float = 0.0, proj_std: float = 0.02):
        super().__init__()
        if nonlin not in ("tanh", "gelu", "none"):
            raise ValueError(f"unknown nonlin {nonlin!r}")
        self.nonlin = nonlin
        self.norm = nn.RMSNorm(d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.gain = nn.Parameter(torch.full((d_model,), float(gain_init)))
        nn.init.normal_(self.proj.weight, std=proj_std)
        nn.init.zeros_(self.proj.bias)

    def forward(self, y: torch.Tensor) -> torch.Tensor:  # (B, C, d) -> (B, d)
        with torch.autocast(y.device.type, enabled=False):  # fp32: bf16 rounding would erase a small recurrent signal
            z = self.proj(self.norm(y.float()))
            z = torch.tanh(z) if self.nonlin == "tanh" else F.gelu(z) if self.nonlin == "gelu" else z
            return z.mean(dim=1)

    def inject(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:  # add the memory to every token
        return x + self.gain * m[:, None, :].to(x.dtype)


class FeedbackSWA(nn.Module):
    """Sliding-window transformer run one segment at a time, with a SegmentFeedback loop around every layer.

    For segment s and layer i:  inp = x_s + gain * m_i(s-1);  y = block_i([tail ; inp])[segment part];  m_i(s) = loop_i(y).
    `tail` holds the last window-1 layer *inputs* of the preceding segments, so attention inside a segment sees exactly
    what plain sliding-window attention would see. m_i(-1) = 0, so the first segment is plain SWA. Information can now
    travel past the window reach by riding along m, one segment per hop."""

    def __init__(self, base: TransformerForCausalLM, window: int, segment: int, nonlin: str = "tanh", noise_std: float = 0.0,
                 gain_init: float = 0.0, proj_std: float = 0.02):
        super().__init__()
        self.base, self.window, self.segment, self.noise_std = base, window, segment, noise_std
        d = base.config.hidden_size
        self.loops = nn.ModuleList(SegmentFeedback(d, nonlin, gain_init, proj_std) for _ in base.model.layers)

    def forward(self, input_ids: torch.Tensor, **_) -> SimpleNamespace:
        B, T = input_ids.shape
        C, W = self.segment, self.window
        if T % C:  # right-pad with PAD (causal, so earlier outputs are unaffected)
            input_ids = F.pad(input_ids, (0, C - T % C), value=0)
        x = self.base.model.embeddings(input_ids)
        layers = self.base.model.layers
        tails = [None] * len(layers)
        mems = [None] * len(layers)
        outs = []
        for s in range(x.shape[1] // C):
            h = x[:, s * C:(s + 1) * C]
            for i, (layer, loop) in enumerate(zip(layers, self.loops)):
                inp = h if mems[i] is None else loop.inject(h, mems[i])
                ctx = inp if tails[i] is None else torch.cat([tails[i], inp], dim=1)
                y = layer(ctx)[0][:, -C:]
                tails[i] = ctx[:, -(W - 1):]
                m = loop(y)
                if self.training and self.noise_std > 0:
                    m = m + self.noise_std * torch.randn_like(m)
                mems[i] = m
                h = y
            outs.append(h)
        hidden = self.base.model.norm(torch.cat(outs, dim=1)[:, :T])
        return SimpleNamespace(logits=self.base.lm_head(hidden))


def build_swa_fb(*, vocab_size: int, d_model: int = 128, n_layers: int = 2, num_heads: int = 4, window: int,
                 segment: int = 0, nonlin: str = "tanh", noise: float = 0.0, gain_init: float = 0.0, proj_std: float = 0.02) -> FeedbackSWA:
    """`segment` = 0 means one segment per window. Needs 1 <= segment; any segment length works (the tail spans segments)."""
    segment = segment or window
    if segment < 1:
        raise ValueError(f"segment must be >= 1, got {segment}")
    base = build_swa(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, num_heads=num_heads, window=window)
    return FeedbackSWA(base, window=window, segment=segment, nonlin=nonlin, noise_std=noise, gain_init=gain_init, proj_std=proj_std)
