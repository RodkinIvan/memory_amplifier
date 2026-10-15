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
`swa_2pass`  : the sliding-window transformer run one segment at a time on a key/value cache where every layer runs twice per segment: pass 1 reads the window
               (and stores nothing) and is what the next layer receives, pass 2 re-runs the block on the input plus the gained update of pass 1 and is the only one that
               writes keys and values to the cache. Nothing is carried between segments but the cache. See `TwoPassSWA`.
Removed in E7 (see `config.REMOVED_IN`; the results of these settings stay in results/ as an archive): kind `swa_va` (value adaptation), the `swa_fb` noise option and
tail-recompute path (`kv_cache` false), and output of pass 2 in `swa_2pass`.
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

import config as C


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
    elif kind not in ("gdn", "gdn2"):  # gdn2 = the stock learned decay (the name that appears in run names and wandb)
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
    """Build the model described by the `model:` block of a run config (see configs/*.yaml). A setting that E7 removed (kind `swa_va`, `noise` other than 0, `kv_cache` false,
    `pass_output` 2) stops the build with a message naming the commit that removed it (`config.check_removed_model`): a stored config without such a key means what the code did when it
    was written (`config.CODE_DEFAULTS`), so a run from before E7 must say `kv_cache: true` or `pass_output: 1` explicitly to be built, e.g. to evaluate old weights on the surviving path."""
    C.check_removed_model(mc)
    kind = mc["kind"]
    if kind == "swa_fb":
        return build_swa_fb(vocab_size=vocab_size, d_model=mc["d_model"], n_layers=mc["n_layers"], num_heads=mc["num_heads"],
                            window=mc["window"], segment=mc.get("segment", 0), nonlin=mc.get("nonlin", "tanh"),
                            gain_init=mc.get("gain_init", 0.0), proj_std=mc.get("proj_std", 0.02))
    if kind == "swa_2pass":
        return build_swa_2pass(vocab_size=vocab_size, d_model=mc["d_model"], n_layers=mc["n_layers"], num_heads=mc["num_heads"], window=mc["window"],
                               segment=mc.get("segment", 0), gain_init=mc.get("gain_init", 0.0))
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
    residual path). `tanh` keeps m bounded (fixed points, no blow-up over hundreds of hops); `none` removes only the tanh ("no tanh"):
    the RMSNorm in front of the projection and the whole layer stack between injection and read-out stay, so the loop is not a linear
    system; `gelu` is unbounded. The injection gain is a learned per-channel scale and `gain_init` sets where it starts. It trades two
    things (finished sweep, configs/swa_fb.yaml vs `model.gain_init=0`, see the report): with `gain_init` = 0 the model starts as exactly
    plain SWA and the AR task is learned early in most runs (AR valid accuracy at N = 256 after 2000 steps: 0.47-0.98 in 21 of the 24
    gain-0 runs, at most 0.02 in 3 no-tanh runs of seed 2 at segments 32 / 16 / 8; the first two learned it later, the segment-8 run never did),
    but the optimiser must find the loop from a zero injection and often does not at small segments (about half of the segment-16 and
    segment-8 runs never started it); with the recipe's 0.1 the loop starts reliably, but the memory is a few times the token signal at first, so the AR task
    is learned later the smaller the segment (0.61 / 0.17 / 0.00 for segments 64 / 32 / 16). Gain 1 stalls AR learning for good (loss stuck
    at 2.1 vs 1.2 after 600 steps). The constructor default is 0; the recipe sets 0.1."""

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


class _TailCache:
    """Key/value cache of the last `keep` tokens of every layer, with the interface fla's `Attention` calls
    (`get_seq_length`, `update`). Used to run a window-W layer on one segment at a time: `update` returns the cached keys and
    values followed by the new ones, i.e. the last W-1 tokens plus the segment, so every query of the segment still sees W
    tokens back. (fla's own cache trims to W *after* appending a chunk, which shrinks the window for chunks longer than one
    token, so it is not used.) Tensors are concatenated, never written in place, so gradients flow through the cache.
    Positions are absolute (RoPE offset = tokens seen), so a stream must be shorter than `max_position_embeddings`."""

    def __init__(self, keep: int, n_layers: int):
        self.keep, self.k, self.v, self.seen = keep, [None] * n_layers, [None] * n_layers, [0] * n_layers

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.seen[layer_idx]

    def update(self, attn_state, layer_idx: int, offset: int = 1, cache_kwargs=None, **_) -> dict:
        k_new, v_new = attn_state
        if self.k[layer_idx] is None:
            k_all, v_all = k_new, v_new
        else:
            k_all, v_all = torch.cat([self.k[layer_idx], k_new], dim=1), torch.cat([self.v[layer_idx], v_new], dim=1)
        self.k[layer_idx], self.v[layer_idx] = k_all[:, -self.keep:], v_all[:, -self.keep:]
        self.seen[layer_idx] += k_new.shape[1]
        return {"attn_state": (k_all, v_all)}


class FeedbackSWA(nn.Module):
    """Sliding-window transformer run one segment at a time on a key/value cache, with a SegmentFeedback loop around every layer.

    For segment s and layer i:  inp = x_s + gain * m_i(s-1);  y = block_i(inp);  m_i(s) = loop_i(y).  The block sees only the new segment: the keys and
    values of the last W-1 tokens come from the cache (they are functions of the stored inputs), so attention inside a segment sees exactly what plain
    sliding-window attention would see. m_i(-1) = 0, so the first segment is plain SWA. Information can now travel past the window reach by riding along m,
    one segment per hop. (Before E7 there was a second path that recomputed the block over the last W-1 inputs plus the segment, with the same result up to rounding,
    and a noise option on m; both were removed.)"""

    def __init__(self, base: TransformerForCausalLM, window: int, segment: int, nonlin: str = "tanh", gain_init: float = 0.0, proj_std: float = 0.02):
        super().__init__()
        self.base, self.window, self.segment = base, window, segment
        self.probe_hook = None  # probes only: called as hook(segment, layer, m) -> m on every loop vector, before it is injected
        d = base.config.hidden_size
        self.loops = nn.ModuleList(SegmentFeedback(d, nonlin, gain_init, proj_std) for _ in base.model.layers)

    def forward(self, input_ids: torch.Tensor, **_) -> SimpleNamespace:
        B, T = input_ids.shape
        C_, W = self.segment, self.window
        if T % C_:  # right-pad with PAD (causal, so earlier outputs are unaffected)
            input_ids = F.pad(input_ids, (0, C_ - T % C_), value=0)
        x = self.base.model.embeddings(input_ids)
        layers = self.base.model.layers
        if x.shape[1] > self.base.config.max_position_embeddings:
            raise ValueError(f"the key/value cache uses absolute positions: {x.shape[1]} tokens exceed max_position_embeddings")
        cache = _TailCache(W - 1, len(layers))
        mems = [None] * len(layers)
        outs = []
        for s in range(x.shape[1] // C_):
            h = x[:, s * C_:(s + 1) * C_]
            for i, (layer, loop) in enumerate(zip(layers, self.loops)):
                inp = h if mems[i] is None else loop.inject(h, mems[i])
                y = layer(inp, past_key_values=cache, use_cache=True)[0]
                m = loop(y)
                if self.probe_hook is not None:
                    m = self.probe_hook(s, i, m)
                mems[i] = m
                h = y
            outs.append(h)
        hidden = self.base.model.norm(torch.cat(outs, dim=1)[:, :T])
        return SimpleNamespace(logits=self.base.lm_head(hidden))


class _PeekCache:
    """Read-only view of a `_TailCache` for one layer call (pass 1 of `TwoPassSWA`): the interface fla's `Attention` calls, but `update` returns the cached keys and values
    followed by the new ones without storing anything, and `get_seq_length` (the RoPE offset) is the cache's and does not move. So a layer run through the view sees exactly the
    window the real cache would give it and leaves the cache as it was."""

    def __init__(self, cache: "_TailCache"):
        self.cache = cache

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.cache.get_seq_length(layer_idx)

    def update(self, attn_state, layer_idx: int, offset: int = 1, cache_kwargs=None, **_) -> dict:
        k_new, v_new = attn_state
        if self.cache.k[layer_idx] is None:
            return {"attn_state": (k_new, v_new)}
        return {"attn_state": (torch.cat([self.cache.k[layer_idx], k_new], dim=1), torch.cat([self.cache.v[layer_idx], v_new], dim=1))}


def write_kv(layer, h: torch.Tensor, cache) -> None:
    """Store in `cache` the keys and values that block `layer` computes from the input `h` (pass 2 of `TwoPassSWA`), and nothing else. fla's `Attention.forward` derives them from the
    block's input alone: k = rotary(k_proj(attn_norm(h))) (after `k_norm` with `qk_norm`), v = v_proj(attn_norm(h)), stored flattened over the heads, with the RoPE offset
    `cache.get_seq_length` and written through `cache.update`, so the window trimming is the cache's own. The attention itself, its output projection and the MLP are not run: with
    the output of pass 1 the result of pass 2 is not used, so this is the same function as running the whole block and discarding what it returns (tests/test_models.py)."""
    attn = layer.attn
    x = layer.attn_norm(h)
    k = rearrange(attn.k_proj(x), "... (h d) -> ... h d", d=attn.head_dim)
    v = rearrange(attn.v_proj(x), "... (h d) -> ... h d", d=attn.head_dim)
    if attn.qk_norm:
        k = attn.k_norm(k)
    offset = cache.get_seq_length(attn.layer_idx)
    _, k = attn.rotary(k, k, seqlen_offset=offset, max_seqlen=max(h.shape[1] + offset, attn.max_position_embeddings or 0))  # (rotates k twice: the first result is q's slot, dropped)
    cache.update(attn_state=(k.flatten(-2, -1), v.flatten(-2, -1)), layer_idx=attn.layer_idx, offset=h.shape[1], cache_kwargs=dict(window_size=attn.window_size))


class PassGain(nn.Module):
    """The learned per-channel gain g of one layer of `TwoPassSWA` (a module of its own so that the optimizer finds it in the `loops` group)."""

    def __init__(self, d_model: int, gain_init: float = 0.0):
        super().__init__()
        self.gain = nn.Parameter(torch.full((d_model,), float(gain_init)))


class TwoPassSWA(nn.Module):
    """Sliding-window transformer run one segment at a time on a key/value cache, every layer twice per segment (rehearsal inside the layer).

    For segment s and layer i, with h the layer's input for the segment:
      pass 1 (read):   y1 = block_i(h), attention over the cached window plus the segment itself (causal); nothing is written to the cache. y1 is what the next layer receives.
      pass 2 (write):  the keys and values the same block (same weights) computes from h + g * (y1 - h) are stored (the entries later segments see). Nothing else of pass 2 is
                       computed: its output would not be used, so only the key and value projections run (`write_kv`).
    Only the update y1 - h is fed back, so with g = 0 pass 2 writes what pass 1 read and the model is plain sliding-window attention. Both passes use the same RoPE positions
    (the view of pass 1 does not advance the cache). Nothing but the cache crosses a segment boundary. (Before E7 the next layer could receive pass 2's output instead,
    `pass_output` 2; it trained much worse and was removed.)"""

    def __init__(self, base: TransformerForCausalLM, window: int, segment: int, gain_init: float = 0.0):
        super().__init__()
        self.base, self.window, self.segment = base, window, segment
        self.cache_hook = None  # probes only: called as hook(segment, cache) after every segment; may read or replace cache.k[i] / cache.v[i] (the window the next segment sees)
        d = base.config.hidden_size
        self.loops = nn.ModuleList(PassGain(d, gain_init) for _ in base.model.layers)  # named `loops` for the optimizer's loop group

    def run_segment(self, h: torch.Tensor, cache: "_TailCache") -> tuple[torch.Tensor, list[torch.Tensor]]:
        """One segment through every layer on `cache` (which is advanced): the input embeddings h [B, C, d] -> (output of the last layer, the pass-2 inputs h2 of every layer
        [B, C, d] each). The h2 are what pass 2 stored the keys and values of: the window the next segment reads is a function of the last W-1 of them."""
        h2s = []
        for layer, loop in zip(self.base.model.layers, self.loops):
            y1 = layer(h, past_key_values=_PeekCache(cache), use_cache=True)[0]
            h2 = h + (loop.gain * (y1 - h).float()).to(h.dtype)
            write_kv(layer, h2, cache)  # pass 2: only the keys and values it writes to the cache count (E6: its attention and MLP are not run)
            h2s.append(h2)
            h = y1
        return h, h2s

    def forward(self, input_ids: torch.Tensor, **_) -> SimpleNamespace:
        B, T = input_ids.shape
        C_, W = self.segment, self.window
        if T % C_:  # right-pad with PAD (causal, so earlier outputs are unaffected)
            input_ids = F.pad(input_ids, (0, C_ - T % C_), value=0)
        x = self.base.model.embeddings(input_ids)
        if x.shape[1] > self.base.config.max_position_embeddings:
            raise ValueError(f"the key/value cache uses absolute positions: {x.shape[1]} tokens exceed max_position_embeddings")
        cache = _TailCache(W - 1, len(self.base.model.layers))
        outs = []
        for s in range(x.shape[1] // C_):
            h, _ = self.run_segment(x[:, s * C_:(s + 1) * C_], cache)
            outs.append(h)
            if self.cache_hook is not None:
                self.cache_hook(s, cache)
        hidden = self.base.model.norm(torch.cat(outs, dim=1)[:, :T])
        return SimpleNamespace(logits=self.base.lm_head(hidden))


def build_swa_2pass(*, vocab_size: int, d_model: int = 128, n_layers: int = 2, num_heads: int = 4, window: int, segment: int = 0,
                    gain_init: float = 0.0) -> TwoPassSWA:
    """`segment` = 0 means one segment per window."""
    segment = segment or window
    if segment < 1:
        raise ValueError(f"segment must be >= 1, got {segment}")
    base = build_swa(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, num_heads=num_heads, window=window)
    return TwoPassSWA(base, window=window, segment=segment, gain_init=gain_init)


def build_swa_fb(*, vocab_size: int, d_model: int = 128, n_layers: int = 2, num_heads: int = 4, window: int,
                 segment: int = 0, nonlin: str = "tanh", gain_init: float = 0.0, proj_std: float = 0.02) -> FeedbackSWA:
    """`segment` = 0 means one segment per window. Needs 1 <= segment; any segment length works (the cache spans segments)."""
    segment = segment or window
    if segment < 1:
        raise ValueError(f"segment must be >= 1, got {segment}")
    base = build_swa(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, num_heads=num_heads, window=window)
    return FeedbackSWA(base, window=window, segment=segment, nonlin=nonlin, gain_init=gain_init, proj_std=proj_std)
