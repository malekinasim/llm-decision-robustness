# margins.py
# Utilities to compute per-option and per-class margins from logits.
# All functions assume the last dimension corresponds to classes/options.

from typing import Optional
import torch
from hooks import *
import tuned
# ---------------------------------------------------------------------------
# Top-1 vs. Top-2 margin (single example)
# ---------------------------------------------------------------------------
def top1_top2_margin_1d(z: torch.Tensor) -> float:
    """
    Compute (top1 - top2) from a 1D logits vector.

    Args:
        z: 1D tensor of shape [V] (V = number of classes/options).

    Returns:
        float: top1 - top2 margin. If V < 2, returns NaN.
    """
    if z.ndim != 1:
        raise ValueError(f"Expected 1D logits, got shape {tuple(z.shape)}")
    if z.numel() < 2:
        return float("nan")
    v = torch.topk(z, k=2).values
    return float((v[0] - v[1]).item())


# ---------------------------------------------------------------------------
# Top-1 vs. Top-2 margin (batched)
# ---------------------------------------------------------------------------
def top1_top2_margin_batched(z: torch.Tensor) -> torch.Tensor:
    """
    Compute (top1 - top2) along the last dimension for a batched logits tensor.

    Args:
        z: Tensor of shape [..., V] where V is the number of classes/options.

    Returns:
        Tensor of shape [...] with top1 - top2 margins.
        If V < 2, returns a tensor filled with NaNs.
    """
    V = z.shape[-1]
    if V < 2:
        return torch.full(z.shape[:-1], float("nan"), dtype=z.dtype, device=z.device)
    # topk(..., k=2) over the last dim: returns (..., 2)
    vals, _ = torch.topk(z, k=2, dim=-1)
    return (vals[..., 0] - vals[..., 1])


# ---------------------------------------------------------------------------
# Gold margin over full vocabulary/classes (single example)
# ---------------------------------------------------------------------------
def gold_margin_from_logits_1d(z: torch.Tensor, gold_id: int) -> Optional[float]:
    """
    Gold margin = logits[gold] - max(logits of all other classes).

    Args:
        z: 1D logits tensor of shape [V].
        gold_id: integer index of the gold/true class in [0, V).

    Returns:
        Optional[float]: margin value as Python float, or None if there is no competitor (V <= 1).
    """
    if z.ndim != 1:
        raise ValueError(f"Expected 1D logits, got shape {tuple(z.shape)}")
    V = z.shape[0]
    if not (0 <= gold_id < V):
        raise IndexError(f"gold_id {gold_id} is out of range [0, {V})")
    if V <= 1:
        return None  # no competitor

    # Build a boolean mask: True for competitors, False for gold.
    mask = torch.ones(V, dtype=torch.bool, device=z.device)
    mask[gold_id] = False

    # Replace gold position with a very negative value so it never wins the max.
    very_neg = torch.finfo(z.dtype).min
    z_comp = z.masked_fill(~mask, very_neg)

    max_comp = torch.max(z_comp)
    margin = (z[gold_id] - max_comp).item()
    return float(margin)


# ---------------------------------------------------------------------------
# Gold margin over full vocabulary/classes (batched)
# ---------------------------------------------------------------------------
def gold_margin_from_logits_batched(z: torch.Tensor, gold_ids: torch.Tensor) -> torch.Tensor:
    """
    Batched version of gold margin over the last dimension.

    Args:
        z: Tensor of shape [..., V] (logits).
        gold_ids: Long tensor of shape [...] with gold indices aligned to z's batch dims.

    Returns:
        Tensor of shape [...] with margins. If V < 2, returns NaNs.
    """
    V = z.shape[-1]
    if V < 2:
        return torch.full(z.shape[:-1], float("nan"), dtype=z.dtype, device=z.device)

    # Build a mask: True for competitors, False where index == gold.
    ar = torch.arange(V, device=z.device).view(*([1] * (z.ndim - 1)), V)
    mask = (ar != gold_ids.unsqueeze(-1))  # shape [..., V]

    very_neg = torch.finfo(z.dtype).min
    z_comp = z.masked_fill(~mask, very_neg)
    max_comp, _ = z_comp.max(dim=-1)

    gold_vals = z.gather(-1, gold_ids.unsqueeze(-1)).squeeze(-1)
    return gold_vals - max_comp


# ---------------------------------------------------------------------------
# Gold margin restricted to options (single example)
# ---------------------------------------------------------------------------
def gold_margin_opts_1d(z: torch.Tensor, gidx: int) -> float:
    """
    Gold margin over a restricted options vector (e.g., MCQ options only).

    Args:
        z: 1D tensor of shape [K] (K = number of options).
        gidx: integer index of the gold option in [0, K).

    Returns:
        float: z[gidx] - max(z[others]). If K < 2, returns NaN.
    """
    if z.ndim != 1:
        raise ValueError(f"Expected 1D logits, got shape {tuple(z.shape)}")
    K = z.numel()
    if not (0 <= gidx < K):
        raise IndexError(f"gidx {gidx} is out of range [0, {K})")
    if K < 2:
        return float("nan")

    # Mask-based variant—avoid empty-slice issues when gidx is at edges.
    mask = torch.ones_like(z, dtype=torch.bool)
    mask[gidx] = False
    rival = torch.max(z[mask])
    return float((z[gidx] - rival).item())


# ---------------------------------------------------------------------------
# Gold margin restricted to options (batched)
# ---------------------------------------------------------------------------
def gold_margin_opts_batched(z: torch.Tensor, gidx: torch.Tensor) -> torch.Tensor:
    """
    Batched gold margin over a restricted options tensor.

    Args:
        z: Tensor of shape [..., K] (K = number of options).
        gidx: Long tensor of shape [...] with gold indices aligned to z's batch dims.

    Returns:
        Tensor of shape [...] with margins. If K < 2, returns NaNs.
    """
    K = z.shape[-1]
    if K < 2:
        return torch.full(z.shape[:-1], float("nan"), dtype=z.dtype, device=z.device)

    ar = torch.arange(K, device=z.device).view(*([1] * (z.ndim - 1)), K)
    mask = (ar != gidx.unsqueeze(-1))  # True for rivals, False for gold

    very_neg = torch.finfo(z.dtype).min
    z_comp = z.masked_fill(~mask, very_neg)
    rival, _ = z_comp.max(dim=-1)

    gold_vals = z.gather(-1, gidx.unsqueeze(-1)).squeeze(-1)
    return gold_vals - rival




@torch.no_grad()
def compute_margins_per_layer_logits(
    model, tokenizer, text=None, outputs=None, pos=-1,
    ln_f_mode="last_only", skip_embedding=False,
    gold_text=None,                   # full-vocab gold (single-token)
    options: "list[str]|None" = None, # MCQ options (single-token)
    gold_option: "str|None" = None,   # gold among options
    tuned: "TunedDiag|None" = None
):
   # IDs
   # Precompute ids
    gold_id = None
    if gold_text and gold_text.strip():
        gids = tokenizer(gold_text, add_special_tokens=False)["input_ids"]
        if len(gids) != 1:
            raise ValueError(f"gold {gold_text!r} must be single-token.")
        gold_id = gids[0]

    option_ids, gold_opt_idx = None, None
    if options:
        option_ids = []
        for o in options:
            ids = tokenizer(o, add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                raise ValueError(f"Option {o!r} must be single-token.")
            option_ids.append(ids[0])
        if gold_option is not None:
            try:
                gold_opt_idx = options.index(gold_option)
            except ValueError:
                raise ValueError("gold_option must be one of options")

    # One forward pass -> two projections
   
    Z_full = layerwise_logits_for_pos(model,tokenizer,text=text,
                                           outputs=outputs,pos=pos,
                                           ln_f_mode=ln_f_mode,skip_embedding=skip_embedding
                                           , option_ids=None, tuned=tuned)
    Z_opts = None
    if option_ids is not None:
        Z_opts = layerwise_logits_for_pos(model,tokenizer,text=text,
                                           outputs=outputs,pos=pos,
                                           ln_f_mode=ln_f_mode,skip_embedding=skip_embedding,
                                             option_ids=option_ids, 
                                             tuned=tuned)

    def pack(Zs, kind):
        def compute_list(t_list):
            res = {
                f"top1_top2_{kind}": [top1_top2_margin_1d(z) for z in t_list]
            }
            if kind == "full" and gold_id is not None:
                res["gold_full"] = [gold_margin_from_logits_1d(z, gold_id) for z in t_list]
            if kind == "opts" and gold_opt_idx is not None:
                res["gold_opts"] = [gold_margin_opts_1d(z, gold_opt_idx) for z in t_list]
            return res

        out = {}
        if isinstance(Zs, dict):  # raw/tuned
            out["raw"] = compute_list(Zs["raw"])
            out["tuned"] = compute_list(Zs["tuned"])
        else:
            out["raw"] = compute_list(Zs)
        return out

    out = {"full": pack(Z_full, "full")}
    if Z_opts is not None:
        out["opts"] = pack(Z_opts, "opts")
    return out


