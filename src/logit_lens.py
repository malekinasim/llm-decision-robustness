# margins.py
# Utilities to compute per-option and per-class margins from logits.
# All functions assume the last dimension corresponds to classes/options.

from typing import Optional
import torch
from src.hooks import *
from src.tuned import TunedDiag
from typing import List, Dict, Optional, Any
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





@torch.no_grad()
def mcq_alllayer_scores(
    model,
    tokenizer,
    prompt_text: str,
    options: List[str],
    *,
    gold_opt: Optional[str] = None,
    pos: int = -1,                     # -1 => last non-pad
    ln_f_mode: str = "last_only",      # {"none","last_only","all"}
    skip_embedding: bool = True,
    tuned: Optional["TunedDiag"] = None,
    outputs: Optional[Any] = None,     # reuse a precomputed forward if provided
) -> Dict[str, Any]:
    """
    Compute per-layer MCQ scores by projecting hidden states onto selected option columns of W_U.

    Returns:
      dict with keys:
        - "raw":   (scores_per_layer, winners_per_layer, top1_top2_margins, gold_margins)
        - "tuned": same tuple or None
        - "outputs": the HF model output object (with hidden_states)
    Notes:
      * scores_per_layer: List[Dict[option_str, float]], one dict per layer
      * winners_per_layer: List[str]
      * top1_top2_margins: List[float]  (NaN if only one option)
      * gold_margins:     List[float]   (only if gold_opt provided; empty otherwise)
    """

    # --------- device / eval / no dropout -----------
    model.eval()
    device = next(model.parameters()).device

    # --------- tokenize prompt with padding mask ----
    enc = tokenizer(prompt_text, return_tensors="pt", padding=False, truncation=False)
    enc = {k: v.to(device) for k, v in enc.items()}

    # Ensure one forward pass with hidden states (or reuse provided)
    if outputs is None:
        outputs = model(**enc, output_hidden_states=True, use_cache=False)
    else:
        # If caller passed outputs, they must contain hidden_states
        if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
            raise ValueError("`outputs` was provided but has no hidden_states. Re-run forward with output_hidden_states=True.")

    hs = outputs.hidden_states  # tuple: [emb, h1, ..., hL]
    if not isinstance(hs, (tuple, list)) or len(hs) < 2:
        raise RuntimeError("hidden_states is missing or malformed.")

    # --------- locate ln_f (if any) -----------------
    # Many decoder-only LMs expose final layer norm differently.
    ln_f = None
    for cand in ("transformer.ln_f", "model.norm", "model.decoder.norm"):
        mod = model
        try:
            for part in cand.split("."):
                mod = getattr(mod, part)
            ln_f = mod
            break
        except AttributeError:
            ln_f = None
    have_ln_f = ln_f is not None

    # --------- get W_U and option columns -----------
    # W_U: [hidden_dim, vocab_size]
    W_U = getattr(model, "lm_head", None)
    if W_U is None or not hasattr(W_U, "weight"):
        raise RuntimeError("model.lm_head.weight not found; cannot build readout.")
    W_U = W_U.weight.T.contiguous()

    # Tokenize options; enforce single-token options
    if not options:
        raise ValueError("`options` must be a non-empty list of candidate answers.")
    opt_ids: List[int] = []
    for o in options:
        ids = tokenizer(o, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise ValueError(f"Option {o!r} must be single-token; got tokens={ids}")
        opt_ids.append(ids[0])

    opt_idx = torch.as_tensor(opt_ids, device=W_U.device, dtype=torch.long)
    # WU_opts: [hidden_dim, num_options]
    WU_opts = torch.index_select(W_U, dim=1, index=opt_idx)

    # --------- pick the position (last non-pad) -----
    # Using the prompt only (no options appended): take last non-pad token
    # Works regardless of tokenizer.padding_side (we didn't pad, but robust anyway)
    ids = enc["input_ids"]  # [1, T]
    T = ids.shape[1]
    if pos == -1:
        # last non-pad index
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            last_idx = torch.tensor([T - 1], device=device, dtype=torch.long)
        else:
            ar = torch.arange(T, device=device).unsqueeze(0).expand_as(ids)
            mask = (ids != pad_id)
            last_idx = (ar * mask).max(dim=1).values  # [1]
    else:
        if not (-T <= pos < T):
            raise IndexError(f"pos={pos} out of bounds for sequence length {T}")
        last_idx = torch.tensor([pos % T], device=device, dtype=torch.long)

    # --------- helper: apply ln_f and project --------
    # i: layer index into hs
    # x: hidden [D]
    L_total = len(hs)            # includes embedding at idx 0
    L_last = L_total - 1         # index of last layer (transformer block L)
    start = 1 if skip_embedding else 0

    def project_layer_hidden(x: torch.Tensor, i: int) -> torch.Tensor:
        """
        Apply ln_f according to ln_f_mode and project to option logits.
        x: [D]
        returns: [num_options]  (float32/float16 depending on model)
        """
        if have_ln_f and (
            (ln_f_mode == "last_only" and i == L_last) or
            (ln_f_mode == "all")
        ):
            x = ln_f(x)
        # [D] @ [D, num_options] -> [num_options]
        return x @ WU_opts

    # --------- containers ----------
    raw_scores:   List[Dict[str, float]] = []
    raw_winners:  List[str] = []
    raw_m12:      List[float] = []
    raw_gold:     List[float] = []

    tuned_scores:  List[Dict[str, float]] = []
    tuned_winners: List[str] = []
    tuned_m12:     List[float] = []
    tuned_gold:    List[float] = []

    # --------- per-layer loop -------
    # hs[i] has shape [B=1, T, D]
    for i in range(start, L_total):
        H = hs[i]
        x = H[0, last_idx.item()]   # [D]
        # RAW
        zr = project_layer_hidden(x, i)  # [num_options]
        # scores dict for user readability (stable order as given in options)
        s_raw = {opt: float(zr[j].item()) for j, opt in enumerate(options)}
        raw_scores.append(s_raw)

        # winner + top1-top2 margin
        # faster than sorting full dict: just use torch ops
        vals_r, idx_r = torch.sort(zr, descending=True)
        winner_r = options[int(idx_r[0].item())]
        raw_winners.append(winner_r)
        if zr.numel() >= 2:
            raw_m12.append(float((vals_r[0] - vals_r[1]).item()))
        else:
            raw_m12.append(float("nan"))

        # gold margin among options
        if gold_opt is not None:
            try:
                gidx = options.index(gold_opt)
                if zr.numel() >= 2:
                    # max of competitors
                    # set gold to very negative to ignore it in the max
                    very_neg = torch.finfo(zr.dtype).min
                    comp = zr.clone()
                    comp[gidx] = very_neg
                    best_comp = comp.max()
                    raw_gold.append(float((zr[gidx] - best_comp).item()))
                else:
                    raw_gold.append(float("nan"))
            except ValueError:
                # gold_opt not in options -> skip
                pass

        # TUNED
        if tuned is not None:
            # expect TunedDiag API: apply_x(i, x) for hidden; alpha(i) for scalar logit scale
            xt = tuned.apply_x(i, x) if hasattr(tuned, "apply_x") else tuned.apply(i, x)
            zt = project_layer_hidden(xt, i)
            a_i = tuned.alpha(i) if hasattr(tuned, "alpha") else None
            if a_i is not None:
                zt = zt * float(a_i)

            s_t = {opt: float(zt[j].item()) for j, opt in enumerate(options)}
            tuned_scores.append(s_t)

            vals_t, idx_t = torch.sort(zt, descending=True)
            winner_t = options[int(idx_t[0].item())]
            tuned_winners.append(winner_t)
            if zt.numel() >= 2:
                tuned_m12.append(float((vals_t[0] - vals_t[1]).item()))
            else:
                tuned_m12.append(float("nan"))

            if gold_opt is not None:
                try:
                    gidx = options.index(gold_opt)
                    if zt.numel() >= 2:
                        very_neg = torch.finfo(zt.dtype).min
                        comp_t = zt.clone()
                        comp_t[gidx] = very_neg
                        best_comp_t = comp_t.max()
                        tuned_gold.append(float((zt[gidx] - best_comp_t).item()))
                    else:
                        tuned_gold.append(float("nan"))
                except ValueError:
                    pass

    return {
        "raw":   (raw_scores, raw_winners, raw_m12, raw_gold),
        "tuned": (tuned_scores, tuned_winners, tuned_m12, tuned_gold) if tuned is not None else None,
        "outputs": outputs
    }

def early_decision_layer(
    res: Dict[str, List],
    margin_thresh: float = 0.0,
    use_tuned: bool = False,     # False -> RAW, True -> TUNED (if available)
    use_gold: bool = False,      # False -> top1–top2, True -> gold-margin (needs res[series][3])
    persist_k: int = 1,          # how many consecutive layers must satisfy the condition
    require_final_lock: bool = True,  # if True, winners in the window must match final (last-layer) winner
    debug: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Find the earliest layer i at which the model's decision is 'locked in' according to:
      - winner stability for a window of length persist_k
      - margin >= margin_thresh for all layers in that window
      - (optionally) winners in the window equal the final winner (last layer)

    Args:
        res: dict with keys "raw" (and optionally "tuned").
             res[series] is a list/tuple where:
               [1] winners_per_layer: List[Any] of length L
               [2] margins_top1:     List[Optional[float]] of length L  (top1 - top2)
               [3] gold_margins:     List[Optional[float]] of length L  (optional)
        margin_thresh: minimum required margin at each layer in the window
        use_tuned: pick "tuned" series if available, otherwise "raw"
        use_gold: use gold margins if available (res[series][3]); else use top1-top2 margins (res[series][2])
        persist_k: number of consecutive layers to check from i onward
        require_final_lock: if True, winners[j] must equal final winner for all j in the window
        debug: print reasons for rejections (useful for diagnosing)

    Returns:
        dict with:
          - idx: earliest index i meeting criteria
          - winner_final: winner at last layer
          - winners_window: winners[i : i+persist_k]
          - margins_window: margins[i : i+persist_k]
          - margin_at_i: margins[i]
          - series: "raw" or "tuned"
          - metric: "answer" or "top1_top2"
          - persist_k, threshold
        or None if no such layer exists.
    """
    # ---------- choose series ----------
    series = "tuned" if use_tuned and (res.get("tuned") is not None) else "raw"
    if series not in res:
        raise ValueError(f"[early_decision_layer_v2] series '{series}' not found in res keys {list(res.keys())}")

    if not isinstance(res[series], (list, tuple)) or len(res[series]) < 3:
        raise ValueError(f"[early_decision_layer_v2] res[{series}] must be list/tuple with at least 3 elements.")

    winners = res[series][1]
    margins_top1 = res[series][2]
    gold_margins = res[series][3] if len(res[series]) > 3 else None

    # ---------- basic validation ----------
    if not isinstance(winners, list) or not isinstance(margins_top1, list):
        raise ValueError("[early_decision_layer_v2] winners and margins must be lists.")

    L = len(winners)
    if L == 0 or len(margins_top1) != L:
        raise ValueError(f"[early_decision_layer_v2] invalid lengths: L={L}, len(margins_top1)={len(margins_top1)}")

    if use_gold:
        if gold_margins is None or len(gold_margins) != L:
            if debug:
                print("[early_decision_layer_v2] gold margins requested but unavailable or wrong length; returning None.")
            return None
        margins = gold_margins
        metric = "answer"
    else:
        margins = margins_top1
        metric = "top1_top2"

    if persist_k < 1:
        raise ValueError("[early_decision_layer_v2] persist_k must be >= 1")

    # ---------- final winner ----------
    final_w = winners[-1]
    if debug:
        print(f"[EDL] series={series}, metric={metric}, L={L}, final_w={final_w}, "
              f"thresh={margin_thresh}, persist_k={persist_k}, require_final_lock={require_final_lock}")

    # ---------- scan for earliest stable layer ----------
    for i in range(L):
        end = min(i + persist_k, L)
        ok = True
        reasons = []
        # window checks
        for j in range(i, end):
            # margin check
            mj = margins[j]
            if mj is None or (mj < margin_thresh):
                ok = False
                if debug:
                    reasons.append(f"layer {j}: margin {mj} < {margin_thresh}")
                break
            # winner lock check (optional)
            if require_final_lock and winners[j] != final_w:
                ok = False
                if debug:
                    reasons.append(f"layer {j}: winner '{winners[j]}' != final '{final_w}'")
                break

        if ok:
            return {
                "idx": i,
                "winner_final": final_w,
                "winner_at_i": winners[i],
                "winners_window": winners[i:end],
                "margins_window": margins[i:end],
                "margin_at_i": margins[i],
                "series": series,
                "metric": metric,
                "persist_k": persist_k,
                "threshold": margin_thresh,
            }
        elif debug:
            print(f"[EDL] reject i={i}: " + ("; ".join(reasons) if reasons else "no reason logged"))

    # ---------- no layer qualifies ----------
    if debug:
        print("[EDL] no early decision layer found under given settings.")
    return None


