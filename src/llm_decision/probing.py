import torch
from typing import List, Dict, Tuple, Optional, Any
from tuned import TunedDiag
from metrics import *
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
import torch
from typing import List, Dict, Tuple, Optional, Any

@torch.no_grad()
def mcq_alllayer_scores_v2(
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
