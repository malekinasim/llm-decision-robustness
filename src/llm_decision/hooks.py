
from typing import Optional, Dict, Any, List
import torch
from tuned import TunedDiag
from metrics import *

@torch.no_grad()
def layerwise_logits_for_pos(
    model, tokenizer, text=None, outputs=None, pos=-1,
    ln_f_mode="last_only",      # "raw" | "last_only" | "all"
    skip_embedding=False,       # True => start from block1 (drop embedding row)
    tuned: "TunedDiag|None" = None,
    option_ids: "list[int]|None" = None   # if provided -> return logits only for these ids
):
    """
    Returns:
      - if tuned is None: list[Tensor]   (per-layer logits)
      - else: {"raw": list[Tensor], "tuned": list[Tensor]}
    If option_ids is not None, each Tensor has shape [|options|] instead of [V].
    """
    device = next(model.parameters()).device
    if outputs is None:
        assert text is not None, "Provide `text` or `outputs`"
        inputs = tokenizer(text, return_tensors="pt").to(device)
        outputs = model(**inputs)

    hs  = outputs.hidden_states                 # [emb, h1, ..., hL]
    W_U = model.lm_head.weight.T               # [d, V]
    ln_f = getattr(model.transformer, "ln_f", None)
    start = 1 if skip_embedding else 0
    L = len(hs) - 1


    WU_opts = None
    if option_ids is not None:
        opt_idx = torch.as_tensor(option_ids, device=W_U.device, dtype=torch.long)
        WU_opts = torch.index_select(W_U, dim=1, index=opt_idx)  # [d, |opts|]

    def project_vec(x, i):
        if ln_f is not None:
            if ln_f_mode == "last_only" and i == len(hs) - 1:
                x = ln_f(x)
            elif ln_f_mode == "all":
                x = ln_f(x)
        if option_ids is None:
            return x @ W_U                    # [V]
        else:
            return x @ WU_opts                # [|opts|]

    layer_raw, layer_tuned = [], []
    for i in range(start, len(hs)):
        x = hs[i][0, pos]                     # [d]
        z_raw = project_vec(x, i)
        layer_raw.append(z_raw)
        if tuned is not None:
            xt = tuned.apply(i, x)
            z_tuned = project_vec(xt, i)
            a_i  = tuned.alpha(i)
            if a_i is not None:    
                z_tuned = a_i * z_tuned
            layer_tuned.append(z_tuned)

    if tuned is None:
        return layer_raw
    else:
        return {"raw": layer_raw, "tuned": layer_tuned}

@torch.no_grad()
def _alllayer_lasttoken_hiddens_core(
    model,
    tokenizer,
    merged_texts: List[str],
    skip_embedding: bool = True,
):
    """
    Core routine: given a batch of full sequences (already merged 'prompt + completion'),
    return per-layer hidden vectors at the *last non-pad token* for each sequence.
    Returns:
        layer_h: List[Tensor], length = n_layers(± embedding), each [B, D]
    """
    device = next(model.parameters()).device

    # ensure pad token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    enc = tokenizer(
        merged_texts, return_tensors="pt", padding=True, truncation=True
    ).to(device)

    out = model(**enc, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states  # [emb, h1, ..., hL]

    start = 1 if skip_embedding else 0
    T = enc["input_ids"].shape[1]

    ids = enc["input_ids"]
    ar = torch.arange(T, device=ids.device).unsqueeze(0).expand_as(ids)
    mask = (ids != tokenizer.pad_token_id)
    last_idx = (ar * mask).max(dim=1).values  # [B]

    layer_h = []
    for i in range(start, len(hs)):
        H = hs[i]                  # [B, T, D]
        B, _, D = H.shape
        vecs = H[torch.arange(B, device=H.device), last_idx]  # [B, D]
        layer_h.append(vecs.detach().cpu())
    return layer_h  # length = n_layers (± embedding), each [B, D]


@torch.no_grad()
def single_alllayer_hiddens(
    model,
    tokenizer,
    prompt_texts: List[str],
    completions: List[str],
    skip_embedding: bool = True,
    add_space_between: bool = True,
):
    merged = []
    for p, c in zip(prompt_texts, completions):
        sep = " " if add_space_between and (len(c) > 0 and not c.startswith((" ", "\n"))) else ""
        merged.append(p + sep + c)

    return _alllayer_lasttoken_hiddens_core(
        model, tokenizer, merged, skip_embedding=skip_embedding
    )
from typing import Dict

@torch.no_grad()
def mcq_alllayer_hiddens(
    model,
    tokenizer,
    prompt_text: str,
    options: List[str],
    skip_embedding: bool = True,
    add_space_between: bool = True,
) -> List[Dict[str, torch.Tensor]]:
    """
    One prompt, multiple options (MCQ). Returns list of layer dicts:
      layer_dicts[layer_idx][option] -> 1D tensor [D] on CPU
    """
    prompt_texts = [prompt_text] * len(options)
    # Reuse the single wrapper
    layer_h = single_alllayer_hiddens(
        model, tokenizer, prompt_texts, options,
        skip_embedding=skip_embedding, add_space_between=add_space_between
    )
    # layer_h: List[Tensor [B, D]] where B = len(options)
    layer_dicts: List[Dict[str, torch.Tensor]] = []
    for layer_i, mat in enumerate(layer_h):
        # mat: [B, D]
        d = {opt: mat[b] for b, opt in enumerate(options)}
        layer_dicts.append(d)
    return layer_dicts

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
          - metric: "gold" or "top1_top2"
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
        metric = "gold"
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
