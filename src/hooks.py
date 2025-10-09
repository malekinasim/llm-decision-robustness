
from typing import Optional, Dict, Any, List
import torch
from src.tuned import TunedDiag
from src.logit_lens import *
from typing import Dict
import numpy as np

def _to_str(x):
    # convert numpy scalars / arrays and bytes to clean str
    if isinstance(x, (bytes, bytearray)):
        try:
            return x.decode("utf-8", "ignore")
        except Exception:
            return str(x)
    if isinstance(x, (np.generic, np.ndarray)):
        try:
            # scalar array
            if np.ndim(x) == 0:
                x = x.item()
            else:
                # join vector elements (fallback)
                x = " ".join(map(str, np.ravel(x).tolist()))
        except Exception:
            x = str(x)
    return str(x)

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
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)

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
    if not hasattr(out, "hidden_states") or out.hidden_states is None:
        raise RuntimeError("model outputs missing hidden_states; ensure output_hidden_states=True")

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
        p = _to_str(p)
        c = _to_str(c)
        if add_space_between:
            # add a single space IFF completion does not already start with whitespace
            if c and not c[:1].isspace():
                merged.append(p + " " + c)
            else:
                merged.append(p + c)
        else:
            merged.append(p + c)
    return _alllayer_lasttoken_hiddens_core(
        model, tokenizer, merged, skip_embedding=skip_embedding
    )


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

