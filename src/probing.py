# mass_mean_probe.py
import random
import numpy as np
import torch

# --------------------
# Helpers
# --------------------


@torch.no_grad()
def get_last_token_hidden(model, tok, device, prompt: str, cont: str, MAX_LEN: int = 512):
    """
    Return last-token hidden state of merged (prompt + cont), robust to padding.
    """
    p = str(prompt) if prompt is not None else ""
    c = str(cont) if cont is not None else ""
    if c and not c[:1].isspace():
        text = p + " " + c
    else:
        text = p + c  
    enc = tok(text, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(device)
    out = model(**enc, output_hidden_states=True, use_cache=False)
    hs  = out.hidden_states[-1][0] 

    ids  = enc["input_ids"]      
    T    = ids.shape[1]
    pad  = tok.pad_token_id
    if pad is None:
        last_idx = T - 1
    else:
        ar   = torch.arange(T, device=ids.device).unsqueeze(0).expand_as(ids)
        mask = (ids != pad)  
        last_idx = int((ar * mask).max(dim=1).values.item())

    return hs[last_idx].detach().cpu().numpy()


@torch.no_grad()
def get_alllayer_hidden_lasttok(model,tok,device,prompt: str, cont: str,max_len=128):
    p = str(prompt) if prompt is not None else ""
    c = str(cont) if cont is not None else ""
    if c and not c[:1].isspace():
        text = p + " " + c
    else:
        text = p + c 
    enc = tok(text, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states  # tuple: [emb, h1, ..., hL]
    ids = enc["input_ids"]  # [1, T]
    T = ids.shape[1]
    pad_id = tok.pad_token_id
    if pad_id is None:
        last_idx = torch.tensor([T-1], device=device)
    else:
        ar = torch.arange(T, device=device).unsqueeze(0).expand_as(ids)
        mask = (ids != pad_id)
        last_idx = (ar * mask).max(dim=1).values  # [1]
    start = 1  # skip embedding
    vecs = []
    for i in range(start, len(hs)):
        H = hs[i][0]                # [T, D]
        vecs.append(H[last_idx.item()].detach().cpu().numpy())  # [D]
    # Output: List of length L (without embedding), each [D]
    return np.stack(vecs, axis=0)   # shape: [L, D]


def corrupt_numeric_answer(ans: str):
    # Generate a simple incorrect numeric answer (±1 or ±2; if the number is large ±(1..3))
    try:
        v = int(ans.strip())
        delta = random.choice([1, -1, 2, -2, 3, -3])
        if delta == 0: delta = 1
        return f" {v + delta}"
    except:
        # If there is no number, a simple replacement
        return " 0"

def build_pairs_mcq(model,tok,device,items):
    H, y, groups = [], [], [] # groups because each MCQ is a question
    for it in items:
        if it.get("task") != "mcq": continue
        prompt = it["question"]
        gold = it["answer"]
        for opt in it["options"]:
            h = get_last_token_hidden(model,tok,device,prompt, opt)
            H.append(h)
            y.append(1 if opt == gold else 0)
            groups.append(it["id"])
    return np.stack(H), np.array(y), np.array(groups)
def build_pairs_mcq_layers(model,tok,device,items):
    X_layers, y = [], []
    for it in items:
        if it.get("task","").lower()!="mcq": continue
        prompt = it["question"]; gold = it["answer"]
        for opt in it["options"]:
            X = get_alllayer_hidden_lasttok(model,tok,device,prompt, " "+opt)  # [L,D]
            X_layers.append(X)
            y.append(1 if opt==gold else 0)
    X_layers = np.stack(X_layers, axis=0)  # [N, L, D]
    y = np.array(y, dtype=np.int64)
    return X_layers, y

def build_pairs_single_layers(model,tok,device,items, negatives_per=2):
    X_layers, y = [], []
    singles = [it for it in items if it.get("task","").lower() in {"single","singel"}]
    # Collect all the gold to make better negatives
    golds = [it["answer"] for it in singles if isinstance(it.get("answer"), str)]
    rng = np.random.default_rng(0)

    def make_neg(base):
        try:
            n = int(base.strip()); return " "+str(n + int(rng.choice([-2,-1,1,2,3,-3])))
        except: 
            cands = [g for g in golds if g!=base]
            return " "+(rng.choice(cands) if cands else "0")

    for it in singles:
        q = it["question"]; gold = it.get("answer")
        # positive(answer option) +
        Xp = get_alllayer_hidden_lasttok(model,tok,device,q, " "+gold); X_layers.append(Xp); y.append(1)
        # negetive(other incorrect option) -
        for _ in range(negatives_per):
            Xn = get_alllayer_hidden_lasttok(model,tok,device,q, make_neg(gold)); X_layers.append(Xn); y.append(0)
    X_layers = np.stack(X_layers, axis=0)  # [N, L, D]
    y = np.array(y, dtype=np.int64)
    return X_layers, y

# src/probing.py
def build_pairs_single(model, tok, device, items, negatives_per=2):
    """
    Return a *list of pairs*: (prompt, completion, label, group_id)
    label = 1 for gold answer, 0 for negative
    """
    pairs = []
    for it in items:
        # accept both 'single' and old 'singel'
        task = (it.get("task") or "").lower()
        if task not in {"single", "singel"}:
            continue

        prompt = str(it["question"])
        gold   = str(it["answer"])
        gid    = it.get("id", "")

        # positive
        pairs.append((prompt, gold, 1, gid))

        # negatives (numeric corruption; you can change to your own generator)
        for _ in range(negatives_per):
            neg = corrupt_numeric_answer(gold)
            if neg == gold:
                continue
            pairs.append((prompt, str(neg), 0, gid))

    return pairs

def eval_binary_from_pairs(pairs, W, model, tokenizer, get_single_hiddens_fn, skip_embedding=True):
    """
    Returns per-layer AUROC-like ranking metric is not computed here to keep deps minimal;
    we return per-layer accuracy at threshold 0 (dot>=0 -> True).
    """


    BATCH = 64
    all_scores = {li: [] for li in W.keys()}
    all_labels = []

    for i in range(0, len(pairs), BATCH):
        chunk = pairs[i:i+BATCH]
        prompts = [x[0] for x in chunk]
        comps   = [x[1] for x in chunk]
        labels  = [x[2] for x in chunk]
        layer_vecs = get_single_hiddens_fn(
            model, tokenizer, prompts, comps, skip_embedding=skip_embedding
        )
        for li, w in W.items():
            V = layer_vecs[li]  # [b,D]
            sc = torch.matmul(V, w)  # [b]
            all_scores[li].extend(sc.tolist())
        all_labels.extend(labels)
    def _to_int_label(l):
        import numpy as np
        if hasattr(l, "item") and not isinstance(l, (list, tuple)):
            try: l = l.item()
            except Exception: pass
        if isinstance(l, (list, tuple, np.ndarray)):
            if len(l) == 0: return 0
            l = l[0]
            if hasattr(l, "item"):
                try: l = l.item()
                except Exception: pass
        try: return int(l)
        except Exception: return int(bool(l))

    
    per_layer_acc = {}
    y = np.array([_to_int_label(l) for l in all_labels], dtype=np.int64)

    for li in W.keys():
        s = np.array(all_scores[li])
        yhat = (s >= 0).astype(int)
        per_layer_acc[li] = float((yhat == y).mean())
    # pick best
    best_layer, best_acc = None, -1.0
    for li, acc in per_layer_acc.items():
        if acc > best_acc:
            best_layer, best_acc = li, acc

    return per_layer_acc, best_layer, best_acc