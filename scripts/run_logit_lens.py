# scripts/run_logit_lens.py
# Layer-wise logit lens for MCQ and Single-token prompts (with optional Tuned Lens)
import sys
from pathlib import Path
import argparse
import os
import random
import numpy as np
import matplotlib.pyplot as plt

# --- make 'src' importable when running as a script ---
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from src.util import load_model_and_tokenizer           
from src.io import load_prompts_with_options, ensure_dirs, \
                    save_CSV_layers_MCQ_Margins, save_csv_margins         
from src.tuned import TunedDiag                       
# logit-lens core + EDL
from src.logit_lens import mcq_alllayer_scores, early_decision_layer   
from src.logit_lens import compute_margins_per_layer_logits              

def pick_item(items, idx: int | None):
    """Robust pick: if idx is None -> first; if out of range -> random; else -> that index."""
    if not items:
        return None
    n = len(items)
    if idx is None:
        return items[0]
    if 0 <= idx < n:
        return items[idx]
    # fall back: random valid
    return items[random.randint(0, n-1)]

def overlay_edl_line(ax: plt.Axes, edl_idx: int | None, color="red", label="EDL"):
    """Draw a vertical line at early-decision layer index (if provided)."""
    if edl_idx is None or edl_idx < 0:
        return
    ax.axvline(x=edl_idx, color=color, linestyle="--", alpha=0.8, label=label)

def save_mcq_plot_with_edl(res, out_png: str, title: str, edl_idx: int | None):
    """
    Reproduce your plot_MCQ_Margin AND add EDL vertical line (no need to modify viz.py).
    """
    # Unpack margins
    raw_scores, _, raw_m12, raw_gold = res["raw"]
    tuned_part = res["tuned"]
    tuned_m12, tuned_gold = None, None
    if tuned_part is not None:
        _, _, tuned_m12, tuned_gold = tuned_part

    # Prepare layer x-axis (skip embedding -> layer index = 0..L-1)
    L = len(raw_m12)
    layers = list(range(L))

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.figure(figsize=(9, 4))
    ax = plt.gca()

    # Curves
    ax.plot(layers, raw_m12, label="top1-top2 (raw)")
    if raw_gold and any(v is not None for v in raw_gold):
        ax.plot(layers, [np.nan if v is None else v for v in raw_gold], label="gold-margin (raw)")
    if tuned_m12 is not None:
        ax.plot(layers, tuned_m12, label="top1-top2 (tuned)")
        if tuned_gold and any(v is not None for v in tuned_gold):
            ax.plot(layers, [np.nan if v is None else v for v in tuned_gold], label="gold-margin (tuned)")

    # Overlay Early Decision Layer
    overlay_edl_line(ax, edl_idx, color="red", label="EDL")

    ax.set_xlabel("Layer index (0 = first after embedding)")
    ax.set_ylabel("Margin")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def save_single_plot_with_edl(Z_full, out_png: str, title: str, edl_idx: int | None):
    """
    Reproduce your plot_margins AND add EDL vertical line (no need to modify viz.py).
    """
    # Unpack
    # Z_full like {"full": {"raw":{"top1_top2_full":[...], "gold_full":[...]}, "tuned": {...}}, "opts": ...}
    raw = Z_full["full"]["raw"]
    tuned = Z_full["full"].get("tuned", None)
    top1_full = raw.get("top1_top2_full", [])
    gold_full = raw.get("gold_full", None)
    L = len(top1_full)
    layers = list(range(L))

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.figure(figsize=(9, 4))
    ax = plt.gca()

    # Curves
    ax.plot(layers, top1_full, label="top1-top2 full (raw)")
    if gold_full:
        ax.plot(layers, gold_full, label="gold full (raw)")
    if tuned:
        t_top1 = tuned.get("top1_top2_full", None)
        if t_top1:
            ax.plot(layers, t_top1, label="top1-top2 full (tuned)")
        t_gold = tuned.get("gold_full", None)
        if t_gold:
            ax.plot(layers, t_gold, label="gold full (tuned)")

    overlay_edl_line(ax, edl_idx, color="red", label="EDL")

    ax.set_xlabel("Layer index (0 = first after embedding)")
    ax.set_ylabel("Margin")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Logit Lens (MCQ/Single) with optional Tuned Lens + EDL overlay")
    parser.add_argument("--model", default="EleutherAI/gpt-neo-125M", help="HF model name or local path")
    parser.add_argument("--dataset", default=str(REPO_ROOT / "data" / "prompt_pool.json"))
    parser.add_argument("--task", choices=["mcq", "single"], required=True)
    parser.add_argument("--mcq_idx", type=int, default=None, help="Index of MCQ item (if None: first)")
    parser.add_argument("--single_idx", type=int, default=None, help="Index of Single item (if None: first)")
    parser.add_argument("--tuned_json", default=None, help="Path to tuned lens JSON (optional)")
    parser.add_argument("--margin_thresh", type=float, default=0.0, help="EDL margin threshold")
    parser.add_argument("--persist_k", type=int, default=2, help="EDL persistence window length")
    parser.add_argument("--ln_f_mode", choices=["none", "last_only", "all"], default="last_only")
    parser.add_argument("--skip_embedding", action="store_true", default=True,
                        help="If set, skip embedding state; first layer is the first transformer block")
    
    parser.add_argument("--remote",  default=False)
    args = parser.parse_args()

    # Load model/tokenizer
    model, tok = load_model_and_tokenizer(args.model,remote=args.remote)

    # Output folders consistent with your new structure
    model_path = args.model.replace("/", "__")
    tables_dir = os.path.join(REPO_ROOT, "reports", model_path, "tables")
    figs_dir   = os.path.join(REPO_ROOT, "reports", model_path, "figures")
    ensure_dirs(tables_dir, figs_dir)  # ✅ your helper:contentReference[oaicite:6]{index=6}

    # Load items
    mcq_items, single_items = load_prompts_with_options(args.dataset, tok, require_single_token=True)

    # Tuned lens (optional)
    tuned = None
    if args.tuned_json and TunedDiag is not None:
        tuned = TunedDiag.from_json(args.tuned_json, device=next(model.parameters()).device)

    # ---------------- MCQ ----------------
    if args.task == "mcq":
        valid_mcq = [it for it in mcq_items if isinstance(it.get("question"), str)
                     and isinstance(it.get("options"), list)
                     and len(it["options"]) >= 2
                     and it.get("answer") in it["options"]]
        if not valid_mcq:
            print("[LogitLens-MCQ] No valid MCQ items.")
            return

        item = pick_item(valid_mcq, args.mcq_idx)
        q, options, gold = item["question"], item["options"], item.get("answer")

        # lens per layer
        res = mcq_alllayer_scores(
            model=model, tokenizer=tok,
            prompt_text=q, options=options, gold_opt=gold,
            pos=-1, ln_f_mode=args.ln_f_mode, skip_embedding=args.skip_embedding,
            tuned=tuned
        )  # returns dict with raw/tuned + winners + margins:contentReference[oaicite:7]{index=7}

        # Early decision
        ed = early_decision_layer(
            res,
            margin_thresh=args.margin_thresh,
            use_tuned=bool(res.get("tuned")),
            use_gold=False,
            persist_k=args.persist_k
        )  # {idx, ...} or None:contentReference[oaicite:8]{index=8}

        print(f"[EDL-MCQ] {ed}")

        # Save CSV with margins (your helper)
        base = f"mcq_{item.get('id', 'item')}"
        save_CSV_layers_MCQ_Margins(res, options=options, out_dir=tables_dir, fname=f"{base}__layer_margins.csv")  # ✅:contentReference[oaicite:9]{index=9}

        # Plot with EDL overlay (custom, no change to viz.py)
        out_png = os.path.join(figs_dir, f"{base}__layer_margins.png")
        save_mcq_plot_with_edl(res, out_png=out_png,
                               title=f"MCQ margins per layer ({'raw + tuned' if res.get('tuned') else 'raw'})",
                               edl_idx=(ed["idx"] if ed else None))
        print(f"[SAVE] {out_png}")
        return

    # ---------------- Single ----------------
    if args.task == "single":
        valid_single = [it for it in single_items if isinstance(it.get("question"), str)
                        and isinstance(it.get("answer"), str)]
        if not valid_single:
            print("[LogitLens-Single] No single items.")
            return

        item = pick_item(valid_single, args.single_idx)
        q, gold_text = item["question"], item["answer"]

        
        Z_full = compute_margins_per_layer_logits(
            model, tok,
            text=q, outputs=None, pos=-1,
            ln_f_mode=args.ln_f_mode, skip_embedding=args.skip_embedding,
            gold_text=gold_text, options=None, gold_option=None,
            tuned=tuned
        ) 

     
        raw_full = Z_full["full"]["raw"]
        gold_full = raw_full.get("gold_full", None)
        ed_idx = None
        if gold_full:
            L = len(gold_full)
           
            for i in range(L):
                end = min(i + args.persist_k, L)
                window = gold_full[i:end]
                if all((m is not None) and (m >= args.margin_thresh) for m in window):
                    ed_idx = i; break

        print(f"[EDL-Single] idx={ed_idx} (gold_full-based)")

        # Save CSV (your helper)
        base = f"single_{item.get('id', 'item')}"
        save_csv_margins(Z_full, out_dir=tables_dir, fname=f"{base}__layer_margins.csv") 
        # Plot with EDL overlay
        out_png = os.path.join(figs_dir, f"{base}__layer_margins.png")
        save_single_plot_with_edl(Z_full, out_png=out_png,
                                  title=f"Single-token margins per layer ({'raw + tuned' if Z_full['full'].get('tuned') else 'raw'})",
                                  edl_idx=ed_idx)
        print(f"[SAVE] {out_png}")
        return

    raise ValueError("Invalid task")

if __name__ == "__main__":
    main()
