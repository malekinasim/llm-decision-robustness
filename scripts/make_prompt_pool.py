# make_mcq_pool.py
import json, random
import argparse
import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # add repo root to sys.path

from src.util import load_model_and_tokenizer, get_device

COUNTRIES = [
    ("France"," Paris"," London"," Berlin"),
    ("Germany"," Berlin"," Paris"," Rome"),
    ("Italy"," Rome"," Madrid"," Berlin"),
    ("Spain"," Madrid"," Rome"," Lisbon"),
    ("Japan"," Tokyo"," Osaka"," Kyoto"),
    ("Canada"," Ottawa"," Toronto"," Montreal"),
    ("Brazil"," Brasilia"," Rio"," Sao Paulo"),
    ("India"," New Delhi"," Mumbai"," Kolkata"),
    ("China"," Beijing"," Shanghai"," Shenzhen"),
    ("Australia"," Canberra"," Sydney"," Melbourne"),
    ("Iran"," Tehran"," Isfahan"," Shiraz"),
    ("Turkey"," Ankara"," Istanbul"," Izmir"),
    ("Russia"," Moscow"," Saint Petersburg"," Kazan"),
    ("Egypt"," Cairo"," Alexandria"," Giza"),
    ("Mexico"," Mexico City"," Guadalajara"," Monterrey"),
]

def is_single_token(tokenizer, s):
    ids = tokenizer.encode(s, add_special_tokens=False)
    return len(ids) == 1

def build_prompt_items(tokenizer, n_math=400, seed=0,require_single_token=False):
    random.seed(seed)
    items = []

    # 1) Country capitals
    for c, a, b, d in COUNTRIES * 20: 
        opts = [a,b,d]
        if all(is_single_token(tokenizer, o) for o in opts):
            items.append({
                "id": f"mcq-cap-{c.lower()}-{random.randint(0,9999)}",
                "task": "mcq",
                "question": f"The capital of {c} is ",
                "options": opts,
                "answer": a
            })
    for c, a ,_,_ in COUNTRIES * 20: 
        if require_single_token==False or  all(is_single_token(tokenizer, o) for o in opts):
            items.append({
                "id": f"single-cap-{c.lower()}-{random.randint(0,9999)}",
                "task": "single",
                "question": f"The capital of {c} is ",
                "answer": a
            })

    # 2) Simple math additions (ensure single-token options)
    for _ in range(n_math):
        x = random.randint(2, 60); y = random.randint(2, 60)
        gold = f" {x+y}"
        decoys = [f" {x+y+1}", f" {x+y-1}", f" {x+y+2}"]
        opts = [gold] + decoys
        random.shuffle(opts)
        if require_single_token==False or all(is_single_token(tokenizer, o) for o in opts):
            items.append({
                "id": f"mcq-math-{x}-{y}-{random.randint(0,9999)}",
                "task": "mcq",
                "question": f"{x} + {y} = ",
                "options": opts,
                "answer": gold
            })

    for _ in range(n_math):
        x = random.randint(2, 60); y = random.randint(2, 60)
        gold = f" {x+y}"
        opts = [gold] + decoys
        random.shuffle(opts)
        if require_single_token==False or all(is_single_token(tokenizer, o) for o in opts):
            items.append({
                "id": f"single-math-{x}-{y}-{random.randint(0,9999)}",
                "task": "single",
                "question": f"{x} + {y} = ",
                "answer": gold
            })

    random.shuffle(items)
    print(len(items))
    return items

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=" probing for MCQ or Single (per-layer).")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--remote", default=False)
    parser.add_argument("--require_single_token", default=False)
    args = parser.parse_args()
    device = get_device()
    model, tokenizer = load_model_and_tokenizer(args.model, device, remote=args.remote)
    mcqs = build_prompt_items(tokenizer, 800, seed=0,require_single_token=args.require_single_token) 
    model_path = args.model.replace("/", "__")
    path_data=os.path.join("data",model_path)
    
    os.makedirs(path_data, exist_ok=True)
    with open(os.path.join(path_data,"prompt_pool.json"),"w",encoding="utf-8") as f:
        json.dump(mcqs, f, ensure_ascii=False, indent=2)
    print("Saved mcq_pool.json with", len(mcqs), "items")
    
    