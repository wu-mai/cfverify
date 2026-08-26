"""
run_mistral_K_ablation.py — Mistral-7B K-ablation (K=1, K=3, K=5).

For each of 60 questions (30 HotpotQA + 30 2Wiki), run:
  - cond1: full evidence (one call)
  - cond2: gold removed (one call)
  - K × cond3: random-set removal with K different RNG seeds (K=1,3,5)

Re-uses the same Mistral-7B model; outputs a single results/mistral7b_Kablation.json
with per-question records and an aggregated summary.

Usage:
    python scripts/run_mistral_K_ablation.py [--K 1,3,5]

Outputs:
    results/mistral7b_Kablation.json
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
ROOT = HERE.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

MODEL = "/root/autodl-tmp/gcy_cf"  # local path; weights moved to data disk

print("Loading Mistral-7B-Instruct-v0.3 (bf16, local)...")
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="auto"
)
model.eval()
print(f"Loaded in {time.time()-t0:.1f}s")


def ask(question, evidence_sents, max_new_tokens=40):
    """Greedy answer call; first non-empty line, stripped."""
    ev_text = (
        "\n".join(f"[{i}] {s}" for i, s in evidence_sents)
        if evidence_sents else "(no evidence)"
    )
    prompt = (
        f"Evidence:\n{ev_text}\n\nQuestion: {question}\n\n"
        "Answer the question based ONLY on the evidence above. Give a concise answer "
        "(a few words), or say \"insufficient evidence\" if the evidence does not "
        "support an answer. Do not use outside knowledge.\n\nAnswer:"
    )
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    ans = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return ans.split('\n')[0].strip().strip('"').strip('.')


def load_questions():
    hot = json.loads((DATA / "hotpotqa_30_questions.json").read_text())
    wiki = json.loads((DATA / "2wiki_30_questions.json").read_text())
    return [("hotpotqa", q) for q in hot] + [("2wiki", q) for q in wiki]


def build_sents(q):
    sents = []
    for d in q["documents"]:
        for s in d["sentences"]:
            sents.append(s)
    return sents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", default="1,3,5",
                        help="comma list of K values to evaluate")
    parser.add_argument("--max_questions", type=int, default=0,
                        help="0 = all; positive = first N (debug)")
    args = parser.parse_args()
    Ks = [int(k) for k in args.K.split(",") if k.strip()]
    questions = load_questions()
    if args.max_questions:
        questions = questions[: args.max_questions]

    print(f"\n=== K ablation: Ks={Ks}, N={len(questions)} ===\n")
    all_records = []
    t_global = time.time()
    for qi, (ds, q) in enumerate(questions, 1):
        all_sents = build_sents(q)
        gold = set(q["gold_sentence_indices"])
        non_gold = [i for i in range(len(all_sents)) if i not in gold]
        n_remove = min(len(gold), len(non_gold))
        rng = random.Random(q["question_id"] * 31 + 7)
        full_ev = [(i, s) for i, s in enumerate(all_sents)]
        gold_removed_ev = [(i, s) for i, s in enumerate(all_sents) if i not in gold]

        t = time.time()
        a1 = ask(q["question"], full_ev)
        a2 = ask(q["question"], gold_removed_ev)

        per_K = {}
        for K in Ks:
            samples = []
            for k in range(K):
                seed = q["question_id"] * 1009 + 31 * (k + 1) + 7
                local_rng = random.Random(seed)
                rr = set(local_rng.sample(non_gold, n_remove))
                rr_ev = [(i, s) for i, s in enumerate(all_sents) if i not in rr]
                a3 = ask(q["question"], rr_ev)
                samples.append({
                    "k_idx": k + 1,
                    "seed": seed,
                    "random_removed_indices": sorted(rr),
                    "answer": a3,
                    "flipped": (a3.strip().lower() != a1.strip().lower()),
                })
            per_K[str(K)] = samples

        rec = {
            "dataset": ds,
            "qid": q["question_id"],
            "question": q["question"],
            "gold_answer": q["answer"],
            "type": q.get("type", ""),
            "gold_indices": sorted(gold),
            "cond1_full": a1,
            "cond2_gold_removed": a2,
            "targeted_flipped": (a2.strip().lower() != a1.strip().lower()),
            "per_K": per_K,
        }
        all_records.append(rec)
        elapsed = time.time() - t
        print(
            f"  [{qi:>2}/{len(questions)}] {ds:>8} q{q['question_id']:>3} "
            f"[{elapsed:>4.1f}s] a1='{a1[:30]}' a2='{a2[:30]}' "
            f"K1flip={per_K[str(Ks[0])][0]['flipped']!s:>5}"
        )

    out_path = RESULTS / "mistral7b_Kablation.json"
    out_path.write_text(json.dumps({
        "model": MODEL,
        "Ks": Ks,
        "n_questions": len(questions),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_seconds": time.time() - t_global,
        "records": all_records,
    }, indent=2))
    print(f"\nWrote {out_path}  (total {time.time()-t_global:.0f}s)")


if __name__ == "__main__":
    main()