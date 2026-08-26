"""
run_decoding_seeds.py — Decoding-seed variance analysis for CF-Verify.

The headline and all extensions use do_sample=False (greedy). This script
runs the same N=60 question pool with N_seeds=5 different sampling seeds
(temperature > 0) to measure how much of the CFScore variance is due to
decoding randomness vs. question difficulty.

Per question: full + targeted + K random = 3 calls × N_seeds = 15 calls.

Outputs:
    results/decoding_seeds.json
"""
import json
import os
import random
import sys
import time
from pathlib import Path

from openai import OpenAI

HERE = Path(__file__).parent
ROOT = HERE.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

SYSTEM_PROMPT = (
    "You are a question-answering assistant. Answer the question based ONLY "
    "on the evidence provided. Give a concise answer (a few words). If the "
    "evidence does not support an answer, say exactly 'insufficient evidence'. "
    "Do not use any other knowledge."
)


def get_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Set OPENAI_API_KEY first.")
    base_url = os.environ.get("OPENAI_BASE_URL")
    return OpenAI(api_key=api_key, base_url=base_url)


def call_llm(client, model, question, evidence_sents, max_output=80, temperature=0.7, seed=None):
    ev_text = "\n".join(f"[{i}] {s}" for i, s in enumerate(evidence_sents))
    user = f"Question: {question}\n\nEvidence:\n{ev_text}\n\nAnswer:"
    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        max_tokens=max_output,
        temperature=temperature,
    )
    if seed is not None:
        kwargs["seed"] = seed
    import time as _t
    for attempt in range(6):
        try:
            r = client.chat.completions.create(**kwargs)
            break
        except Exception as e:
            if attempt == 5:
                raise
            wait = min(60 * (attempt + 1), 300)
            print(f"    [retry {attempt+1}/6] {str(e)[:80]} — sleeping {wait}s")
            _t.sleep(wait)
    return r.choices[0].message.content.strip().split("\n")[0].strip().strip('"').strip(".").lower()


def norm(s):
    return s.strip().lower().strip(".").strip('"').strip()


def main():
    client = get_client()
    model = "gpt-5.4"
    N_seeds = 5
    K = 1
    seeds = [42, 123, 456, 789, 1337]
    assert len(seeds) == N_seeds

    hot = json.loads((DATA / "hotpotqa_30_questions.json").read_text())
    wiki = json.loads((DATA / "2wiki_30_questions.json").read_text())
    questions = [("hotpotqa", q) for q in hot] + [("2wiki", q) for q in wiki]
    print(f"Model: {model}, K={K}, N_seeds={N_seeds}")
    print(f"Per Q: 3 calls × {N_seeds} seeds = {3*N_seeds} API calls")
    print(f"Questions: {len(questions)}, total API calls ≈ {len(questions) * 3 * N_seeds}")

    all_records = []
    t0 = time.time()
    for qi, (ds_label, q) in enumerate(questions, 1):
        all_sents = []
        for d in q["documents"]:
            for s in d["sentences"]:
                all_sents.append(s)
        gold = set(q["gold_sentence_indices"])
        non_gold = [i for i in range(len(all_sents)) if i not in gold]
        n_remove = min(len(gold), len(non_gold))
        full_ev = [(i, s) for i, s in enumerate(all_sents)]
        target_ev = [(i, s) for i, s in enumerate(all_sents) if i not in gold]

        full_a = call_llm(client, model, q["question"], full_ev, seed=seeds[0])
        abstained = full_a.startswith("insufficient") or not full_a.strip()

        seed_results = []
        for seed in seeds:
            tgt_a = call_llm(client, model, q["question"], target_ev, seed=seed)
            rng = random.Random(q["question_id"] * 1009 + 31 + seed)
            rr = set(rng.sample(non_gold, n_remove))
            rr_ev = [(i, s) for i, s in enumerate(all_sents) if i not in rr]
            rnd_a = call_llm(client, model, q["question"], rr_ev, seed=seed)
            t_flipped = (norm(full_a) != norm(tgt_a))
            r_flipped = (norm(full_a) != norm(rnd_a))
            seed_results.append({
                "seed": seed,
                "target_answer": tgt_a,
                "random_answer": rnd_a,
                "targeted_flipped": t_flipped,
                "random_flipped": r_flipped,
            })

        rec = {
            "dataset": ds_label,
            "qid": q["question_id"],
            "question": q["question"],
            "gold_answer": q["answer"],
            "full_answer": full_a,
            "abstained": abstained,
            "per_seed": seed_results,
        }
        all_records.append(rec)

        elapsed = time.time() - t0
        if qi % 5 == 0 or qi == len(questions):
            n_ans = sum(1 for r in all_records if not r["abstained"])
            print(f"  [{qi:>2}/{len(questions)}] {ds_label:>8} q{q['question_id']:>3} "
                  f"[{elapsed:>5.0f}s] answered={n_ans}/{qi}")

    # Compute per-seed F_T, F_R, CFScore
    seed_stats = []
    for s_idx, seed in enumerate(seeds):
        ans = [r for r in all_records if not r["abstained"]]
        ft = sum(r["per_seed"][s_idx]["targeted_flipped"] for r in ans) / len(ans)
        fr = sum(r["per_seed"][s_idx]["random_flipped"] for r in ans) / len(ans)
        seed_stats.append({"seed": seed, "F_T": ft, "F_R": fr, "CFScore": ft - fr,
                          "n_answered": len(ans)})

    # Aggregate
    import statistics
    ft_list = [s["F_T"] for s in seed_stats]
    fr_list = [s["F_R"] for s in seed_stats]
    cf_list = [s["CFScore"] for s in seed_stats]
    summary = {
        "model": model,
        "K": K,
        "N_seeds": N_seeds,
        "N_questions": len(questions),
        "per_seed": seed_stats,
        "mean_F_T": statistics.mean(ft_list),
        "mean_F_R": statistics.mean(fr_list),
        "mean_CFScore": statistics.mean(cf_list),
        "sd_CFScore": statistics.stdev(cf_list) if len(cf_list) > 1 else 0,
        "min_CFScore": min(cf_list),
        "max_CFScore": max(cf_list),
    }
    out_path = RESULTS / "decoding_seeds.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n=== Decoding-seed variance (N_seeds={N_seeds}) ===")
    print(f"  F_T:  mean={summary['mean_F_T']:.1%}, SD={statistics.stdev(ft_list):.3f}")
    print(f"  F_R:  mean={summary['mean_F_R']:.1%}, SD={statistics.stdev(fr_list):.3f}")
    print(f"  CFScore: mean={summary['mean_CFScore']:.3f}, SD={summary['sd_CFScore']:.3f}")
    print(f"  CFScore range: [{summary['min_CFScore']:.3f}, {summary['max_CFScore']:.3f}]")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()