"""
run_multimodel.py — Cross-model generalisation of CF-Verify (Task 2 of plan).

Runs the headline CF-Verify protocol (gold labels, K=1) on the same N=60
question pool (30 HotpotQA + 30 2Wiki) with additional models via the codex
API, in addition to the existing GPT-5.4 numbers.

For each question: full / targeted / 1 random = 3 calls.

NOTE: Blocked on codex proxy quota — every model returned 402 "余额不足"
during testing. Run with: python scripts/run_multimodel.py
when the proxy is reloaded.

Usage:
    python scripts/run_multimodel.py
    python scripts/run_multimodel.py gpt-5.4-mini gpt-5.5  # explicit models
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
RESULTS.mkdir(exist_ok=True)


SYSTEM_PROMPT = (
    "You are a question-answering assistant. Answer the question based ONLY "
    "on the evidence provided. Give a concise answer (a few words). If the "
    "evidence does not support an answer, say exactly 'insufficient evidence'. "
    "Do not use any other knowledge."
)


def get_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Set OPENAI_API_KEY first (and OPENAI_BASE_URL if using a proxy).")
    base_url = os.environ.get("OPENAI_BASE_URL")
    return OpenAI(api_key=api_key, base_url=base_url)


def call_llm(client, model, question, evidence_sents, max_output=120, reasoning="low"):
    ev_text = "\n".join(f"[{i}] {s}" for i, s in enumerate(evidence_sents))
    user = f"Question: {question}\n\nEvidence:\n{ev_text}\n\nAnswer:"
    r = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        reasoning={"effort": reasoning},
        max_output_tokens=max_output,
    )
    return r.output_text.strip().split("\n")[0].strip().strip('"').strip(".").lower()


def load_questions():
    hot = json.loads((DATA / "hotpotqa_30_questions.json").read_text())
    wiki = json.loads((DATA / "2wiki_30_questions.json").read_text())
    return [("hotpotqa", q) for q in hot] + [("2wiki", q) for q in wiki]


def norm(s):
    return s.strip().lower().strip(".").strip('"').strip()


def main():
    if len(sys.argv) > 1:
        models = sys.argv[1:]
    else:
        # Default: two cheaper GPT-5.x variants in addition to gpt-5.4
        models = ["gpt-5.4-mini", "gpt-5.4-mini-openai-compact"]

    client = get_client()
    questions = load_questions()
    K = 1
    print(f"Models: {models}")
    print(f"Questions: {len(questions)}, K={K}")
    print(f"Per Q: 3 calls × {len(models)} models ≈ {3 * len(questions) * len(models)} API calls")

    for model in models:
        out_path = RESULTS / f"multimodel_{model.replace('/', '_').replace(':', '_')}.json"
        if out_path.exists():
            print(f"\n--- {model} (skipping, {out_path.name} exists) ---")
            continue

        print(f"\n=== {model} ===")
        t0 = time.time()
        records = []
        n_flip_t = 0
        n_flip_r = 0
        n_total = 0
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

            full_a = call_llm(client, model, q["question"], full_ev)
            target_a = call_llm(client, model, q["question"], target_ev)

            seed = q["question_id"] * 1009 + 31 + 7
            local_rng = random.Random(seed)
            rr = set(local_rng.sample(non_gold, n_remove))
            rr_ev = [(i, s) for i, s in enumerate(all_sents) if i not in rr]
            rnd_a = call_llm(client, model, q["question"], rr_ev)

            t_flipped = (norm(full_a) != norm(target_a))
            r_flipped = (norm(full_a) != norm(rnd_a))
            full_cl = full_a.lower()
            abstained = full_cl.startswith("insufficient") or not full_a.strip()

            records.append({
                "dataset": ds_label,
                "qid": q["question_id"],
                "question": q["question"],
                "gold_answer": q["answer"],
                "full_answer": full_a,
                "target_removed_answer": target_a,
                "random_removed_answer": rnd_a,
                "abstained": abstained,
                "targeted_flipped": t_flipped,
                "random_flipped": r_flipped,
            })
            n_total += 1
            if not abstained:
                if t_flipped:
                    n_flip_t += 1
                if r_flipped:
                    n_flip_r += 1

            elapsed = time.time() - t0
            print(
                f"  [{qi:>2}/{len(questions)}] {ds_label:>8} q{q['question_id']:>3} "
                f"[{elapsed:>5.0f}s] full='{full_a[:25]}' targ='{target_a[:25]}' "
                f"tflip={t_flipped} rflip={r_flipped}"
            )

        f_t_rate = n_flip_t / n_total
        f_r_rate = n_flip_r / n_total
        cfscore = f_t_rate - f_r_rate
        summary = {
            "model": model,
            "K": K,
            "n_questions": n_total,
            "targeted_flipped": n_flip_t,
            "random_flipped": n_flip_r,
            "F_T": f_t_rate,
            "F_R": f_r_rate,
            "CFScore": cfscore,
            "total_seconds": time.time() - t0,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "records": records,
        }
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"\nWrote {out_path}")
        print(f"  {model}: F_T={f_t_rate:.1%}  F_R={f_r_rate:.1%}  CFScore={cfscore:+.3f}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
