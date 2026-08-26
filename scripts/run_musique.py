"""
run_musique.py — Run the headline CF-Verify protocol on MuSiQue (third dataset,
Reviewer D1/D3). Gold supports, K=1, gpt-5.4 via codex proxy.

Outputs:
    results/musique_gpt54.json
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


def call_llm(client, model, question, evidence_sents, max_output=60, max_retries=10):
    ev_text = "\n".join(f"[{i}] {s}" for i, s in enumerate(evidence_sents))
    user = f"Question: {question}\n\nEvidence:\n{ev_text}\n\nAnswer:"
    for attempt in range(max_retries):
        try:
            r = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                max_output_tokens=max_output,
                timeout=120.0,
            )
            break
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = min(30 * (attempt + 1), 300)
            print(f"    [retry {attempt+1}] {str(e)[:70]} — sleep {wait}s", flush=True)
            time.sleep(wait)
    return r.output_text.strip().split("\n")[0].strip().strip('"').strip(".").lower()


def norm(s):
    return s.strip().lower().strip(".").strip('"').strip()


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Set OPENAI_API_KEY first.")
    base_url = os.environ.get("OPENAI_BASE_URL")
    client = OpenAI(api_key=api_key, base_url=base_url)

    questions = json.loads((DATA / "musique_60_questions.json").read_text())
    model = "gpt-5.4"
    K = 1
    print(f"Model: {model}, K={K}, N={len(questions)}")

    out_path = RESULTS / "musique_gpt54.json"
    records = []
    t0 = time.time()
    n_flip_t = n_flip_r = n_total = 0
    n_ans = 0
    for qi, q in enumerate(questions, 1):
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
        rng = random.Random(seed)
        rr = set(rng.sample(non_gold, n_remove))
        rr_ev = [(i, s) for i, s in enumerate(all_sents) if i not in rr]
        rnd_a = call_llm(client, model, q["question"], rr_ev)

        t_flip = norm(full_a) != norm(target_a)
        r_flip = norm(full_a) != norm(rnd_a)
        abst = full_a.startswith("insufficient") or not full_a.strip()
        records.append({
            "dataset": "musique", "qid": q["question_id"],
            "question": q["question"], "gold_answer": q["answer"],
            "full_answer": full_a, "target_removed_answer": target_a,
            "random_removed_answer": rnd_a,
            "abstained": abst, "targeted_flipped": t_flip, "random_flipped": r_flip,
        })
        n_total += 1
        if not abst:
            n_ans += 1
            if t_flip: n_flip_t += 1
            if r_flip: n_flip_r += 1
        if qi % 5 == 0:
            # checkpoint
            out_path.write_text(json.dumps({
                "model": model, "K": K, "n_done": qi,
                "records": records,
            }, indent=1))
            el = time.time() - t0
            print(f"  [{qi}/{len(questions)}] [{el:.0f}s] answered={n_ans} T={n_flip_t} R={n_flip_r}", flush=True)

    summary = {
        "model": model, "K": K, "n_questions": n_total,
        "n_answered": n_ans,
        "F_T": n_flip_t / n_ans if n_ans else 0,
        "F_R": n_flip_r / n_ans if n_ans else 0,
        "CFScore": (n_flip_t - n_flip_r) / n_ans if n_ans else 0,
        "records": records,
    }
    out_path.write_text(json.dumps(summary, indent=1))
    print(f"\nWrote {out_path}")
    print(f"Answered: {n_ans}/{n_total}")
    if n_ans:
        print(f"F_T={summary['F_T']:.1%} F_R={summary['F_R']:.1%} CFScore={summary['CFScore']:+.3f}")


if __name__ == "__main__":
    main()