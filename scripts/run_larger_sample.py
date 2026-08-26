"""
run_larger_sample.py — N-extension of the headline CF-Verify protocol.

Adds 60 NEW HotpotQA + 60 NEW 2Wiki questions (beyond the existing 30+30)
to push the headline sample from N=57 to N=177 (paired 354 records).
This addresses Reviewer 3 (and 2)'s D1 "expand evaluation scale to N>=500".

For each new question, runs the same headline protocol (K=1):
  - cond1: full evidence answer
  - cond2: gold supporting sentences removed (targeted)
  - cond3: matched random deletion (size |gold|)

API key from env: OPENAI_API_KEY (+ optional OPENAI_BASE_URL).

Usage:
    export OPENAI_API_KEY=...; export OPENAI_BASE_URL=...
    python scripts/run_larger_sample.py           # default: 60+60 NEW
    python scripts/run_larger_sample.py 30         # smaller test
"""
import argparse
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
        sys.exit("Set OPENAI_API_KEY first.")
    return OpenAI(api_key=api_key, base_url=os.environ.get("OPENAI_BASE_URL"))


def call_llm(client, model, question, evidence_sents, max_output=80, reasoning="low"):
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
    return r.output_text.strip()


def norm(s):
    return s.strip().lower().strip(".").strip('"').strip()


def load_new_questions(n_hot=60, n_wiki=60, skip_qids=None):
    """Load NEW questions (question_id > 30) from extra_250 files."""
    skip_qids = set(skip_qids or [])
    hot = json.loads((DATA / "hotpotqa_extra_250.json").read_text())
    wiki = json.loads((DATA / "2wiki_extra_250.json").read_text())
    # Filter to ids > 30 and not in skip list
    hot_new = [q for q in hot if q["question_id"] > 30 and q["question_id"] not in skip_qids][:n_hot]
    wiki_new = [q for q in wiki if q["question_id"] > 30 and q["question_id"] not in skip_qids][:n_wiki]
    print(f"Loaded {len(hot_new)} new HotpotQA + {len(wiki_new)} new 2Wiki (skipped qids: {sorted(skip_qids)[:5]}{'...' if len(skip_qids) > 5 else ''})")
    return [("hotpotqa", q) for q in hot_new] + [("2wiki", q) for q in wiki_new]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-hot", type=int, default=60)
    ap.add_argument("--n-wiki", type=int, default=60)
    ap.add_argument("--model", default="gpt-5.4")
    ap.add_argument("--k", type=int, default=1, dest="K")
    ap.add_argument("--out", default="larger_sample_gpt54.json")
    ap.add_argument("--skip-qids", default="31-70", help="qid range to skip (already in ls80)")
    args = ap.parse_args()

    # Parse skip-qids
    skip = set()
    if args.skip_qids:
        for part in args.skip_qids.split(","):
            if "-" in part:
                lo, hi = map(int, part.split("-"))
                skip.update(range(lo, hi + 1))
            else:
                skip.add(int(part))

    client = get_client()
    questions = load_new_questions(args.n_hot, args.n_wiki, skip_qids=skip)
    K = args.K
    print(f"Model: {args.model}, K={K}, N={len(questions)}")
    print(f"Per Q: 2+K = {2+K} calls → {len(questions)*(2+K)} API calls total")

    out_path = RESULTS / args.out
    records = []
    t_global = time.time()
    n_flip_t = 0; n_flip_r = 0; n_total = 0

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

        try:
            full_a = call_llm(client, args.model, q["question"], full_ev)
        except Exception as e:
            print(f"  [q {qi}] full ERR: {str(e)[:80]}")
            full_a = ""
        try:
            target_a = call_llm(client, args.model, q["question"], target_ev)
        except Exception as e:
            print(f"  [q {qi}] target ERR: {str(e)[:80]}")
            target_a = full_a  # don't credit flip on error

        rnd_answers = []
        for k in range(K):
            seed = q["question_id"] * 1009 + 31 * (k + 1) + 7
            local_rng = random.Random(seed)
            rr = set(local_rng.sample(non_gold, n_remove))
            rr_ev = [(i, s) for i, s in enumerate(all_sents) if i not in rr]
            try:
                rnd_a = call_llm(client, args.model, q["question"], rr_ev)
            except Exception as e:
                rnd_a = full_a
            rnd_answers.append({
                "k": k + 1, "seed": seed,
                "removed": sorted(rr), "answer": rnd_a,
                "flipped": norm(rnd_a) != norm(full_a),
            })

        t_flipped = norm(target_a) != norm(full_a)
        full_cl = full_a.lower()
        abstained = full_cl.startswith("insufficient") or not full_a.strip()

        records.append({
            "dataset": ds_label, "qid": q["question_id"],
            "question": q["question"], "gold_answer": q["answer"],
            "full_answer": full_a, "target_answer": target_a,
            "abstained": abstained,
            "targeted_flipped": t_flipped,
            "per_K": {str(K): rnd_answers},
        })
        n_total += 1
        if not abstained:
            if t_flipped: n_flip_t += 1
            for s in rnd_answers:
                if s["flipped"]: n_flip_r += 1

        # Periodic save
        if qi % 5 == 0 or qi == len(questions):
            elapsed = time.time() - t_global
            summary_partial = {
                "model": args.model, "K": K, "n_questions": len(questions),
                "n_answered_so_far": sum(1 for r in records if not r["abstained"]),
                "n_targeted_flip": n_flip_t,
                "n_random_flip": n_flip_r,
                "elapsed": elapsed,
                "records": records,
            }
            out_path.write_text(json.dumps(summary_partial, indent=2))
            print(f"  [{qi:>3}/{len(questions)}] {ds_label:>8} q{q['question_id']:>3} "
                  f"[{elapsed:>5.0f}s, ETA {elapsed/qi*(len(questions)-qi):>5.0f}s] "
                  f"answered={n_total}  flips T={n_flip_t} R={n_flip_r}")

    print(f"\n=== Summary ===")
    print(f"  Model: {args.model}, K={K}, N={len(questions)}")
    print(f"  Answered: {n_total}/{len(questions)}")
    print(f"  Targeted flip: {n_flip_t}/{n_total}")
    print(f"  Random flip:   {n_flip_r}/{n_total*K}")
    cf = (n_flip_t/n_total - n_flip_r/(n_total*K)) if n_total else 0
    print(f"  CFScore: {cf:+.3f}")
    print(f"  Total time: {time.time()-t_global:.0f}s")
    print(f"  Wrote {out_path}")


if __name__ == "__main__":
    main()