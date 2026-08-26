"""
Construct grounded/ungrounded test set + run CF-Verify for detection metrics.

For each of 30 HotpotQA + 27 2Wiki questions, we construct TWO versions:
  - G (grounded):     full evidence set E (as in the original N=57 experiment)
  - U (ungrounded):   E but with gold supporting sentences REPLACED by semantically
                      related but incorrect facts (so the model cannot answer from
                      E and must fall back to parametric memory or abstain)

CF-Verify then runs the same 3+K protocol on each version:
  - full evidence answer
  - targeted deletion (gold removed)        # in G: flips to "insufficient" or other
                                            # in U: removal may flip or may not (no support to remove)
  - random deletion baseline                # K=1, K=3, K=5 sweep

Detection metric: a question is "flagged ungrounded" if CFScore < τ for various τ;
report AUROC, precision, recall, F1 against the synthetic G/U labels.

API key is read from env: OPENAI_API_KEY (and optional OPENAI_BASE_URL).
"""
import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

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
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Set OPENAI_API_KEY first (and OPENAI_BASE_URL if using a proxy).")
    base_url = os.environ.get("OPENAI_BASE_URL")
    return OpenAI(api_key=api_key, base_url=base_url)


def call_llm(client, model, question, evidence_sents, max_output=120, reasoning="low"):
    ev_text = "\n".join(f"[{i}] {s}" for i, s in enumerate(evidence_sents))
    user = (
        f"Question: {question}\n\nEvidence:\n{ev_text}\n\nAnswer:"
    )
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


def fabricate_ungrounded(client, model, q, all_sents, gold_indices):
    """Ask gpt-5.4 to rewrite each gold sentence to a semantically-adjacent
    but factually incorrect sentence. The rest of E is unchanged.

    The result is an E' where the gold support is present but wrong, so a model
    that follows E faithfully will produce a wrong or 'insufficient' answer
    rather than the gold answer. An ungrounded model that ignores E will still
    produce the gold answer (correctly, by parametric memory) — exactly the
    setting CF-Verify needs to flag.
    """
    n_gold = len(gold_indices)
    prompt = (
        f"You are helping construct an adversarial test set. Given a question, "
        f"its correct answer, and {n_gold} supporting evidence sentences, "
        f"rewrite EACH of those {n_gold} sentences to a new sentence that "
        f"(a) remains about the same entity/topic, (b) is internally plausible, "
        f"but (c) does NOT actually support the correct answer. The rewrites "
        f"should be the kind of distractor a real RAG system might retrieve.\n\n"
        f"Question: {q['question']}\n"
        f"Correct answer: {q['answer']}\n\n"
        f"Rewrite these {n_gold} sentences:\n"
    )
    for idx in gold_indices:
        prompt += f"[{idx}] {all_sents[idx]}\n"
    prompt += (
        f"\nOutput strictly a JSON object with a single field \"rewrites\" "
        f"that is a list of {n_gold} rewritten strings in the SAME order as "
        f"the input sentences."
    )
    r = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": (
                "You rewrite evidence sentences into plausible but non-supporting "
                "distractors. Output valid JSON only."
            )},
            {"role": "user", "content": prompt},
        ],
        reasoning={"effort": "medium"},
        max_output_tokens=800,
    )
    txt = r.output_text.strip()
    try:
        obj = json.loads(txt)
        rewrites = obj["rewrites"]
    except Exception:
        # Fallback: try to find a JSON object in the output
        m = re.search(r"\{[\s\S]*\}", txt)
        if m:
            try:
                obj = json.loads(m.group(0))
                rewrites = obj.get("rewrites", [])
            except Exception:
                rewrites = []
        else:
            rewrites = []
    if not isinstance(rewrites, list) or len(rewrites) != n_gold:
        # If parsing failed, mark this question as failed-construction
        return None
    return rewrites


def build_sents(q):
    sents = []
    for d in q["documents"]:
        for s in d["sentences"]:
            sents.append(s)
    return sents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--Ks", default="1,3,5")
    parser.add_argument("--datasets", default="hotpotqa,2wiki",
                        help="comma list")
    parser.add_argument("--max_questions", type=int, default=0)
    args = parser.parse_args()
    Ks = [int(k) for k in args.Ks.split(",") if k.strip()]
    client = get_client()

    qs = []
    if "hotpotqa" in args.datasets:
        hot = json.loads((DATA / "hotpotqa_30_questions.json").read_text())
        qs.extend([("hotpotqa", q) for q in hot])
    if "2wiki" in args.datasets:
        wiki = json.loads((DATA / "2wiki_30_questions.json").read_text())
        qs.extend([("2wiki", q) for q in wiki])
    if args.max_questions:
        qs = qs[: args.max_questions]

    print(f"[setup] {len(qs)} questions, Ks={Ks}, model={args.model}")
    all_records = []
    t_global = time.time()

    for qi, (ds, q) in enumerate(qs, 1):
        all_sents = build_sents(q)
        gold = sorted(q["gold_sentence_indices"])
        # ----- Step 1: fabricate ungrounded E' -----
        rewrites = fabricate_ungrounded(client, args.model, q, all_sents, gold)
        if rewrites is None:
            print(f"  [{qi}] {ds} q{q['question_id']}: rewrite failed, skip")
            continue
        e_prime = list(all_sents)
        for idx, rw in zip(gold, rewrites):
            e_prime[idx] = rw

        # ----- Step 2: run CF-Verify on both E and E' -----
        # For each evidence set, do: full / targeted (gold removed) / K random
        rec = {
            "dataset": ds, "qid": q["question_id"],
            "question": q["question"], "gold_answer": q["answer"],
            "gold_indices": gold, "rewrites": rewrites,
            "grounded": {}, "ungrounded": {},
        }
        rng = random.Random(q["question_id"] * 31 + 7)
        for label, ev in [("grounded", all_sents), ("ungrounded", e_prime)]:
            t = time.time()
            full_a = call_llm(client, args.model, q["question"], ev)
            target_ev = [s for i, s in enumerate(ev) if i not in gold]
            target_a = call_llm(client, args.model, q["question"], target_ev)
            per_K = {}
            for K in Ks:
                samples = []
                for k in range(K):
                    seed = q["question_id"] * 1009 + 31 * (k + 1) + 7
                    local_rng = random.Random(seed)
                    non_gold = [i for i in range(len(ev)) if i not in gold]
                    n_remove = min(len(gold), len(non_gold))
                    rr = set(local_rng.sample(non_gold, n_remove))
                    rr_ev = [s for i, s in enumerate(ev) if i not in rr]
                    a3 = call_llm(client, args.model, q["question"], rr_ev)
                    samples.append({
                        "k": k + 1, "removed": sorted(rr),
                        "answer": a3,
                        "flipped": (a3.strip().lower() != full_a.strip().lower()),
                    })
                per_K[str(K)] = samples
            rec[label] = {
                "full_answer": full_a,
                "target_answer": target_a,
                "targeted_flipped": (
                    target_a.strip().lower() != full_a.strip().lower()
                ),
                "per_K": per_K,
            }
            elapsed = time.time() - t
            print(f"  [{qi:>2}/{len(qs)}] {ds:>8} q{q['question_id']:>3} "
                  f"{label:<11} [{elapsed:>4.1f}s] "
                  f"full='{full_a[:30]}' targ='{target_a[:30]}'")

        all_records.append(rec)

    out = RESULTS / "grounded_ungrounded.json"
    out.write_text(json.dumps({
        "model": args.model,
        "Ks": Ks,
        "n_questions": len(all_records),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_seconds": time.time() - t_global,
        "records": all_records,
    }, indent=2))
    print(f"\nWrote {out}  (total {time.time()-t_global:.0f}s)")


if __name__ == "__main__":
    main()
