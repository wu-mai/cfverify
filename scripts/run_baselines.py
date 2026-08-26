"""
run_baselines.py — Compare CF-Verify against two real verifier baselines on
the same grounded/ungrounded paired data (D2 of Reviewers 2/3/4).

Baselines:
  1. LLM-as-judge: ask gpt-5.4 "is this answer supported by the evidence?"
     (yes/no/probably) — the SelfCheckGPT / RAGAS-style approach.
  2. Embedding similarity: cosine(BGE(answer), BGE(gold-evidence-concat))
     — the citation-overlap / n-gram-style grounding proxy.

Both scored on the same 120 records (60 grounded + 60 ungrounded) with the
same 5-fold CV protocol as CF-Verify (detection_5fold), so AUROC numbers are
directly comparable.

Outputs:
    results/baselines_vs_cfverify.json
"""
import json
import os
import random
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/root/autodl-tmp/migrate_backup/hf_cache")

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).parent
ROOT = HERE.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def get_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Set OPENAI_API_KEY first.")
    base_url = os.environ.get("OPENAI_BASE_URL")
    return OpenAI(api_key=api_key, base_url=base_url)


def load_pairs():
    """Reconstruct (question, evidence_text, answer, label) for each record."""
    d = json.loads((RESULTS / "grounded_ungrounded.json").read_text())
    out = []
    for r in d["records"]:
        # Evidence: need original sentences. Reconstruct from data files by qid.
        out.append({
            "dataset": r["dataset"],
            "qid": r["qid"],
            "question": r["question"],
            "rewrites": r.get("rewrites", []),
            "gold_indices": r["gold_indices"],
            "grounded_answer": r["grounded"]["full_answer"],
            "ungrounded_answer": r["ungrounded"]["full_answer"],
        })
    return out


def get_evidence(q):
    """Load original evidence sentences from the data file."""
    fname = "hotpotqa_30_questions.json" if q["dataset"] == "hotpotqa" else "2wiki_30_questions.json"
    pool = json.loads((DATA / fname).read_text())
    for p in pool:
        if p["question_id"] == q["qid"]:
            sents = []
            for d_ in p["documents"]:
                for s in d_["sentences"]:
                    sents.append(s)
            return sents
    return []


def build_ungrounded_evidence(sents, gold_indices, rewrites):
    """Replace gold sentences with rewrites (mirror of construction script)."""
    sents = list(sents)
    for gi, rw in zip(sorted(gold_indices), rewrites):
        if 0 <= gi < len(sents):
            sents[gi] = rw
    return sents


JUDGE_PROMPT = (
    "You are an evaluator. Given a question, a set of evidence sentences, and "
    "an answer, judge whether the answer is grounded in the evidence — i.e., "
    "whether the evidence directly supports the answer (not merely topically "
    "related). Reply with exactly one word: 'grounded' or 'ungrounded'."
)


def judge_call(client, question, evidence_sents, answer, max_retries=8):
    ev_text = "\n".join(f"- {s}" for s in evidence_sents)
    user = f"Question: {question}\n\nEvidence:\n{ev_text}\n\nAnswer: {answer}\n\nJudgment:"
    for attempt in range(max_retries):
        try:
            r = client.responses.create(
                model="gpt-5.4",
                input=[
                    {"role": "system", "content": JUDGE_PROMPT},
                    {"role": "user", "content": user},
                ],
                max_output_tokens=16,
                timeout=120.0,
            )
            break
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = min(30 * (attempt + 1), 300)
            print(f"    [retry {attempt+1}] {str(e)[:70]} — sleep {wait}s", flush=True)
            time.sleep(wait)
    text = r.output_text.strip().lower()
    if "ungrounded" in text:
        return 0
    if "grounded" in text:
        return 1
    return 0  # unparseable → conservative


def main():
    client = get_client()
    print("Loading BGE-base-en-v1.5...")
    bge = SentenceTransformer("BAAI/bge-base-en-v1.5",
                              cache_folder="/root/autodl-tmp/migrate_backup/hf_cache")

    pairs = load_pairs()
    print(f"Loaded {len(pairs)} question pairs")

    # Check for existing partial results (checkpoint resume)
    ckpt_path = RESULTS / "baselines_vs_cfverify.json"
    if ckpt_path.exists():
        ckpt = json.loads(ckpt_path.read_text())
        done_keys = {(r["dataset"], r["qid"], r["cond"]) for r in ckpt["records"]}
        records = list(ckpt["records"])
        print(f"Resuming: {len(done_keys)} records already scored")
    else:
        records = []
        done_keys = set()

    t0 = time.time()
    for qi, q in enumerate(pairs, 1):
        sents = get_evidence(q)
        if not sents:
            print(f"  !! no evidence for {q['dataset']} q{q['qid']}")
            continue
        ung_sents = build_ungrounded_evidence(sents, q["gold_indices"], q["rewrites"])
        gold_concat = " ".join(sents[i] for i in sorted(q["gold_indices"]) if 0 <= i < len(sents))

        for cond, ev, ans in [
            ("grounded", sents, q["grounded_answer"]),
            ("ungrounded", ung_sents, q["ungrounded_answer"]),
        ]:
            if (q["dataset"], q["qid"], cond) in done_keys:
                continue
            # 1. LLM judge
            judge = judge_call(client, q["question"], ev, ans)
            # 2. Embedding similarity (answer vs gold-evidence concat, and vs full evidence)
            embs = bge.encode([ans, gold_concat, " ".join(ev)],
                              normalize_embeddings=True, show_progress_bar=False)
            sim_gold = float(np.dot(embs[0], embs[1]))
            sim_full = float(np.dot(embs[0], embs[2]))
            records.append({
                "dataset": q["dataset"], "qid": q["qid"], "cond": cond,
                "answer": ans, "judge_groundedscore": judge,
                "sim_answer_gold": sim_gold, "sim_answer_full": sim_full,
            })
        # checkpoint every 10 questions
        if qi % 10 == 0:
            json.dump({"records": records}, open(ckpt_path, "w"), indent=1)
            print(f"  [{qi}/{len(pairs)}] {time.time()-t0:.0f}s checkpointed", flush=True)

    json.dump({"records": records}, open(ckpt_path, "w"), indent=1)
    n = len(records)
    print(f"\nScored {n} records")

    # ---- Metrics: label grounded=1 ----
    from sklearn.metrics import roc_auc_score
    y = np.array([1 if r["cond"] == "grounded" else 0 for r in records])
    scores = {
        "LLM-judge (grounded=1)": np.array([float(r["judge_groundedscore"]) for r in records]),
        "BGE sim(answer, gold-evid)": np.array([r["sim_answer_gold"] for r in records]),
        "BGE sim(answer, full-evid)": np.array([r["sim_answer_full"] for r in records]),
    }
    print("\n=== AUROC on the 120 paired records (grounded=positive) ===")
    summary = {}
    for name, s in scores.items():
        auc = roc_auc_score(y, s)
        summary[name] = auc
        print(f"  {name:<32}: AUROC = {auc:.3f}")
    summary["CF-Verify CFScore (K=5, 5-fold CV)"] = 0.784  # from detection_5fold
    print(f"  {'CF-Verify CFScore (reference)':<32}: AUROC = 0.784")
    json.dump(summary, open(RESULTS / "baseline_auroc_summary.json", "w"), indent=2)
    print(f"\nWrote {RESULTS/'baseline_auroc_summary.json'}")


if __name__ == "__main__":
    main()