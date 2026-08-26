"""
run_content_matched.py — Content-matched random control on the 120 paired
grounded/ungrounded records. For each, run CF-Verify with the random subset
chosen as the k most-similar-to-gold evidence sentences (BGE ranking), rather
than uniform random.

This directly addresses reviewer #1 construct-validity: if CF-Verify's signal
is just "delete anything answer-related," then content-matched random will
flip answers nearly as often as targeted deletion, and CFScore will collapse.
If CFScore stays high, CF-Verify is genuinely sensitive to the predicted
support, not just to any answer-related text.

API calls: 120 records × 1 extra call = ~120 calls.
Output: results/content_matched.json
"""
import json
import os
import random
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


def load_evidence(s):
    fname = "hotpotqa_30_questions.json" if s["dataset"] == "hotpotqa" else "2wiki_30_questions.json"
    pool = json.loads((DATA / fname).read_text())
    for q in pool:
        if q["question_id"] == s["qid"]:
            sents = []
            for d in q["documents"]:
                for sent in d["sentences"]:
                    sents.append(sent)
            return sents
    return []


def build_ungrounded(s, sents):
    rewrites = s["rewrites"]
    sents = list(sents)
    for gi, rw in zip(sorted(s["gold_indices"]), rewrites):
        if 0 <= gi < len(sents):
            sents[gi] = rw
    return sents


def main():
    client = get_client()
    print("Loading BGE...")
    bge = SentenceTransformer("BAAI/bge-base-en-v1.5",
                              cache_folder="/root/autodl-tmp/migrate_backup/hf_cache")

    gu = json.loads(open(RESULTS / "grounded_ungrounded.json").read())
    pairs = gu["records"]
    print(f"{len(pairs)} records, will run content-matched random for each (~{len(pairs)} calls)")

    # Compute content-matched deleted sets in batch (encoding only)
    summary = []
    t0 = time.time()
    for qi, r in enumerate(pairs, 1):
        sents = load_evidence(r)
        if not sents:
            continue
        ung_sents = build_ungrounded(r, sents)
        gold_idx = sorted(r["gold_indices"])
        gold_concat = " ".join(sents[i] for i in gold_idx if 0 <= i < len(sents))

        for cond in ("grounded", "ungrounded"):
            ev = sents if cond == "grounded" else ung_sents
            n_remove = len(gold_idx)
            non_gold = [i for i in range(len(ev)) if i not in set(gold_idx)]
            if len(non_gold) <= n_remove:
                continue

            cand_idx = [i for i in non_gold if 0 <= i < len(ev)]
            cand_texts = [ev[i] for i in cand_idx]
            embs = bge.encode([gold_concat] + cand_texts,
                              normalize_embeddings=True, show_progress_bar=False)
            gold_emb = embs[0]
            cand_embs = embs[1:]
            sims = sorted(
                [(cand_idx[j], float(np.dot(gold_emb, cand_embs[j]))) for j in range(len(cand_idx))],
                key=lambda x: -x[1]
            )
            top_match = [i for i, _ in sims[:n_remove]]

            # Run model with content-matched deletion
            cm_ev = [(i, s) for i, s in enumerate(ev) if i not in set(top_match)]
            # We need the FULL answer (cond1) from the existing record
            full_ans = r[cond]["full_answer"]
            cm_ans = call_llm(client, "gpt-5.4", r["question"], cm_ev)
            cm_flip = norm(full_ans) != norm(cm_ans)

            summary.append({
                "dataset": r["dataset"], "qid": r["qid"], "cond": cond,
                "ft": int(r[cond]["targeted_flipped"]),
                "f_r_contentmatched": int(cm_flip),
                "top_match_indices": sorted(top_match),
                "f_r_uniform_existing": sum(s["flipped"] for s in r[cond]["per_K"]["5"]) / 5,
            })
        if qi % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{qi}/{len(pairs)}] [{elapsed:.0f}s] done", flush=True)
            json.dump(summary, open(RESULTS / "content_matched.json", "w"), indent=1)

    json.dump({"records": summary}, open(RESULTS / "content_matched.json", "w"), indent=2)

    # Aggregate
    ans_g = [r for r in summary if r["cond"] == "grounded"]
    ans_u = [r for r in summary if r["cond"] == "ungrounded"]
    f_t = sum(r["ft"] for r in ans_g) / len(ans_g)
    f_r_cm = sum(r["f_r_contentmatched"] for r in ans_g) / len(ans_g)
    f_r_un = sum(r["f_r_uniform_existing"] for r in ans_g) / len(ans_g)
    cf_cm = f_t - f_r_cm
    cf_un = f_t - f_r_un
    print(f"\n=== Content-matched random control on GROUNDED records ===")
    print(f"  F_T = {f_t:.1%}")
    print(f"  F_R (uniform random) = {f_r_un:.1%} → CFScore = {cf_un:+.3f}")
    print(f"  F_R (content-matched) = {f_r_cm:.1%} → CFScore = {cf_cm:+.3f}")
    print(f"  CFScore drop under content-matched: {cf_un - cf_cm:+.3f}")


if __name__ == "__main__":
    main()