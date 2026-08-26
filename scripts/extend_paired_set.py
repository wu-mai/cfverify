"""
extend_paired_set.py — Extend the grounded/ungrounded paired detection set
from 60 to 300 questions (addresses review rounds 1-3 sample-size criticism).

New questions: 80 HotpotQA (qid 101-180 from extra_250 pool) +
80 2Wiki (qid 101-180) + 80 MuSiQue (qid 61-140, new from dev split).

For each new question:
  1. Rewrite gold sentences (ungrounded construction) — 1 call
  2. Full CF-Verify protocol on both grounded and ungrounded evidence
     (full + targeted + K=5 random) — 2 × (1 + 1 + 5) = 14 calls
Total ≈ 240 × 15 = 3600 calls. Run in three background stages with
checkpointing every 10 questions.

Outputs:
    results/paired_set_ext.json  (checkpointed; merged with original later)
"""
import json
import os
import random
import re
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

REWRITE_PROMPT = (
    "Rewrite each numbered sentence so that it keeps the same topic, entities, "
    "and style, but no longer supports answering the question. Change the "
    "specific fact that connects to the answer (e.g., swap a date, location, "
    "or relation) so the sentence remains fluent and plausible but is "
    "irrelevant for the question. Output ONLY the rewrites, numbered the same "
    "as the input."
)


def call_llm(client, model, question, evidence_sents, max_output=80, max_retries=10):
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


def rewrite_gold(client, model, question, gold_sents, max_retries=10):
    text = "\n".join(f"[{i}] {s}" for i, s in enumerate(gold_sents))
    user = f"Question: {question}\n\nSentences to rewrite:\n{text}\n\nRewrites:"
    for attempt in range(max_retries):
        try:
            r = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": REWRITE_PROMPT},
                    {"role": "user", "content": user},
                ],
                max_output_tokens=600,
                timeout=120.0,
            )
            break
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = min(30 * (attempt + 1), 300)
            time.sleep(wait)
    raw = r.output_text.strip()
    # Parse numbered rewrites
    rewrites = []
    for line in raw.split("\n"):
        m = re.match(r"\s*\[?(\d+)\]?\s*[.:)]?\s*(.+)", line)
        if m:
            rewrites.append(m.group(2).strip())
    if len(rewrites) < len(gold_sents):
        # fallback: pad with originals (will weaken but not crash)
        while len(rewrites) < len(gold_sents):
            rewrites.append(gold_sents[len(rewrites)])
    return rewrites[:len(gold_sents)]


def norm(s):
    return s.strip().lower().strip(".").strip('"').strip()


def load_new_questions():
    """Load 240 new questions: 80 hot + 80 wiki (qid 101-180) + 80 musique (qid 61-140)."""
    out = []
    hot = json.loads((DATA / "hotpotqa_extra_250.json").read_text())
    hot_new = [q for q in hot if 101 <= q["question_id"] <= 180]
    for q in hot_new:
        out.append(("hotpotqa", q))
    wiki = json.loads((DATA / "2wiki_extra_250.json").read_text())
    wiki_new = [q for q in wiki if 101 <= q["question_id"] <= 180]
    for q in wiki_new:
        out.append(("2wiki", q))
    # MuSiQue: build 80 more (qid 61-140)
    mus_rows = [json.loads(l) for l in open(DATA / "musique_ans_v1.0_dev.jsonl")]
    mus_pool = json.loads((DATA / "musique_60_questions.json").read_text())
    used_ids = {q["id"] for q in mus_pool}
    added = 0
    qid = 61
    for row in mus_rows:
        if row["id"] in used_ids or not row.get("answerable", True):
            continue
        docs, gold_idx = [], []
        flat = 0
        for p in row["paragraphs"]:
            text = p["paragraph_text"].strip()
            if not text:
                continue
            sents = [s.strip() + "." for s in text.split(". ") if s.strip()]
            docs.append({"title": p["title"], "sentences": sents})
            for s in sents:
                if p["is_supporting"]:
                    gold_idx.append(flat)
                flat += 1
        if not gold_idx or len(docs) < 4:
            continue
        out.append(("musique", {
            "question_id": qid, "id": row["id"], "question": row["question"],
            "answer": row["answer"], "documents": docs,
            "gold_sentence_indices": gold_idx,
        }))
        qid += 1
        added += 1
        if added >= 80:
            break
    return out


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Set OPENAI_API_KEY first.")
    client = OpenAI(api_key=api_key, base_url=os.environ.get("OPENAI_BASE_URL"))
    model = "gpt-5.4"
    K = 5

    questions = load_new_questions()
    print(f"New questions: {len(questions)} ({sum(1 for d,_ in questions if d=='hotpotqa')} hot, "
          f"{sum(1 for d,_ in questions if d=='2wiki')} wiki, {sum(1 for d,_ in questions if d=='musique')} musique)")

    out_path = RESULTS / "paired_set_ext.json"
    records = []
    done = set()
    if out_path.exists():
        ckpt = json.loads(out_path.read_text())
        records = ckpt.get("records", [])
        done = {(r["dataset"], r["qid"]) for r in records}
        print(f"Resuming: {len(done)} done")

    t0 = time.time()
    for qi, (ds, q) in enumerate(questions, 1):
        if (ds, q["question_id"]) in done:
            continue
        all_sents = []
        for d in q["documents"]:
            for s in d["sentences"]:
                all_sents.append(s)
        gold = set(q["gold_sentence_indices"])
        non_gold = [i for i in range(len(all_sents)) if i not in gold]
        n_remove = min(len(gold), len(non_gold))
        gold_list = sorted(gold)

        # 1. Rewrite gold sentences
        gold_sents = [all_sents[i] for i in gold_list if 0 <= i < len(all_sents)]
        rewrites = rewrite_gold(client, model, q["question"], gold_sents)
        # 2. Build ungrounded evidence
        ung_sents = list(all_sents)
        for gi, rw in zip(gold_list, rewrites):
            if 0 <= gi < len(ung_sents):
                ung_sents[gi] = rw

        rec = {"dataset": ds, "qid": q["question_id"], "question": q["question"],
               "gold_answer": q["answer"], "gold_indices": gold_list, "rewrites": rewrites}
        for cond, ev in [("grounded", all_sents), ("ungrounded", ung_sents)]:
            full_ev = [(i, s) for i, s in enumerate(ev)]
            full_a = call_llm(client, model, q["question"], full_ev)
            tgt_ev = [(i, s) for i, s in enumerate(ev) if i not in gold]
            tgt_a = call_llm(client, model, q["question"], tgt_ev)
            per_K = []
            for k in range(K):
                seed = q["question_id"] * 1009 + 31 * (k + 1) + 7 + (17 if cond == "ungrounded" else 0)
                rng = random.Random(seed)
                rr = set(rng.sample(non_gold, n_remove))
                rr_ev = [(i, s) for i, s in enumerate(ev) if i not in rr]
                ra = call_llm(client, model, q["question"], rr_ev)
                per_K.append({"k": k + 1, "seed": seed, "removed": sorted(rr),
                              "answer": ra, "flipped": norm(ra) != norm(full_a)})
            rec[cond] = {
                "full_answer": full_a, "target_answer": tgt_a,
                "targeted_flipped": norm(tgt_a) != norm(full_a),
                "per_K": {str(K): per_K},
            }
        records.append(rec)
        if len(records) % 10 == 0:
            out_path.write_text(json.dumps({"records": records}, indent=1))
            el = time.time() - t0
            print(f"  [{len(records)}/{len(questions)}] [{el:.0f}s] checkpointed", flush=True)

    out_path.write_text(json.dumps({"records": records}, indent=1))
    print(f"\nDONE: {len(records)} records in {time.time()-t0:.0f}s -> {out_path}")


if __name__ == "__main__":
    main()