"""
build_musique_pool.py — Convert MuSiQue dev to the CF-Verify question-pool
format, then run the headline protocol (via run_larger_sample-style calls).

MuSiQue: 20 paragraphs/question, is_supporting flags at paragraph level.
CF-Verify works at sentence level, so we flatten paragraphs to sentences and
mark all sentences of supporting paragraphs as gold.

Output: data/musique_60_questions.json
"""
import json
from pathlib import Path

DATA = Path("/root/gcy/cf/data")


def main(n=60):
    rows = [json.loads(l) for l in open(DATA / "musique_ans_v1.0_dev.jsonl")]
    out = []
    qid = 1
    for row in rows:
        if not row.get("answerable", True):
            continue
        # Flatten: each paragraph becomes one "document"; split into sentences
        # crudely on '. ' to keep granularity closer to HotpotQA sentences.
        docs = []
        gold_indices = []
        flat_idx = 0
        for p in row["paragraphs"]:
            text = p["paragraph_text"].strip()
            if not text:
                continue
            # Split into sentences (simple heuristic)
            sents = [s.strip() + "." for s in text.split(". ") if s.strip()]
            docs.append({"title": p["title"], "sentences": sents})
            for s in sents:
                if p["is_supporting"]:
                    gold_indices.append(flat_idx)
                flat_idx += 1
        if not gold_indices or len(docs) < 4:
            continue
        out.append({
            "question_id": qid,
            "id": row["id"],
            "question": row["question"],
            "answer": row["answer"],
            "documents": docs,
            "gold_sentence_indices": gold_indices,
        })
        qid += 1
        if qid > n:
            break
    out_path = DATA / "musique_60_questions.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path} ({len(out)} questions)")
    # stats
    nsent = [sum(len(d['sentences']) for d in q['documents']) for q in out]
    ngold = [len(q['gold_sentence_indices']) for q in out]
    print(f"mean sentences: {sum(nsent)/len(nsent):.1f}, mean gold: {sum(ngold)/len(ngold):.1f}")


if __name__ == "__main__":
    main()