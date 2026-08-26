"""
build_larger_pool.py — Convert the raw HuggingFace parquets to the JSON
question-pool format that run_larger_sample.py expects.

HotpotQA context is dict {"title": [...], "sentences": [[...]]}.
2wiki context is list of [title, sentences].
Output:
    data/hotpotqa_extra_250.json
    data/2wiki_extra_250.json
"""
import json
import pandas as pd
from pathlib import Path

DATA = Path("/root/gcy/cf/data")


def to_format(question_id, question, answer, docs, gold_indices):
    return {
        "question_id": question_id,
        "id": str(question_id),
        "question": question,
        "answer": answer,
        "documents": docs,
        "gold_sentence_indices": gold_indices,
    }


def build_hot():
    hot = pd.read_parquet(DATA / "hotpotqa_distractor_val.parquet")
    out = []
    next_id = 101
    for _, row in hot.iterrows():
        ctx = row["context"]  # dict with "title" list and "sentences" list-of-lists
        titles = list(ctx["title"])
        sents_per_title = list(ctx["sentences"])
        gold_pairs = row["supporting_facts"]  # dict with "title" and "sent_id"

        # Flatten to per-sentence indices
        flat = []
        flat_to_orig = []
        for ti, (title, sents) in enumerate(zip(titles, sents_per_title)):
            for si, sent in enumerate(sents):
                flat.append(sent)
                flat_to_orig.append((title, si))
        gold_indices = []
        sf_titles = gold_pairs["title"] if isinstance(gold_pairs, dict) else [t for t, _ in gold_pairs]
        sf_sids = gold_pairs["sent_id"] if isinstance(gold_pairs, dict) else [s for _, s in gold_pairs]
        for title, si in zip(sf_titles, sf_sids):
            for flat_idx, (t, s) in enumerate(flat_to_orig):
                if t == title and s == si and flat_idx not in gold_indices:
                    gold_indices.append(flat_idx)
                    break
        if not gold_indices:
            continue
        docs = [{"title": str(t), "sentences": list(s)} for t, s in zip(titles, sents_per_title)]
        rec = to_format(next_id, row["question"], row["answer"], docs, gold_indices)
        out.append(rec)
        next_id += 1
        if next_id > 350:
            break
    return out


def build_wiki():
    wiki = pd.read_parquet(DATA / "2wiki_dev.parquet")
    out = []
    next_id = 101
    for _, row in wiki.iterrows():
        # context is JSON string in 2wiki
        ctx_raw = row["context"]
        sf_raw = row["supporting_facts"]
        if isinstance(ctx_raw, str):
            try:
                ctx = json.loads(ctx_raw)
            except Exception:
                continue
        else:
            ctx = ctx_raw
        if isinstance(sf_raw, str):
            try:
                gold_pairs = json.loads(sf_raw)
            except Exception:
                continue
        else:
            gold_pairs = sf_raw
        if not isinstance(ctx, list):
            continue
        flat = []
        flat_to_orig = []
        for di, item in enumerate(ctx):
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            title = item[0]
            sentences = item[1]
            for si, sent in enumerate(sentences):
                flat.append(sent)
                flat_to_orig.append((title, si))
        gold_indices = []
        sf_titles = gold_pairs["title"] if isinstance(gold_pairs, dict) else [t for t, _ in gold_pairs]
        sf_sids = gold_pairs["sent_id"] if isinstance(gold_pairs, dict) else [s for _, s in gold_pairs]
        for title, si in zip(sf_titles, sf_sids):
            for flat_idx, (t, s) in enumerate(flat_to_orig):
                if t == title and s == si and flat_idx not in gold_indices:
                    gold_indices.append(flat_idx)
                    break
        if not gold_indices:
            continue
        ans = row["answer"]
        if isinstance(ans, dict):
            ans = ans.get("answer") or ans.get("text") or str(ans)
        elif isinstance(ans, str):
            try:
                aobj = json.loads(ans)
                ans = aobj.get("answer") or aobj.get("text") or str(aobj)
            except Exception:
                pass
        docs = [{"title": str(t), "sentences": list(s)} for t, s in ctx if isinstance(s, (list, tuple))]
        rec = to_format(next_id, row["question"], str(ans), docs, gold_indices)
        out.append(rec)
        next_id += 1
        if next_id > 350:
            break
    return out


def main():
    hot = build_hot()
    wiki = build_wiki()
    print(f"hot: {len(hot)}, wiki: {len(wiki)}")
    (DATA / "hotpotqa_extra_250.json").write_text(json.dumps(hot, indent=2))
    (DATA / "2wiki_extra_250.json").write_text(json.dumps(wiki, indent=2))
    print(f"Wrote {DATA/'hotpotqa_extra_250.json'} ({len(hot)} records)")
    print(f"Wrote {DATA/'2wiki_extra_250.json'} ({len(wiki)} records)")


if __name__ == "__main__":
    main()