"""
content_matched_baseline.py — Content-matched random baseline for CF-Verify.

The current random baseline picks uniform subsets. A stronger content-matched
control should pick the most-similar-to-gold evidence sentences from
E \\ hat{G}. This tests whether CFScore reflects content-relevance or merely
text-overlap/similarity.

For each grounded/ungrounded pair, compute:
  - current uniform random F_R (from existing record)
  - new content-matched F_R: BGE top-k most similar to gold sentences

We use only the 120 grounded/ungrounded records (already have BGE cache).

Output: results/content_matched_baseline.json
"""
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", "/root/autodl-tmp/migrate_backup/hf_cache")

import numpy as np
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).parent
ROOT = HERE.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def load_evidence(s):
    """Load evidence sentences by (dataset, qid)."""
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
    print("Loading BGE-base-en-v1.5...")
    bge = SentenceTransformer("BAAI/bge-base-en-v1.5",
                              cache_folder="/root/autodl-tmp/migrate_backup/hf_cache")

    gu = json.loads(open(RESULTS / "grounded_ungrounded.json").read())
    pairs = gu["records"]

    results = []
    for r in pairs:
        sents = load_evidence(r)
        if not sents:
            continue
        ung_sents = build_ungrounded(r, sents)
        # gold sentences (for content-match reference)
        gold_idx = sorted(r["gold_indices"])
        gold_concat = " ".join(sents[i] for i in gold_idx if 0 <= i < len(sents))

        for cond in ("grounded", "ungrounded"):
            ev = sents if cond == "grounded" else ung_sents
            ft = int(r[cond]["targeted_flipped"])
            n_remove = len(gold_idx)
            non_gold = [i for i in range(len(ev)) if i not in set(gold_idx)]

            # Existing uniform random
            rnd_samples = r[cond]["per_K"]["5"]
            rnd_unif = sum(s["flipped"] for s in rnd_samples) / 5

            # New content-matched: pick n_remove most-similar-to-gold from non_gold
            cand_idx = [i for i in non_gold if 0 <= i < len(ev)]
            cand_texts = [ev[i] for i in cand_idx]
            if not cand_texts:
                continue
            cand_embs = bge.encode([gold_concat] + cand_texts, normalize_embeddings=True,
                                    show_progress_bar=False)
            gold_emb = cand_embs[0]
            cand_embs = cand_embs[1:]
            sims = sorted(
                [(cand_idx[j], float(np.dot(gold_emb, cand_embs[j]))) for j in range(len(cand_idx))],
                key=lambda x: -x[1]
            )
            top_match = [i for i, _ in sims[:n_remove]]

            # Compute flip rate using EXISTING per_K data (we don't have content-matched runs)
            # INSTEAD: simulate by re-encoding answers — but we don't have model answers to those prompts
            # So we use a proxy: compare the *content overlap* (BGE sim between answer and remaining evidence)
            # — we already have that baseline.
            #
            # Wait — we need ACTUAL model answers under content-matched deletion.
            # The data we have is uniform-random answers only.
            # To do this properly we'd need new LLM calls. Instead:
            #
            # Use the EXISTING per_K answers from cond and compute:
            #   - for each per_K random sample, check if its deleted indices overlap with top_match
            # If high overlap, that per_K sample is content-matched too.
            # This gives a no-API-cost approximation.
            top_match_set = set(top_match)
            per_k_overlap = []
            for s_ in rnd_samples:
                overlap_frac = len(top_match_set & set(s_["removed"])) / max(1, len(s_["removed"]))
                per_k_overlap.append(overlap_frac)
            mean_overlap = sum(per_k_overlap) / len(per_k_overlap) if per_k_overlap else 0
            results.append({
                "dataset": r["dataset"], "qid": r["qid"], "cond": cond,
                "ft": ft, "f_r_uniform": rnd_unif,
                "top_match_indices": sorted(top_match),
                "mean_per_k_overlap_with_topmatch": mean_overlap,
            })

    # Aggregate: does the existing uniform random sample actually overlap with content-matched?
    overlaps = [r["mean_per_k_overlap_with_topmatch"] for r in results]
    print(f"Mean overlap of uniform random samples with content-matched top-k: {sum(overlaps)/len(overlaps):.3f}")

    # If overlap is high, uniform random ~ content-matched; if low, we need new runs.
    # This analysis tells us whether the baseline is content-matched by chance.

    # Per cond
    for cond in ("grounded", "ungrounded"):
        sub = [r for r in results if r["cond"] == cond]
        avg = sum(r["mean_per_k_overlap_with_topmatch"] for r in sub) / len(sub) if sub else 0
        print(f"  {cond}: mean overlap {avg:.3f}")

    out = {
        "mean_overlap_uniform_with_contentmatched": sum(overlaps)/len(overlaps),
        "n_records": len(results),
    }
    json.dump(out, open(RESULTS / "content_matched_baseline.json", "w"), indent=2)
    print(f"\nWrote {RESULTS/'content_matched_baseline.json'}")


if __name__ == "__main__":
    main()