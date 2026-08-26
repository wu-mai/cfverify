"""
semantic_delta_detection.py — Re-do detection with semantic-distance Δ instead
of exact-match string equality (D3 of the new review).

Uses BAAI/bge-base-en-v1.5 (cached locally) to compute cosine distance between
the full-evidence answer and the targeted-deleted / random-deleted answer.
Reports the same detection metrics as detection_5fold.py but with the soft
metric.

Outputs:
    results/semantic_delta_5fold.json
    prints summary table

Usage:
    python scripts/semantic_delta_detection.py
"""
import json
import os
import random
from pathlib import Path

# Force offline mode using the local cache
os.environ["HF_HOME"] = "/root/autodl-tmp/migrate_backup/hf_cache"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    precision_recall_fscore_support, roc_auc_score,
)

HERE = Path(__file__).parent
ROOT = HERE.parent
RESULTS = ROOT / "results"


def build_records_with_sem(recs, model):
    """Return dict[(ds, qid, cond, K)] -> {'targeted_dist', 'random_dist', 'tgt_ans', 'rnd_ans', 'full_ans'}."""
    out = {}
    for r in recs:
        full_a = r["grounded"]["full_answer"]
        for cond in ("grounded", "ungrounded"):
            tgt_a = r[cond]["target_answer"]
            rnd_ans = r[cond]["per_K"]["5"][0]["answer"]
            embs = model.encode([full_a, tgt_a, rnd_ans], normalize_embeddings=True, show_progress_bar=False)
            tgt_dist = float(1.0 - np.dot(embs[0], embs[1]))
            rnd_dist = float(1.0 - np.dot(embs[0], embs[2]))
            out[(r["dataset"], r["qid"], cond, 5)] = {
                "tgt_dist": tgt_dist, "rnd_dist": rnd_dist,
                "tgt_ans": tgt_a, "rnd_ans": rnd_ans, "full_ans": full_a,
            }
    return out


def stratified_folds(recs, n_folds=5, seed=42):
    hot = [(r["dataset"], r["qid"]) for r in recs if r["dataset"] == "hotpotqa"]
    wiki = [(r["dataset"], r["qid"]) for r in recs if r["dataset"] == "2wiki"]
    rng = random.Random(seed); rng.shuffle(hot); rng.shuffle(wiki)
    folds = [[] for _ in range(n_folds)]
    for i, q in enumerate(hot): folds[i % n_folds].append(q)
    for i, q in enumerate(wiki): folds[i % n_folds].append(q)
    return folds


def cv_report(rows, folds, name, key_h_g, key_h_u, sign=1.0):
    """Score = sign * key. Pick τ on holdout, evaluate on test."""
    f1s, aurocs = [], []
    all_qids = sorted(set((k[0], k[1]) for k in rows.keys()))
    for fold_keys in folds:
        hold = set(fold_keys)
        h_g = [sign*rows[(q[0], q[1], "grounded", 5)][key_h_g] for q in fold_keys]
        h_u = [sign*rows[(q[0], q[1], "ungrounded", 5)][key_h_u] for q in fold_keys]
        test = [q for q in all_qids if q not in hold]
        test_g = [sign*rows[(q[0], q[1], "grounded", 5)][key_h_g] for q in test]
        test_u = [sign*rows[(q[0], q[1], "ungrounded", 5)][key_h_u] for q in test]
        y_h = np.array([1]*len(h_g) + [0]*len(h_u))
        y_t = np.array([1]*len(test_g) + [0]*len(test_u))
        s_h = np.array(h_g + h_u); s_t = np.array(test_g + test_u)
        best = (-1, None)
        for tau in np.arange(0.0, 1.001, 0.01):
            pred = (s_h >= tau).astype(int)
            p, r, f, _ = precision_recall_fscore_support(y_h, pred, average="binary", zero_division=0)
            if f > best[0]: best = (f, tau, p, r)
        f_best, tau, _, _ = best
        pred_t = (s_t >= tau).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y_t, pred_t, average="binary", zero_division=0)
        auroc = roc_auc_score(y_t, s_t)
        f1s.append(f); aurocs.append(auroc)
    print(f"  {name:>40}: F1={np.mean(f1s):.3f}±{np.std(f1s):.3f}  "
          f"AUROC={np.mean(aurocs):.3f}±{np.std(aurocs):.3f}")
    return {"f1_mean": float(np.mean(f1s)), "f1_sd": float(np.std(f1s)),
            "auroc_mean": float(np.mean(aurocs)), "auroc_sd": float(np.std(aurocs))}


def main():
    p = RESULTS / "grounded_ungrounded.json"
    data = json.loads(p.read_text())
    recs = data["records"]
    print(f"Loaded {len(recs)} records")

    print("Loading BAAI/bge-base-en-v1.5 from local cache...")
    model = SentenceTransformer("BAAI/bge-base-en-v1.5",
                                cache_folder="/root/autodl-tmp/migrate_backup/hf_cache")
    print("Encoding...")
    rows = build_records_with_sem(recs, model)

    # Distribution
    tgt_g = [rows[(r["dataset"], r["qid"], "grounded", 5)]["tgt_dist"] for r in recs]
    tgt_u = [rows[(r["dataset"], r["qid"], "ungrounded", 5)]["tgt_dist"] for r in recs]
    rnd_g = [rows[(r["dataset"], r["qid"], "grounded", 5)]["rnd_dist"] for r in recs]
    rnd_u = [rows[(r["dataset"], r["qid"], "ungrounded", 5)]["rnd_dist"] for r in recs]
    print(f"\nSemantic distance (1 - cosine, BGE-base):")
    print(f"  Targeted, grounded    : mean={np.mean(tgt_g):.3f} median={np.median(tgt_g):.3f}")
    print(f"  Targeted, ungrounded  : mean={np.mean(tgt_u):.3f} median={np.median(tgt_u):.3f}")
    print(f"  Random,   grounded    : mean={np.mean(rnd_g):.3f} median={np.median(rnd_g):.3f}")
    print(f"  Random,   ungrounded  : mean={np.mean(rnd_u):.3f} median={np.median(rnd_u):.3f}")

    folds = stratified_folds(recs, n_folds=5, seed=42)

    print("\n=== 5-fold CV with semantic-distance Δ (no API needed) ===")
    # Reformulate: predict "grounded" if targeted_dist is HIGH (deletion matters)
    # or equivalently, semantic CFScore = tgt_dist - rnd_dist (large = grounded)
    # Note: higher score = more grounded
    summary = {}
    summary["targeted_dist_only"] = cv_report(rows, folds, "Targeted distance only (higher = grounded)",
                                              "tgt_dist", "tgt_dist")
    summary["random_dist_only_inv"] = cv_report(rows, folds, "Random distance (inverted)",
                                                "rnd_dist", "rnd_dist", sign=-1.0)
    summary["semantic_cfscore"] = {}
    # Build semantic_cfscore rows
    rows_with_cf = {}
    for k, v in rows.items():
        v = dict(v)
        v["sem_cf"] = v["tgt_dist"] - v["rnd_dist"]
        rows_with_cf[k] = v
    summary["semantic_cfscore"] = cv_report(rows_with_cf, folds, "Semantic CFScore (tgt - rnd)",
                                              "sem_cf", "sem_cf")
    summary["baseline_exact_match_cfscore"] = {"note": "F1=0.692, AUROC=0.784 from detection_5fold.json"}

    out_path = RESULTS / "semantic_delta_5fold.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()