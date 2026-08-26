"""
detection_5fold.py — 5-fold cross-validated detection metrics.

For each fold:
  - 12 questions (24 paired records) held out
  - τ* selected on holdout (maximising F1 over fine grid Δτ = 0.05)
  - metrics reported on the remaining 48 questions (96 paired records)
Aggregate: mean ± SD across 5 folds.

Stratified by dataset (6 HotpotQA + 6 2Wiki in each holdout), seed 42.
"""
import json
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support, roc_auc_score,
)

HERE = Path(__file__).parent
ROOT = HERE.parent
RESULTS = ROOT / "results"


def build_rows(recs):
    """Return list of (dataset, qid, cond, cf_K5, cf_K1)."""
    rows = []
    for r in recs:
        for cond in ("grounded", "ungrounded"):
            tgt = int(r[cond]["targeted_flipped"])
            r5 = sum(int(s["flipped"]) for s in r[cond]["per_K"]["5"]) / len(r[cond]["per_K"]["5"])
            r1 = sum(int(s["flipped"]) for s in r[cond]["per_K"]["1"]) / len(r[cond]["per_K"]["1"])
            rows.append({
                "dataset": r["dataset"], "qid": r["qid"], "cond": cond,
                "cf5": tgt - r5, "cf1": tgt - r1,
                "tgt": tgt, "r5": r5,
            })
    return rows


def stratified_folds(recs, n_folds=5, seed=42):
    """Yield list of held-out question keys per fold, stratified by dataset."""
    hot = [(r["dataset"], r["qid"]) for r in recs if r["dataset"] == "hotpotqa"]
    wiki = [(r["dataset"], r["qid"]) for r in recs if r["dataset"] == "2wiki"]
    rng = random.Random(seed)
    rng.shuffle(hot)
    rng.shuffle(wiki)
    # Round-robin assignment
    folds = [[] for _ in range(n_folds)]
    for i, q in enumerate(hot):
        folds[i % n_folds].append(q)
    for i, q in enumerate(wiki):
        folds[i % n_folds].append(q)
    return folds


def report_fold(rows, holdout_keys, K):
    """Pick τ* on holdout, report on test. K = 'cf5' or 'cf1'."""
    holdout = [r for r in rows if (r["dataset"], r["qid"]) in set(holdout_keys)]
    test = [r for r in rows if (r["dataset"], r["qid"]) not in set(holdout_keys)]
    if not holdout or not test:
        return None
    y_h = np.array([1 if r["cond"] == "grounded" else 0 for r in holdout])
    y_t = np.array([1 if r["cond"] == "grounded" else 0 for r in test])
    s_h = np.array([r[K] for r in holdout])
    s_t = np.array([r[K] for r in test])
    # Fine τ grid
    best = (-1, None)
    for tau in np.arange(-1.0, 1.001, 0.05):
        pred = (s_h >= tau).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y_h, pred, average="binary", zero_division=0)
        if f > best[0]:
            best = (f, tau, p, r)
    f_best, tau, _, _ = best
    pred_t = (s_t >= tau).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y_t, pred_t, average="binary", zero_division=0)
    auroc = roc_auc_score(y_t, s_t)
    return {
        "tau_star": float(tau),
        "holdout_f1": float(f_best),
        "test_precision": float(p),
        "test_recall": float(r),
        "test_f1": float(f),
        "test_auroc": float(auroc),
        "n_holdout_records": len(holdout),
        "n_test_records": len(test),
    }


def main():
    p = RESULTS / "grounded_ungrounded.json"
    data = json.loads(p.read_text())
    recs = data["records"]
    print(f"Loaded {len(recs)} records (60 paired Q = 120 paired records)")

    rows = build_rows(recs)
    folds = stratified_folds(recs, n_folds=5, seed=42)
    print(f"Folds: {[len(f) for f in folds]} (stratified, seed 42)")

    out = {"K=5": [], "K=1": []}
    for K_label, key in [("K=5", "cf5"), ("K=1", "cf1")]:
        print(f"\n=== {K_label} ===")
        per_fold = []
        for fi, holdout_keys in enumerate(folds, 1):
            res = report_fold(rows, holdout_keys, key)
            print(f"  Fold {fi}: τ*={res['tau_star']:>5.2f}  "
                  f"hold F1={res['holdout_f1']:.3f}  "
                  f"test P={res['test_precision']:.3f} R={res['test_recall']:.3f} "
                  f"F1={res['test_f1']:.3f} AUROC={res['test_auroc']:.3f}")
            per_fold.append(res)
        out[K_label] = per_fold

        # Aggregate
        f1s = [f["test_f1"] for f in per_fold]
        aurocs = [f["test_auroc"] for f in per_fold]
        ps = [f["test_precision"] for f in per_fold]
        rs = [f["test_recall"] for f in per_fold]
        print(f"  --- {K_label} aggregate (mean ± SD across 5 folds) ---")
        print(f"  P   = {np.mean(ps):.3f} ± {np.std(ps):.3f}")
        print(f"  R   = {np.mean(rs):.3f} ± {np.std(rs):.3f}")
        print(f"  F1  = {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
        print(f"  AUROC = {np.mean(aurocs):.3f} ± {np.std(aurocs):.3f}")
        out[f"{K_label}_aggregate"] = {
            "precision_mean": float(np.mean(ps)), "precision_sd": float(np.std(ps)),
            "recall_mean": float(np.mean(rs)), "recall_sd": float(np.std(rs)),
            "f1_mean": float(np.mean(f1s)), "f1_sd": float(np.std(f1s)),
            "auroc_mean": float(np.mean(aurocs)), "auroc_sd": float(np.std(aurocs)),
        }

    out_path = RESULTS / "detection_5fold.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
