"""
analyze_detection.py — compute detection metrics from grounded/ungrounded pairs.

Reads results/grounded_ungrounded.json and prints:
  - Per-condition F_T, F_R, CFScore (mean across K) for grounded vs ungrounded
  - Threshold sweep on CFScore: precision/recall/F1/AUROC
  - Also: K=1 threshold sweep (single random sample) vs K=5 (mean over 5 samples)

Usage:
    python scripts/analyze_detection.py
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support, roc_auc_score, average_precision_score,
)

HERE = Path(__file__).parent
ROOT = HERE.parent
RESULTS = ROOT / "results"


def cfscore_record(rec, label, K):
    """Compute CFScore for one record on (label, K).

    label in {"grounded", "ungrounded"}.
    K is the integer number of random samples to average.
    Returns (targeted_flip, random_flip_mean, cfscore, n_random_samples_used).
    """
    side = rec[label]
    tgt_flip = int(side["targeted_flipped"])
    samples = side["per_K"][str(K)]
    rnd_flips = [int(s["flipped"]) for s in samples]
    rnd_mean = sum(rnd_flips) / len(rnd_flips)
    return tgt_flip, rnd_mean, tgt_flip - rnd_mean, len(samples)


def sweep_threshold(cf_scores_grounded, cf_scores_ungrounded):
    """Sweep CFScore threshold and report detection metrics.

    Label convention: grounded → 1 (positive = grounded), ungrounded → 0.
    Decision: predict grounded if CFScore >= tau.
    """
    labels = np.array([1] * len(cf_scores_grounded) + [0] * len(cf_scores_ungrounded))
    scores = np.array(list(cf_scores_grounded) + list(cf_scores_ungrounded))

    if labels.sum() == 0 or labels.sum() == len(labels):
        print("  (single-class; AUROC undefined)")
        return None

    # AUROC: higher score = grounded
    try:
        auroc = roc_auc_score(labels, scores)
    except Exception as e:
        auroc = float("nan")
    try:
        auprc = average_precision_score(labels, scores)
    except Exception:
        auprc = float("nan")

    rows = []
    for tau in [0.0, 0.25, 0.5, 0.75, 1.0]:
        pred = (scores >= tau).astype(int)
        p, r, f, _ = precision_recall_fscore_support(
            labels, pred, average="binary", pos_label=1, zero_division=0
        )
        rows.append((tau, p, r, f))
    return {
        "auroc": auroc,
        "auprc": auprc,
        "sweep": rows,
        "n_grounded": int(labels.sum()),
        "n_ungrounded": int(len(labels) - labels.sum()),
    }


def main():
    p = RESULTS / "grounded_ungrounded.json"
    if not p.exists():
        print(f"missing {p}; run scripts/run_grounded_ungrounded.py first")
        return
    data = json.loads(p.read_text())
    recs = data["records"]
    print(f"Loaded {len(recs)} records (Ks = {data['Ks']})")

    # ----- Per-condition table -----
    print()
    print("=" * 90)
    print(f"{'Setting':<22} {'N':>3} {'F_T':>6} {'F_R (K=1)':>10} {'CFScore (K=1)':>14}")
    print("-" * 90)
    rows_for_threshold = {1: {"grounded": [], "ungrounded": []},
                          3: {"grounded": [], "ungrounded": []},
                          5: {"grounded": [], "ungrounded": []}}
    for label in ("grounded", "ungrounded"):
        fts, frs, cfs = [], [], []
        for r in recs:
            t, rm, cf, _ = cfscore_record(r, label, K=1)
            fts.append(t); frs.append(rm); cfs.append(cf)
        n = len(fts)
        ft_m = sum(fts) / n
        fr_m = sum(frs) / n
        cf_m = sum(cfs) / n
        print(f"{label:<22} {n:>3} {ft_m:>6.1%} {fr_m:>10.1%} {cf_m:>+14.3f}")
        rows_for_threshold[1][label] = cfs
        # also K=3, K=5
        for K in (3, 5):
            for r in recs:
                _, _, cf, _ = cfscore_record(r, label, K=K)
                rows_for_threshold[K][label].append(cf)

    # ----- Threshold sweep -----
    print()
    print("=" * 90)
    print("Threshold sweep on CFScore (predict grounded iff CFScore >= tau)")
    print("=" * 90)
    for K in (1, 3, 5):
        cf_g = rows_for_threshold[K]["grounded"]
        cf_u = rows_for_threshold[K]["ungrounded"]
        print(f"\nK={K}:")
        res = sweep_threshold(cf_g, cf_u)
        if res is None:
            continue
        print(f"  AUROC = {res['auroc']:.3f}   AUPRC = {res['auprc']:.3f}   "
              f"N grounded={res['n_grounded']}  N ungrounded={res['n_ungrounded']}")
        print(f"  {'tau':>5}  {'precision':>10}  {'recall':>8}  {'F1':>6}")
        for tau, p, r, f in res["sweep"]:
            print(f"  {tau:>5.2f}  {p:>10.3f}  {r:>8.3f}  {f:>6.3f}")

    # ----- Also report "answer does not change" as a simpler detector -----
    print()
    print("=" * 90)
    print("Simpler detector: predict grounded iff targeted deletion FLIPS the answer")
    print("(this is the K=1 binary rule; AUROC over CFScore is shown above for graded variants)")
    print("=" * 90)
    flips_g = [int(rec["grounded"]["targeted_flipped"]) for rec in recs]
    flips_u = [int(rec["ungrounded"]["targeted_flipped"]) for rec in recs]
    p = sum(flips_g) / len(flips_g) if flips_g else 0
    q = sum(flips_u) / len(flips_u) if flips_u else 0
    print(f"  Grounded   : targeted_flipped in {sum(flips_g)}/{len(flips_g)} = {p:.1%}")
    print(f"  Ungrounded : targeted_flipped in {sum(flips_u)}/{len(flips_u)} = {q:.1%}")
    print(f"  Gap = {p - q:.1%}  (interpret as: positive = grounded flagged correctly)")


if __name__ == "__main__":
    main()