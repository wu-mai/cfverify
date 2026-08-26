"""
conditional_300.py — Stratified analysis on the 600-record paired set.

Compute AUROC and Routing F1 on:
  - All 600 records (primary result)
  - Evidence-sensitive subset: records where targeted deletion changes the
    answer (F_T = 1) — model behaviorally depends on the gold evidence
  - Evidence-insensitive subset: F_T = 0 (model did not flip when gold removed,
    suggesting parametric answer or redundant support)
  - Per-dataset
  - Per-dataset restricted to evidence-sensitive

Outputs:
    results/conditional_300.json
    prints summary
"""
import json, random
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
import statistics

RES = Path("/root/gcy/cf/results")


def load_all():
    """Return list of records with (cf, label, F_T, F_R, dataset, qid)."""
    orig = json.loads(open(RES / "grounded_ungrounded.json").read())
    ext = json.loads(open(RES / "paired_set_ext.json").read())

    records = []
    for r in orig["records"]:
        for cond in ("grounded", "ungrounded"):
            ft = int(r[cond]["targeted_flipped"])
            per_K = r[cond]["per_K"]["5"]
            fr = sum(s["flipped"] for s in per_K) / len(per_K)
            records.append({
                "dataset": r["dataset"], "qid": r["qid"],
                "cond": cond, "label": 1 if cond == "grounded" else 0,
                "ft": ft, "fr": fr, "cf": ft - fr,
                "source": "orig",
            })
    for r in ext["records"]:
        for cond in ("grounded", "ungrounded"):
            ft = int(r[cond]["targeted_flipped"])
            per_K = r[cond]["per_K"]["5"]
            fr = sum(s["flipped"] for s in per_K) / len(per_K)
            records.append({
                "dataset": r["dataset"], "qid": r["qid"],
                "cond": cond, "label": 1 if cond == "grounded" else 0,
                "ft": ft, "fr": fr, "cf": ft - fr,
                "source": "ext",
            })
    return records


def cv_auroc(records, group_by_qid=False):
    """5-fold CV AUROC. If group_by_qid, both cond records of a question
    stay in the same fold."""
    X = np.array([r["cf"] for r in records])
    y = np.array([r["label"] for r in records])
    if group_by_qid:
        # Group by (dataset, qid)
        groups = []
        grp_map = {}
        for i, r in enumerate(records):
            k = (r["dataset"], r["qid"])
            if k not in grp_map:
                grp_map[k] = len(groups)
                groups.append([])
            groups[grp_map[k]].append(i)
        # Split groups
        rng = random.Random(42)
        rng.shuffle(groups)
        folds = [groups[i::5] for i in range(5)]
        aucs = []
        for fi, fold_groups in enumerate(folds):
            test_idx = [i for g in fold_groups for i in g]
            if len(set(y[test_idx])) < 2: continue
            aucs.append(roc_auc_score(y[test_idx], X[test_idx]))
        return np.mean(aucs), np.std(aucs)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for tr, te in skf.split(X, y):
        if len(set(y[te])) < 2: continue
        aucs.append(roc_auc_score(y[te], X[te]))
    return np.mean(aucs), np.std(aucs)


def cv_routing_f1(records):
    X = np.array([r["cf"] for r in records])
    y = np.array([r["label"] for r in records])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1s = []
    for tr, te in skf.split(X, y):
        best = (-1, 0)
        for tau in np.linspace(X.min(), X.max(), 50):
            pred = (X[tr] >= tau).astype(int)
            if pred.sum() == 0: continue
            p,r,f,_ = precision_recall_fscore_support(y[tr], pred, average="binary", zero_division=0)
            if f > best[0]: best = (f, tau)
        pred_te = (X[te] >= best[1]).astype(int)
        _,_,f1,_ = precision_recall_fscore_support(y[te], pred_te, average="binary", zero_division=0)
        f1s.append(f1)
    return np.mean(f1s), np.std(f1s)


def main():
    records = load_all()
    print(f"Total records: {len(records)}")

    # Per-question F_T (same for both conds)
    q_ft = {}
    for r in records:
        k = (r["dataset"], r["qid"])
        if k not in q_ft:
            q_ft[k] = r["ft"]

    # Sensitive subset: questions with F_T == 1 (targeted deletion changed answer at least once)
    sensitive_keys = {k for k, v in q_ft.items() if v == 1}
    insensitive_keys = {k for k, v in q_ft.items() if v == 0}
    print(f"Questions: {len(q_ft)}; sensitive (F_T=1): {len(sensitive_keys)}; insensitive: {len(insensitive_keys)}")

    sensitive_recs = [r for r in records if (r["dataset"], r["qid"]) in sensitive_keys]
    insensitive_recs = [r for r in records if (r["dataset"], r["qid"]) in insensitive_keys]
    print(f"Sensitive records: {len(sensitive_recs)}; insensitive: {len(insensitive_recs)}")

    # Always-accept baseline
    y = np.array([r["label"] for r in records])
    f1_always = 2*y.mean()/(1+y.mean())
    print(f"\nAlways-accept baseline F1: {f1_always:.3f}\n")

    summary = {"n_records_total": len(records), "n_questions": len(q_ft),
               "n_sensitive_questions": len(sensitive_keys),
               "n_insensitive_questions": len(insensitive_keys),
               "always_accept_f1": float(f1_always)}

    # AUROC + F1 on subsets
    print(f"{'subset':>30} {'N_rec':>7} {'AUROC':>12} {'Routing F1':>12}")
    print("-" * 65)
    for name, subset in [
        ("ALL (primary)", records),
        ("Sensitive (F_T=1)", sensitive_recs),
        ("Insensitive (F_T=0)", insensitive_recs),
    ]:
        if len(subset) < 10: continue
        au_m, au_s = cv_auroc(subset)
        f1_m, f1_s = cv_routing_f1(subset)
        y_sub = np.array([r["label"] for r in subset])
        f1_always_sub = 2*y_sub.mean()/(1+y_sub.mean())
        print(f"{name:>30} {len(subset):>7} {au_m:>6.3f}±{au_s:.3f}  {f1_m:>6.3f}±{f1_s:.3f}  [always={f1_always_sub:.3f}]")
        summary[f"{name}_auroc"] = float(au_m)
        summary[f"{name}_auroc_sd"] = float(au_s)
        summary[f"{name}_routing_f1"] = float(f1_m)
        summary[f"{name}_routing_f1_sd"] = float(f1_s)

    # Per-dataset within sensitive subset
    print("\nPer-dataset within sensitive subset:")
    for ds in ("hotpotqa", "2wiki", "musique"):
        sub = [r for r in sensitive_recs if r["dataset"] == ds]
        if len(sub) < 10: continue
        au_m, au_s = cv_auroc(sub)
        f1_m, f1_s = cv_routing_f1(sub)
        y_sub = np.array([r["label"] for r in sub])
        f1_always_sub = 2*y_sub.mean()/(1+y_sub.mean())
        print(f"  {ds:>20} N={len(sub):>4} AUROC={au_m:>6.3f}±{au_s:.3f}  F1={f1_m:>6.3f}±{f1_s:.3f}  [always={f1_always_sub:.3f}]")
        summary[f"sensitive_{ds}_auroc"] = float(au_m)
        summary[f"sensitive_{ds}_routing_f1"] = float(f1_m)

    json.dump(summary, open(RES / "conditional_300.json", "w"), indent=2)
    print(f"\nWrote {RES/'conditional_300.json'}")


if __name__ == "__main__":
    main()