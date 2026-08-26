"""
aggregate_300.py — Merge original 60 paired records with 240 new ones,
recompute detection AUROC, routing F1, ensemble, and CIs on the full
N=300 paired sample.

Outputs:
    results/aggregated_300.json
"""
import json, math, random
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support, roc_auc_score, average_precision_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import statistics

RES = Path("/root/gcy/cf/results")
DATA = Path("/root/gcy/cf/data")


def normalize_record(r, cond):
    """Return (cfscore, label) for one cond."""
    label = 1 if cond == "grounded" else 0
    if cond == "grounded":
        ft = int(r["grounded"]["targeted_flipped"])
        rnd_samples = r["grounded"]["per_K"]["5"]
    else:
        ft = int(r["ungrounded"]["targeted_flipped"])
        rnd_samples = r["ungrounded"]["per_K"]["5"]
    fr = sum(s["flipped"] for s in rnd_samples) / len(rnd_samples)
    return ft - fr, label


def load_ungrounded_for_extension(r):
    """For new ext records we don't have rewrites/ungrounded stored separately —
    they were stored as cond='grounded'/'ungrounded' top-level keys.
    Returns (question, cfscore_per_cond, label) per record."""
    out = []
    for cond in ("grounded", "ungrounded"):
        ft = int(r[cond]["targeted_flipped"])
        per_K = r[cond]["per_K"]["5"]
        fr = sum(s["flipped"] for s in per_K) / len(per_K)
        cf = ft - fr
        out.append((cond, cf))
    return out


def main():
    # Original 60
    orig = json.loads(open(RES / "grounded_ungrounded.json").read())
    orig_records = []
    for r in orig["records"]:
        sents = load_sents(r)
        ung = build_ungrounded(r, sents)
        cf_g, l_g = normalize_record(r, "grounded")
        cf_u, l_u = normalize_record(r, "ungrounded")
        orig_records.append({
            "dataset": r["dataset"], "qid": r["qid"],
            "grounded": {"cfscore": cf_g, "answer": r["grounded"]["full_answer"],
                         "rewritten": False},
            "ungrounded": {"cfscore": cf_u, "answer": r["ungrounded"]["full_answer"],
                           "rewritten": True},
        })

    # New 240
    ext = json.loads(open(RES / "paired_set_ext.json").read())
    ext_records = []
    for r in ext["records"]:
        conds = load_ungrounded_for_extension(r)
        d = {"dataset": r["dataset"], "qid": r["qid"]}
        for cond, cf in conds:
            d[cond] = {"cfscore": cf, "answer": r[cond]["full_answer"],
                        "rewritten": True}
        ext_records.append(d)

    all_records = orig_records + ext_records
    print(f"Total questions: {len(all_records)}")
    n_records = sum(2 for _ in all_records)
    print(f"Total records (paired): {n_records}")

    # Build flat arrays for detection analysis
    X = []  # features: cfscore, (later: BGE baseline comparison if avail)
    y = []
    grp = []  # question index for group-wise CV
    for qi, r in enumerate(all_records):
        for cond in ("grounded", "ungrounded"):
            X.append(r[cond]["cfscore"])
            y.append(1 if cond == "grounded" else 0)
            grp.append(qi)
    X = np.array(X); y = np.array(y); grp = np.array(grp)

    # ===== 5-fold CV: random split =====
    print("\n=== 5-fold CV (random split) ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aurocs = []
    for tr, te in skf.split(X, y):
        aurocs.append(roc_auc_score(y[te], X[te]))
    print(f"  CFScore AUROC: {np.mean(aurocs):.3f} ± {np.std(aurocs):.3f}")

    # ===== 5-fold CV: group-wise split =====
    print("\n=== 5-fold CV (group-wise, by question) ===")
    unique_groups = sorted(set(grp))
    rng = random.Random(42)
    rng.shuffle(unique_groups)
    folds = [unique_groups[i::5] for i in range(5)]
    grp_aurocs = []
    for fi, fold_groups in enumerate(folds):
        test_mask = np.isin(grp, fold_groups)
        train_mask = ~test_mask
        if len(set(y[test_mask])) < 2: continue
        au = roc_auc_score(y[test_mask], X[test_mask])
        grp_aurocs.append(au)
        print(f"  Fold {fi+1}: AUROC = {au:.3f}")
    print(f"  Mean: {np.mean(grp_aurocs):.3f} ± {np.std(grp_aurocs):.3f}")

    # ===== Bootstrap CI on pooled AUROC =====
    rng = random.Random(42)
    boot_aurocs = []
    for _ in range(1000):
        idx = rng.choices(range(len(y)), k=len(y))
        if len(set(y[idx])) < 2: continue
        boot_aurocs.append(roc_auc_score(y[idx], X[idx]))
    boot_aurocs.sort()
    ci_lo, ci_hi = boot_aurocs[25], boot_aurocs[975]
    print(f"  Pooled AUROC bootstrap 95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")

    # ===== Per-dataset stats =====
    print("\n=== Per-dataset AUROC ===")
    per_ds = {}
    for ds in ("hotpotqa", "2wiki", "musique"):
        mask = np.array([r["dataset"] == ds for r in all_records for _ in (0, 1)])
        if mask.sum() == 0: continue
        au = roc_auc_score(y[mask], X[mask])
        per_ds[ds] = au
        print(f"  {ds}: N_records={mask.sum()}, AUROC={au:.3f}")

    # ===== Routing experiment =====
    print("\n=== Routing utility (5-fold CV) ===")
    f1s_cf = []; f1s_bge = []; f1s_ensemble = []
    for tr, te in skf.split(X, y):
        # CF alone
        best = (-1, 0)
        for tau in np.linspace(X.min(), X.max(), 50):
            pred = (X[tr] >= tau).astype(int)
            if pred.sum() == 0: continue
            p,r,f,_ = precision_recall_fscore_support(y[tr], pred, average="binary", zero_division=0)
            if f > best[0]: best = (f, tau)
        pred_te = (X[te] >= best[1]).astype(int)
        _,_,f1,_ = precision_recall_fscore_support(y[te], pred_te, average="binary", zero_division=0)
        f1s_cf.append(f1)
    f1_cf = statistics.mean(f1s_cf); sd_cf = statistics.stdev(f1s_cf)
    print(f"  CF-Verify CFScore: F1 = {f1_cf:.3f} ± {sd_cf:.3f}")

    # Always ACCEPT baseline
    f1_always = 2*y.mean()/(1+y.mean())
    print(f"  Always ACCEPT: F1 = {f1_always:.3f}")

    summary = {
        "n_questions": len(all_records),
        "n_records": n_records,
        "n_origin": len(orig_records),
        "n_ext": len(ext_records),
        "y_mean": float(y.mean()),
        "random_split_auroc_mean": float(np.mean(aurocs)),
        "random_split_auroc_sd": float(np.std(aurocs)),
        "groupwise_auroc_mean": float(np.mean(grp_aurocs)),
        "groupwise_auroc_sd": float(np.std(grp_aurocs)),
        "pooled_auroc_bootstrap95": [float(ci_lo), float(ci_hi)],
        "per_dataset_auroc": per_ds,
        "routing_f1_cf": f1_cf,
        "routing_f1_cf_sd": sd_cf,
        "routing_f1_always": f1_always,
    }
    json.dump(summary, open(RES / "aggregated_300.json", "w"), indent=2)
    print(f"\nWrote {RES/'aggregated_300.json'}")
    return summary


def load_sents(r):
    fname = "hotpotqa_30_questions.json" if r["dataset"] == "hotpotqa" else "2wiki_30_questions.json"
    pool = json.loads((DATA / fname).read_text())
    for q in pool:
        if q["question_id"] == r["qid"]:
            return [(i, s) for d in q["documents"] for i, s in enumerate(d["sentences"])]
    return []


def build_ungrounded(r, sents):
    """For original 60: build ungrounded evidence using rewrites."""
    sents_text = [s for _, s in sents]
    out = list(sents_text)
    rewrites = r.get("rewrites", [])
    gold_idx = sorted(r["gold_indices"])
    for gi, rw in zip(gold_idx, rewrites):
        if 0 <= gi < len(out):
            out[gi] = rw
    return out


if __name__ == "__main__":
    main()