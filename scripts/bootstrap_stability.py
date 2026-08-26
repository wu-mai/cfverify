"""
bootstrap_stability.py — bootstrap / LOO / N-vs-CFScore convergence analysis for CF-Verify.

Purpose: answer reviewer W1/D1 (sample size N=57) by showing the headline CFScore
is statistically stable. Reads existing flip-state data from results/ — does NOT
invoke any LLM.

Run:
    python scripts/bootstrap_stability.py

Outputs:
    results/bootstrap_summary.json
    figures/bootstrap_convergence.pdf  (or .png if matplotlib has no usetex)
"""
import json
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)


def wilson_95ci(k, n):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    z = 1.96
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def load_flip_states():
    """Return list of (dataset, qid, targeted_flip, random_flip) tuples.

    `targeted_flip`: 1 if gold-removed answer differs from full, else 0.
    `random_flip`:   1 if random-removed answer differs from full, else 0.
    """
    rows = []

    # HotpotQA main
    hot_main = json.loads((RESULTS / "gpt5.4_hotpotqa_main.json").read_text())
    for q in hot_main["questions"]:
        rows.append(("hotpotqa", q["qid"], int(q["flipped"]), None))
    # HotpotQA control: single per-question flag
    hot_ctrl = json.loads((RESULTS / "gpt5.4_hotpotqa_control.json").read_text())
    # control is a single aggregated number; we rely on per-question results in the file
    ctrl_lookup = {r["qid"]: int(r.get("control_flipped", False)) for r in hot_ctrl["results"]}
    rows = []
    for q in hot_main["questions"]:
        rows.append(("hotpotqa", q["qid"], int(q["flipped"]),
                     ctrl_lookup.get(q["qid"], 0)))

    # 2Wiki: cond1 = full, cond2 = gold removed, cond3 = random removed
    wiki = json.loads((RESULTS / "gpt5.4_2wiki_main.json").read_text())
    for qid, q in wiki["results"].items():
        rows.append((
            "2wiki",
            int(qid),
            int(q["cond2"] != q["cond1"]),
            int(q["cond3"] != q["cond1"]),
        ))

    return rows


def bootstrap_cfscore(rows, n_boot=10000, seed=0):
    """Return list of bootstrap CFScore estimates."""
    rng = random.Random(seed)
    cf = []
    n = len(rows)
    for _ in range(n_boot):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        t = sum(r[2] for r in sample)
        c = sum(r[3] for r in sample)
        cf.append((t - c) / n)
    cf.sort()
    return cf


def percentile_ci(sorted_x, alpha=0.05):
    n = len(sorted_x)
    lo = int(alpha / 2 * n)
    hi = int((1 - alpha / 2) * n)
    return sorted_x[lo], sorted_x[hi]


def loo_influence(rows):
    """Leave-one-out CFScore vs full CFScore."""
    base_t = sum(r[2] for r in rows) / len(rows)
    base_c = sum(r[3] for r in rows) / len(rows)
    base_cf = base_t - base_c
    influences = []
    for i in range(len(rows)):
        sub = rows[:i] + rows[i + 1:]
        t = sum(r[2] for r in sub) / len(sub)
        c = sum(r[3] for r in sub) / len(sub)
        influences.append((rows[i], base_cf - (t - c)))
    influences.sort(key=lambda x: -abs(x[1]))
    return base_cf, influences


def convergence_curve(rows, seed=0):
    """At each N in [10,20,30,40,50,57], 1000 random subsamples, mean±SD of CFScore."""
    rng = random.Random(seed)
    Ns = [10, 20, 30, 40, 50, 57]
    out = {}
    for N in Ns:
        if N > len(rows):
            continue
        vals = []
        for _ in range(1000):
            sample = [rows[rng.randrange(len(rows))] for _ in range(N)]
            t = sum(r[2] for r in sample) / N
            c = sum(r[3] for r in sample) / N
            vals.append(t - c)
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
        out[N] = {"mean": m, "sd": math.sqrt(var), "n_subsamples": len(vals)}
    return out


def main():
    rows = load_flip_states()
    print(f"Loaded {len(rows)} flip-state rows")
    print(f"  HotpotQA: {sum(1 for r in rows if r[0]=='hotpotqa')}")
    print(f"  2Wiki:    {sum(1 for r in rows if r[0]=='2wiki')}")

    # Pooled CFScore and CI
    n = len(rows)
    n_t = sum(r[2] for r in rows)
    n_c = sum(r[3] for r in rows)
    cf_pool = (n_t - n_c) / n
    t_lo, t_hi = wilson_95ci(n_t, n)
    c_lo, c_hi = wilson_95ci(n_c, n)
    print()
    print("=" * 72)
    print("Pooled CFScore on the existing N=57 sample")
    print("=" * 72)
    print(f"  targeted flip: {n_t}/{n} = {n_t/n:.1%}  Wilson 95% CI [{t_lo:.1%}, {t_hi:.1%}]")
    print(f"  random   flip: {n_c}/{n} = {n_c/n:.1%}  Wilson 95% CI [{c_lo:.1%}, {c_hi:.1%}]")
    print(f"  CFScore  = {cf_pool:.3f}")
    print(f"  CIs non-overlapping: {t_lo > c_hi}")

    # Bootstrap
    print()
    print("=" * 72)
    print("Bootstrap (10,000 resamples of N=57)")
    print("=" * 72)
    boot = bootstrap_cfscore(rows, n_boot=10000, seed=42)
    bmean = sum(boot) / len(boot)
    lo90, hi90 = percentile_ci(boot, 0.10)
    lo95, hi95 = percentile_ci(boot, 0.05)
    lo99, hi99 = percentile_ci(boot, 0.01)
    print(f"  Mean CFScore = {bmean:.3f}")
    print(f"  90% bootstrap percentile CI: [{lo90:.3f}, {hi90:.3f}]")
    print(f"  95% bootstrap percentile CI: [{lo95:.3f}, {hi95:.3f}]")
    print(f"  99% bootstrap percentile CI: [{lo99:.3f}, {hi99:.3f}]")

    # LOO influence
    print()
    print("=" * 72)
    print("Leave-one-out influence on CFScore")
    print("=" * 72)
    base_cf, influences = loo_influence(rows)
    print(f"  Full-sample CFScore = {base_cf:.3f}")
    print("  Top-5 most-influential questions (drop = change in CFScore):")
    for (ds, qid, _, _), drop in influences[:5]:
        print(f"    {ds} q{qid:>3}: drop = {drop:+.4f}")
    print("  Bottom-5 least-influential:")
    for (ds, qid, _, _), drop in influences[-5:]:
        print(f"    {ds} q{qid:>3}: drop = {drop:+.4f}")

    # Convergence
    print()
    print("=" * 72)
    print("CFScore convergence with sample size N (1000 random subsamples each)")
    print("=" * 72)
    conv = convergence_curve(rows, seed=42)
    for N in sorted(conv):
        s = conv[N]
        print(f"  N={N:>3}: mean CFScore = {s['mean']:.3f}  SD = {s['sd']:.4f}")

    # Save JSON
    summary = {
        "pooled": {
            "n": n,
            "targeted_flip": n_t,
            "random_flip": n_c,
            "cfscore": cf_pool,
            "wilson_95_targeted": [t_lo, t_hi],
            "wilson_95_random": [c_lo, c_hi],
            "cis_non_overlapping": bool(t_lo > c_hi),
        },
        "bootstrap": {
            "n_resamples": len(boot),
            "mean": bmean,
            "percentile_90": [lo90, hi90],
            "percentile_95": [lo95, hi95],
            "percentile_99": [lo99, hi99],
        },
        "loo_top5_influential": [
            {"dataset": r[0], "qid": r[1], "cfscore_drop_if_removed": drop}
            for r, drop in influences[:5]
        ],
        "convergence": {str(N): conv[N] for N in sorted(conv)},
    }
    (RESULTS / "bootstrap_summary.json").write_text(json.dumps(summary, indent=2))
    print()
    print(f"Wrote results/bootstrap_summary.json")

    # Convergence plot
    Ns = sorted(conv)
    means = [conv[N]["mean"] for N in Ns]
    sds = [conv[N]["sd"] for N in Ns]
    plt.figure(figsize=(5, 3.2))
    plt.errorbar(Ns, means, yerr=sds, marker="o", capsize=3, color="#1f4e79")
    plt.axhline(cf_pool, color="#888", linestyle="--", linewidth=0.8, label="full-sample CFScore")
    plt.xlabel("Subsample size N")
    plt.ylabel("CFScore (mean ± SD)")
    plt.title("CFScore convergence with sample size (gpt-5.4, HotpotQA+2Wiki)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_pdf = FIGS / "bootstrap_convergence.pdf"
    out_png = FIGS / "bootstrap_convergence.png"
    try:
        plt.savefig(out_pdf)
    except Exception as e:
        print(f"PDF save failed ({e}); saving PNG instead.")
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Wrote {out_png} and (attempted) {out_pdf}")


if __name__ == "__main__":
    main()