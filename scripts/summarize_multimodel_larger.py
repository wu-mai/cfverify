"""
summarize_multimodel_larger.py — Print the multi-model + larger-sample summary
that goes into §5 / §4.11 of the paper.

Reads:
    results/multimodel_*.json
    results/larger_sample_*.json
    results/gpt5.4_hotpotqa_main.json
    results/gpt5.4_2wiki_main.json
    results/gpt5.4_hotpotqa_control.json

Prints tables + writes:
    results/multimodel_summary.json
"""
import json
import math
from pathlib import Path

HERE = Path("/root/gcy/cf")
RES = HERE / "results"


def wilson_95(k, n):
    if n == 0: return 0, 0
    p = k / n
    z = 1.96
    d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    s = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return max(0, c-s), min(1, c+s)


def main():
    # Existing GPT-5.4 headline
    hot = json.loads((RES / "gpt5.4_hotpotqa_main.json").read_text())
    wiki = json.loads((RES / "gpt5.4_2wiki_main.json").read_text())
    ctrl = json.loads((RES / "gpt5.4_hotpotqa_control.json").read_text())
    n_t_h = len(hot["questions"])
    n_t_w = len(wiki["results"])
    n_flip_h = sum(1 for q in hot["questions"] if q["flipped"])
    n_flip_w = sum(1 for q in wiki["results"].values() if q["cond2"] != q["cond1"])
    n_flip_r_h = ctrl["control_flipped_count"]
    n_flip_r_w = sum(1 for q in wiki["results"].values() if q["cond3"] != q["cond1"])
    n_total = n_t_h + n_t_w
    n_t = n_flip_h + n_flip_w
    n_r = n_flip_r_h + n_flip_r_w
    cf54 = n_t/n_total - n_r/n_total
    cf54_lo, cf54_hi = wilson_95(n_t, n_total)

    # GPT-5.4-mini multi-model
    mm_path = RES / "multimodel_gpt-5.4-mini.json"
    mm = json.loads(mm_path.read_text()) if mm_path.exists() else None

    # Larger sample
    ls_path = RES / "larger_sample_gpt54_n60.json"
    ls = json.loads(ls_path.read_text()) if ls_path.exists() else None

    print("=" * 100)
    print("Multi-model + larger-sample summary")
    print("=" * 100)

    print("\n--- GPT-5.4 (headline, N=57) ---")
    print(f"  HotpotQA: F_T={n_flip_h}/{n_t_h}={n_flip_h/n_t_h:.1%}, "
          f"F_R={n_flip_r_h}/{n_t_h}={n_flip_r_h/n_t_h:.1%}")
    print(f"  2Wiki:    F_T={n_flip_w}/{n_t_w}={n_flip_w/n_t_w:.1%}, "
          f"F_R={n_flip_r_w}/{n_t_w}={n_flip_r_w/n_t_w:.1%}")
    print(f"  POOLED N={n_total}, F_T={n_t/n_total:.1%}, F_R={n_r/n_total:.1%}, "
          f"CFScore={cf54:+.3f}")

    if mm:
        print("\n--- GPT-5.4-mini (N=60, K=1) ---")
        recs = mm["records"]
        ans = [r for r in recs if not r["abstained"]]
        ft_m = sum(r["targeted_flipped"] for r in ans) / len(ans)
        fr_m = sum(r["random_flipped"] for r in ans) / len(ans)
        cf_m = ft_m - fr_m
        print(f"  Answered: {len(ans)}/{len(recs)} ({len(recs)-len(ans)} abstained)")
        print(f"  F_T={ft_m:.1%}, F_R={fr_m:.1%}, CFScore={cf_m:+.3f}")
        # Per-dataset
        for ds in ("hotpotqa", "2wiki"):
            sub = [r for r in ans if r["dataset"] == ds]
            ft = sum(r["targeted_flipped"] for r in sub) / len(sub)
            fr = sum(r["random_flipped"] for r in sub) / len(sub)
            print(f"    {ds}: N={len(sub)}, F_T={ft:.1%}, F_R={fr:.1%}, CFScore={ft-fr:+.3f}")

    if ls:
        print("\n--- Larger sample GPT-5.4 (N=60 NEW) ---")
        recs = ls["records"]
        ans = [r for r in recs if not r["abstained"]]
        # Random flips are inside per_K → sum over all K samples
        def random_flips(r):
            return sum(s["flipped"] for s in r["per_K"][str(ls["K"])])
        ft_l = sum(r["targeted_flipped"] for r in ans) / len(ans)
        fr_l = sum(random_flips(r) for r in ans) / (len(ans) * ls["K"])
        cf_l = ft_l - fr_l
        print(f"  Answered: {len(ans)}/{len(recs)}")
        print(f"  F_T={ft_l:.1%}, F_R={fr_l:.1%}, CFScore={cf_l:+.3f}")
        for ds in ("hotpotqa", "2wiki"):
            sub = [r for r in ans if r["dataset"] == ds]
            if not sub: continue
            ft = sum(r["targeted_flipped"] for r in sub) / len(sub)
            fr = sum(random_flips(r) for r in sub) / (len(sub) * ls["K"])
            print(f"    {ds}: N={len(sub)}, F_T={ft:.1%}, F_R={fr:.1%}, CFScore={ft-fr:+.3f}")

        # Combined with headline
        all_n = n_total + len(ans)
        all_t = n_t + sum(r["targeted_flipped"] for r in ans)
        all_r = n_r + sum(random_flips(r) for r in ans)
        cf_pooled = all_t/all_n - all_r/(all_n * ls["K"])
        print(f"\n  COMBINED (headline 57 + new 60), N_answered={all_n}:")
        print(f"  F_T={all_t/all_n:.1%}, F_R={all_r/(all_n*ls['K']):.1%}, CFScore={cf_pooled:+.3f}")

    summary = {
        "gpt54_headline": {
            "N_total": n_total, "F_T_rate": n_t/n_total, "F_R_rate": n_r/n_total,
            "CFScore": cf54, "wilson_95_lo": cf54_lo, "wilson_95_hi": cf54_hi,
        },
    }
    if mm:
        ans = [r for r in mm["records"] if not r["abstained"]]
        ft_m = sum(r["targeted_flipped"] for r in ans) / len(ans)
        fr_m = sum(r["random_flipped"] for r in ans) / len(ans)
        summary["gpt54_mini"] = {
            "model": "gpt-5.4-mini", "N_total": len(mm["records"]),
            "N_answered": len(ans),
            "F_T_rate": ft_m, "F_R_rate": fr_m, "CFScore": ft_m - fr_m,
        }
    if ls:
        ans = [r for r in ls["records"] if not r["abstained"]]
        K = ls["K"]
        def random_flips(r):
            return sum(s["flipped"] for s in r["per_K"][str(K)])
        ft_l = sum(r["targeted_flipped"] for r in ans) / len(ans)
        fr_l = sum(random_flips(r) for r in ans) / (len(ans) * K)
        summary["gpt54_larger_sample"] = {
            "model": "gpt-5.4", "N_total": len(ls["records"]),
            "N_answered": len(ans),
            "F_T_rate": ft_l, "F_R_rate": fr_l, "CFScore": ft_l - fr_l,
        }
        # Pooled (note: random_flips in headline is count over N=57 / K=1; new is K=1 too)
        all_n = n_total + len(ans)
        all_t = n_t + sum(r["targeted_flipped"] for r in ans)
        all_r = n_r + sum(random_flips(r) for r in ans)
        summary["gpt54_pooled"] = {
            "N_total": all_n, "N_answered": all_n,
            "F_T_rate": all_t/all_n, "F_R_rate": all_r/(all_n*K),
            "CFScore": all_t/all_n - all_r/(all_n*K),
        }
    out = RES / "multimodel_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()