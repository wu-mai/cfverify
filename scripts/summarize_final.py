"""
summarize_final.py — After the larger_sample run finishes, merge all results
into the final pooled sample numbers and write the corresponding table-row
additions ready for the paper.

Reads:
    results/gpt5.4_hotpotqa_main.json
    results/gpt5.4_2wiki_main.json
    results/gpt5.4_hotpotqa_control.json
    results/larger_sample_gpt54_n60.json
    results/larger_sample_gpt54_n80.json   (when present)
    results/multimodel_gpt-5.4-mini.json

Prints a final summary suitable for drop-in into §4.11 / §5.
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


def random_flips(r, K):
    return sum(s["flipped"] for s in r["per_K"][str(K)])


def cf(records):
    """Compute F_T, F_R, CFScore on answered subset."""
    ans = [r for r in records if not r["abstained"]]
    if not ans:
        return None
    K = list(records[0]["per_K"].keys())[0] if "per_K" in records[0] else None
    if K is None:
        # multimodel format
        ft = sum(r["targeted_flipped"] for r in ans) / len(ans)
        fr = sum(r["random_flipped"] for r in ans) / len(ans)
    else:
        ft = sum(r["targeted_flipped"] for r in ans) / len(ans)
        fr = sum(random_flips(r, int(K)) for r in ans) / (len(ans) * int(K))
    return ft, fr, ft - fr, len(ans), len(records)


def main():
    # Headline
    hot = json.loads((RES / "gpt5.4_hotpotqa_main.json").read_text())
    wiki = json.loads((RES / "gpt5.4_2wiki_main.json").read_text())
    ctrl = json.loads((RES / "gpt5.4_hotpotqa_control.json").read_text())
    n_flip_h = sum(1 for q in hot["questions"] if q["flipped"])
    n_flip_w = sum(1 for q in wiki["results"].values() if q["cond2"] != q["cond1"])
    n_flip_r_h = ctrl["control_flipped_count"]
    n_flip_r_w = sum(1 for q in wiki["results"].values() if q["cond3"] != q["cond1"])
    n_h = len(hot["questions"])
    n_w = len(wiki["results"])
    n_total = n_h + n_w
    F_T_h = n_flip_h/n_h; F_R_h = n_flip_r_h/n_h
    F_T_w = n_flip_w/n_w; F_R_w = n_flip_r_w/n_w
    F_T = (n_flip_h+n_flip_w)/n_total
    F_R = (n_flip_r_h+n_flip_r_w)/n_total
    cf54 = F_T - F_R

    # Larger sample (60 Q, qid 31-60)
    ls60 = json.loads((RES / "larger_sample_gpt54_n60.json").read_text())
    ft_60, fr_60, c_60, a_60, t_60 = cf(ls60["records"])

    # Larger sample (80 Q, qid 31-100) — if present
    ls80_path = RES / "larger_sample_gpt54_n80.json"
    ls80 = json.loads(ls80_path.read_text()) if ls80_path.exists() else None
    if ls80:
        ft_80, fr_80, c_80, a_80, t_80 = cf(ls80["records"])
    else:
        ft_80 = fr_80 = c_80 = a_80 = t_80 = None

    # Multi-model
    mm = json.loads((RES / "multimodel_gpt-5.4-mini.json").read_text())
    ft_mm, fr_mm, c_mm, a_mm, t_mm = cf(mm["records"])

    # Pooled (headline + both extension runs)
    pooled_n = n_total + t_60 + (t_80 if t_80 else 0)
    pooled_t = n_flip_h + n_flip_w + sum(r["targeted_flipped"] for r in ls60["records"] if not r["abstained"])
    if ls80:
        pooled_t += sum(r["targeted_flipped"] for r in ls80["records"] if not r["abstained"])
    pooled_r = n_flip_r_h + n_flip_r_w + sum(random_flips(r, 1) for r in ls60["records"] if not r["abstained"])
    if ls80:
        pooled_r += sum(random_flips(r, 1) for r in ls80["records"] if not r["abstained"])
    pooled_ans = a_60 + n_total + (a_80 if a_80 else 0)
    # F_T / F_R / CFScore are over answered subset
    F_T_pool = pooled_t / pooled_ans
    F_R_pool = pooled_r / pooled_ans
    cf_pool = F_T_pool - F_R_pool
    lo, hi = wilson_95(pooled_t, pooled_ans)

    print("=" * 100)
    print("FINAL MULTI-MODEL + LARGER-SAMPLE SUMMARY (gpt-5.4 family)")
    print("=" * 100)
    print(f"\n--- GPT-5.4 headline (N={n_total}) ---")
    print(f"  HotpotQA: F_T={F_T_h:.1%}, F_R={F_R_h:.1%}, CFScore={F_T_h-F_R_h:+.3f}")
    print(f"  2Wiki:    F_T={F_T_w:.1%}, F_R={F_R_w:.1%}, CFScore={F_T_w-F_R_w:+.3f}")
    print(f"  POOLED:   F_T={F_T:.1%}, F_R={F_R:.1%}, CFScore={cf54:+.3f}")

    print(f"\n--- GPT-5.4 extension batch 1 (qid 31-60, N=60) ---")
    print(f"  Answered: {a_60}/{t_60}")
    print(f"  F_T={ft_60:.1%}, F_R={fr_60:.1%}, CFScore={c_60:+.3f}")

    if ls80:
        print(f"\n--- GPT-5.4 extension batch 2 (qid 31-100, N=80) ---")
        print(f"  Answered: {a_80}/{t_80}")
        print(f"  F_T={ft_80:.1%}, F_R={fr_80:.1%}, CFScore={c_80:+.3f}")

    print(f"\n--- GPT-5.4-mini (N={t_mm}) ---")
    print(f"  Answered: {a_mm}/{t_mm}")
    print(f"  F_T={ft_mm:.1%}, F_R={fr_mm:.1%}, CFScore={c_mm:+.3f}")

    print(f"\n=== GPT-5.4 POOLED (headline + extension) ===")
    print(f"  N_answered = {pooled_ans} (out of {pooled_n} attempted)")
    print(f"  F_T={F_T_pool:.1%}, F_R={F_R_pool:.1%}, CFScore={cf_pool:+.3f}")
    print(f"  Wilson 95% CI on F_T: [{lo:.1%}, {hi:.1%}]")

    # Save JSON
    out = {
        "gpt54_headline": {"N": n_total, "F_T": F_T, "F_R": F_R, "CFScore": cf54},
        "gpt54_extension_60": {"N_answered": a_60, "N_total": t_60,
                               "F_T": ft_60, "F_R": fr_60, "CFScore": c_60},
        "gpt54_mini": {"N_answered": a_mm, "N_total": t_mm,
                       "F_T": ft_mm, "F_R": fr_mm, "CFScore": c_mm},
        "gpt54_pooled": {"N_answered": pooled_ans, "N_total": pooled_n,
                          "F_T": F_T_pool, "F_R": F_R_pool, "CFScore": cf_pool,
                          "wilson_95_lo": lo, "wilson_95_hi": hi},
    }
    if ls80:
        out["gpt54_extension_80"] = {"N_answered": a_80, "N_total": t_80,
                                     "F_T": ft_80, "F_R": fr_80, "CFScore": c_80}
    out_path = RES / "final_summary.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()