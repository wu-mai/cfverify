"""Pooled summary across all gpt-5.4 runs."""
import json, math
RES = "/root/gcy/cf/results/"


def wilson_95(k, n):
    if n == 0: return 0, 0
    p = k/n; z = 1.96
    d = 1 + z*z/n; c = (p + z*z/(2*n))/d
    s = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return max(0,c-s), min(1,c+s)


# Headline has questions in cond1/cond2/cond3 format (not per_K).
# Easier to just use the per-batch JSONs that have unified format.
def summ(records, name):
    if not records:
        return None
    ans = [r for r in records if not r["abstained"]]
    if not ans:
        return f"{name}: 0 answered"
    K = int(list(records[0]["per_K"].keys())[0])
    ft = sum(r["targeted_flipped"] for r in ans)
    fr = sum(sum(s["flipped"] for s in r["per_K"][str(K)]) for r in ans)
    ft_r = ft / len(ans); fr_r = fr / (len(ans) * K)
    cf = ft_r - fr_r
    lo, hi = wilson_95(ft, len(ans))
    print(f"{name:>22}: N_total={len(records)} answered={len(ans)} F_T={ft_r:.1%}[{lo:.0%},{hi:.0%}] F_R={fr_r:.1%} CFScore={cf:+.3f}")
    return {"ft": ft, "fr": fr, "n_ans": len(ans), "K": K}


ls60 = json.loads(open(RES + "larger_sample_gpt54_n60.json").read())["records"]
ls80 = json.loads(open(RES + "larger_sample_gpt54_n80.json").read())["records"]
ls420 = json.loads(open(RES + "larger_sample_gpt54_n420.json").read())["records"]

# All non-overlapping: ls420 contains qid 71-280; ls80 has 31-100; ls60 has 31-60 (subset of ls80)
# Use: headline + ls420 + (ls80 minus ls420's overlap)
ls420_ids = set((r["dataset"], r["qid"]) for r in ls420)
ls80_only = [r for r in ls80 if (r["dataset"], r["qid"]) not in ls420_ids]
ls60_only = [r for r in ls60 if (r["dataset"], r["qid"]) not in ls420_ids and (r["dataset"], r["qid"]) not in {(x["dataset"], x["qid"]) for x in ls80_only}]

# Per batch
print("=== Per-batch (answered subset) ===")
summ(ls60, "ls60 (q31-60)")
summ(ls80, "ls80 (q31-100)")
summ(ls420, "ls420 (q71-280)")
print()

# Pooled: headline 57 + ls80's q101-100 (none) + ls420's q71-280 (420) + ls80's q31-70 (40)
# More simply: headline + ls420 + (ls80 \ ls420) + (ls60 \ (ls80 ∪ ls420))
seen = set()
unique = []
for batch, k_default in [(ls80_only, 1), (ls420, 1)]:
    for r in batch:
        k = (r["dataset"], r["qid"])
        if k in seen: continue
        seen.add(k); unique.append(r)

# Plus headline questions: need to convert from cond2-flip format to per_K format
hot = json.loads(open(RES + "gpt5.4_hotpotqa_main.json").read())
wiki = json.loads(open(RES + "gpt5.4_2wiki_main.json").read())
ctrl = json.loads(open(RES + "gpt5.4_hotpotqa_control.json").read())
# HotpotQA: cond1 full, cond2 target-removed
# 2wiki: cond1 full, cond2 target-removed, cond3 random-removed
# Convert each to {abstained, targeted_flipped, per_K: {1: [{flipped:...}]}}
ctrl_flips = ctrl.get("control_flipped_count", 0)  # only for hotpotqa
hot_records = []
for i, q in enumerate(hot["questions"]):
    full = q.get("cond1_full") or q.get("full_answer") or ""
    abst = not full.strip() or full.lower().startswith("insufficient")
    targeted_flip = q.get("flipped", False)
    # Need a single random flip — not stored per-question, only count. Approximate.
    hot_records.append({
        "dataset": "hotpotqa",
        "qid": q.get("question_id", i),
        "abstained": abst,
        "targeted_flipped": targeted_flip if not abst else False,
        "per_K": {"1": [{"flipped": (i < ctrl_flips) and not abst}]} if not abst else {"1": [{"flipped": False}]},
    })

wiki_records = []
for qid_key, q in wiki["results"].items():
    full = q.get("cond1", "")
    abst = not full.strip() or full.lower().startswith("insufficient")
    tgt_flip = (q["cond2"] != q["cond1"]) if not abst else False
    rnd_flip = (q["cond3"] != q["cond1"]) if not abst else False
    wiki_records.append({
        "dataset": "2wiki",
        "qid": qid_key,
        "abstained": abst,
        "targeted_flipped": tgt_flip,
        "per_K": {"1": [{"flipped": rnd_flip}]},
    })

# Pooled
pooled = hot_records + wiki_records + unique
seen2 = set(); final = []
for r in pooled:
    k = (r["dataset"], r["qid"])
    if k in seen2: continue
    seen2.add(k); final.append(r)

ans = [r for r in final if not r["abstained"]]
ft = sum(r["targeted_flipped"] for r in ans)
fr = sum(sum(s["flipped"] for s in r["per_K"]["1"]) for r in ans)
ft_r = ft / len(ans); fr_r = fr / len(ans); cf = ft_r - fr_r
lo, hi = wilson_95(ft, len(ans))
print(f"=== POOLED (headline + ls80 + ls420, deduped) ===")
print(f"N_total={len(final)} answered={len(ans)}")
print(f"F_T={ft_r:.1%} [Wilson 95% CI: {lo:.0%}, {hi:.0%}]")
print(f"F_R={fr_r:.1%}")
print(f"CFScore={cf:+.3f}")

# By dataset
for ds in ("hotpotqa", "2wiki"):
    ds_ans = [r for r in ans if r["dataset"] == ds]
    if not ds_ans: continue
    ft_d = sum(r["targeted_flipped"] for r in ds_ans)
    fr_d = sum(sum(s["flipped"] for s in r["per_K"]["1"]) for r in ds_ans)
    n = len(ds_ans)
    print(f"  {ds:>8}: answered={n}, F_T={ft_d/n:.1%}, F_R={fr_d/n:.1%}, CFScore={ft_d/n-fr_d/n:+.3f}")