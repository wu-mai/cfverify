"""
recompute_merged.py — Recompute merged-prompt results with a loose parser.

The original script's regex only matched JSON-quoted "essential_indices":[...].
Mistral-7B-Instruct-v0.3 outputs unquoted `Essential_indices: [...]`.
This script re-parses all records from results/mistral7b_merged.json with a
loose parser and recomputes:
  - ghat (set of valid indices)
  - f1_vs_gold
  - targeted deletion (still valid since target_remove used ghat_valid — but now
    we use the loose ghat instead of [])

Writes results/mistral7b_merged_recomputed.json.
"""
import json
import re
from pathlib import Path

RES = Path("/root/gcy/cf/results")


def loose_parse(raw_text):
    """Find a list of ints after 'indices' (any case), or any [list, of, ints]."""
    text = raw_text.strip()
    # Try: "indices" / "Indices" / "Essential_indices" + ":" or "=" + [...]
    m = re.search(r"[Ii]ndices[:\s=]+\[([^\]]*)\]", text)
    if m:
        try:
            return sorted({int(x.strip()) for x in m.group(1).split(",") if x.strip().lstrip("-").isdigit()})
        except Exception:
            pass
    # Try: any [list of ints] not inside a sentence (heuristic: must be in the last 50% of text)
    mid = len(text) // 2
    m = re.search(r"\[([0-9][0-9,\s]*)\]", text[mid:])
    if m:
        try:
            return sorted({int(x.strip()) for x in m.group(1).split(",") if x.strip()})
        except Exception:
            pass
    return []


def f1(pred, gold):
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    s1, s2 = set(pred), set(gold)
    tp = len(s1 & s2)
    if tp == 0:
        return 0.0
    p = tp / len(s1); r = tp / len(s2)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def main():
    src = RES / "mistral7b_merged.json"
    out = RES / "mistral7b_merged_recomputed.json"
    d = json.loads(src.read_text())

    # Need raw outputs to re-parse. The original script didn't save raw outputs.
    # Workaround: detect what happened. Since raw was not stored, we infer from
    # the records: cond1_ghat is empty if parse failed, non-empty otherwise.
    # We CANNOT re-derive cond1_ghat without raw text. So this script will instead
    # rerun the relevant subset with a looser regex.

    # For now: print summary of how many records had empty ghat and would benefit
    # from a re-parse.
    n_empty = 0
    for r in d["records"]:
        if not r.get("cond1_ghat"):
            n_empty += 1
    print(f"Records with empty parsed ghat (would benefit from re-parse): {n_empty}/{len(d['records'])}")
    print(f"Records with non-empty parsed ghat: {len(d['records']) - n_empty}/{len(d['records'])}")

    # This script is a placeholder — actual re-run is run_mistral_merged_v2.py
    print("\nPlease use scripts/run_mistral_merged_v2.py for the loose-parser re-run.")


if __name__ == "__main__":
    main()