"""
run_gpt54_merged.py — Merged-prompt variant on GPT-5.4 via codex proxy.

Mirrors scripts/run_mistral_merged.py but uses OpenAI responses API instead of
local Mistral, and the gpt-5.4 /codex proxy endpoint.

For each of 60 questions:
  cond1_merged: (answer, ghat) in one call
  cond2_gold_removed: targeted deletion using ghat as G
  K × cond3: random-set deletion baseline

Outputs:
    results/gpt54_merged.json
"""
import json
import os
import random
import re
import sys
import time
from pathlib import Path

from openai import OpenAI

HERE = Path(__file__).parent
ROOT = HERE.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

SYSTEM_PROMPT = (
    "You are a question-answering assistant. Answer the question based ONLY "
    "on the evidence provided. Give a concise answer (a few words). If the "
    "evidence does not support an answer, say exactly 'insufficient evidence'. "
    "Do not use any other knowledge."
)


def get_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Set OPENAI_API_KEY first.")
    base_url = os.environ.get("OPENAI_BASE_URL")
    return OpenAI(api_key=api_key, base_url=base_url)


def ask_merged(client, model, question, evidence_sents, max_output=200, max_retries=10):
    ev_text = "\n".join(f"[{i}] {s}" for i, s in enumerate(evidence_sents))
    prompt = (
        f"Evidence:\n{ev_text}\n\nQuestion: {question}\n\n"
        "First, answer the question based ONLY on the evidence above. "
        "Give a concise answer (a few words), or say \"insufficient evidence\" if "
        "the evidence does not support an answer. Do not use outside knowledge.\n\n"
        "Then, on a new line, output a JSON object with the field "
        "\"essential_indices\" listing the integer indices of the evidence "
        "sentences you actually used to produce the answer. Only include indices "
        "that were strictly necessary.\n\nAnswer:"
    )
    for attempt in range(max_retries):
        try:
            r = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_output_tokens=max_output,
                timeout=120.0,  # long timeout to survive flaky proxy
            )
            break
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = min(30 * (attempt + 1), 300)
            print(f"    [retry {attempt+1}/{max_retries}] {str(e)[:80]} — sleeping {wait}s",
                  flush=True)
            time.sleep(wait)
    raw = r.output_text.strip()

    # Parse
    ghat = []
    # 1. JSON object
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        try:
            obj = json.loads(m.group(0))
            for k, v in obj.items():
                if "index" in k.lower() and isinstance(v, list):
                    ghat = [int(x) for x in v if isinstance(x, (int, float))]
                    break
        except Exception:
            pass
    # 2. Prose-style: "essential_indices: [...]"
    if not ghat:
        m = re.search(r"[Ii]ndices[:\s=]+\[([^\]]*)\]", raw)
        if m:
            try:
                ghat = sorted({int(x.strip()) for x in m.group(1).split(",")
                               if x.strip().lstrip("-").isdigit()})
            except Exception:
                pass
    # 3. Bare list
    if not ghat:
        mid = len(raw) // 2
        m = re.search(r"\[([0-9][0-9,\s]*)\]", raw[mid:])
        if m:
            try:
                ghat = sorted({int(x.strip()) for x in m.group(1).split(",") if x.strip()})
            except Exception:
                pass
    ans = raw.split("{", 1)[0].strip()
    ans = re.sub(r"^(answer\s*:?\s*)", "", ans, flags=re.IGNORECASE).strip()
    ans = ans.split("\n")[0].strip().strip('"').strip(".")
    return raw, ans, ghat


def ask_simple(client, model, question, evidence_sents, max_output=60, max_retries=10):
    ev_text = "\n".join(f"[{i}] {s}" for i, s in enumerate(evidence_sents))
    user = f"Question: {question}\n\nEvidence:\n{ev_text}\n\nAnswer:"
    for attempt in range(max_retries):
        try:
            r = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                max_output_tokens=max_output,
                timeout=120.0,
            )
            break
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = min(30 * (attempt + 1), 300)
            print(f"    [retry {attempt+1}/{max_retries}] {str(e)[:80]} — sleeping {wait}s",
                  flush=True)
            time.sleep(wait)
    ans = r.output_text.strip().split("\n")[0].strip().strip('"').strip(".").lower()
    return ans


def load_questions():
    hot = json.loads((DATA / "hotpotqa_30_questions.json").read_text())
    wiki = json.loads((DATA / "2wiki_30_questions.json").read_text())
    return [("hotpotqa", q) for q in hot] + [("2wiki", q) for q in wiki]


def f1(ghat, gold):
    if not ghat and not gold:
        return 1.0
    s1, s2 = set(ghat), set(gold)
    if not s1 or not s2:
        return 0.0
    p = len(s1 & s2) / len(s1)
    r = len(s1 & s2) / len(s2)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def main():
    client = get_client()
    model = "gpt-5.4"
    K = 3
    questions = load_questions()
    print(f"\n=== GPT-5.4 merged-prompt variant, K={K}, N={len(questions)} ===\n")
    records = []
    t_global = time.time()
    f1s = []
    n_answered = 0
    n_flip_targeted = 0
    n_flip_random = 0
    n_pred_nonempty = 0

    for qi, (ds, q) in enumerate(questions, 1):
        all_sents = []
        for d in q["documents"]:
            for s in d["sentences"]:
                all_sents.append(s)
        gold = set(q["gold_sentence_indices"])
        non_gold = [i for i in range(len(all_sents)) if i not in gold]
        n_remove = min(len(gold), len(non_gold))
        full_ev = [(i, s) for i, s in enumerate(all_sents)]

        t = time.time()
        raw, ans_merged, ghat = ask_merged(client, model, q["question"], full_ev)
        ghat_valid = sorted({x for x in ghat if 0 <= x < len(all_sents)})
        target_remove = set(ghat_valid) if ghat_valid else gold
        ghat_removed_ev = [
            (i, s) for i, s in enumerate(all_sents) if i not in target_remove
        ]
        a_target = ask_simple(client, model, q["question"], ghat_removed_ev)

        random_samples = []
        for k in range(K):
            seed = q["question_id"] * 1009 + 31 * (k + 1) + 7
            local_rng = random.Random(seed)
            rr = set(local_rng.sample(non_gold, n_remove))
            rr_ev = [(i, s) for i, s in enumerate(all_sents) if i not in rr]
            a3 = ask_simple(client, model, q["question"], rr_ev)
            random_samples.append({
                "k_idx": k + 1, "seed": seed,
                "random_removed": sorted(rr),
                "answer": a3,
                "flipped": (a3.strip().lower() != ans_merged.strip().lower()),
            })

        f1_score = f1(ghat_valid, sorted(gold))
        f1s.append(f1_score)
        if ghat_valid:
            n_pred_nonempty += 1

        abst = ans_merged.lower().startswith("insufficient") or not ans_merged.strip()
        if not abst:
            n_answered += 1
            if a_target.strip().lower() != ans_merged.strip().lower():
                n_flip_targeted += 1
            for s in random_samples:
                if s["flipped"]:
                    n_flip_random += 1

        records.append({
            "dataset": ds,
            "qid": q["question_id"],
            "question": q["question"],
            "gold_answer": q["answer"],
            "gold_indices": sorted(gold),
            "raw_merged_output": raw,
            "cond1_merged_answer": ans_merged,
            "cond1_ghat": ghat_valid,
            "f1_vs_gold": f1_score,
            "target_removed_indices": sorted(target_remove),
            "cond2_target_removed_answer": a_target,
            "targeted_flipped": (a_target.strip().lower() != ans_merged.strip().lower()),
            "per_K_random": random_samples,
        })
        elapsed = time.time() - t
        if qi % 10 == 0 or qi == len(questions):
            print(
                f"  [{qi:>2}/{len(questions)}] {ds:>8} q{q['question_id']:>3} "
                f"[{elapsed:>4.1f}s] ghat={len(ghat_valid)} f1={f1_score:.2f} "
                f"ans='{ans_merged[:30]}'"
            )

    summary = {
        "model": model,
        "K": K,
        "n_questions": len(questions),
        "answered_count": n_answered,
        "f1_ghat_vs_gold_mean": sum(f1s) / len(f1s) if f1s else 0,
        "f1_ghat_vs_gold_median": sorted(f1s)[len(f1s) // 2] if f1s else 0,
        "answered_targeted_flip": n_flip_targeted,
        "answered_random_flip": n_flip_random,
        "answered_random_total": n_answered * K,
        "pred_nonempty": n_pred_nonempty,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_seconds": time.time() - t_global,
        "records": records,
    }
    # Write checkpoint after every question
    out_path = RESULTS / "gpt54_merged.json"
    out_path.write_text(json.dumps(summary, indent=2))
    out_path = RESULTS / "gpt54_merged.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}  (total {time.time()-t_global:.0f}s)")
    print(f"\n=== Summary ===")
    print(f"Answered (full): {n_answered}/{len(questions)}")
    print(f"Predictions non-empty (loose parser): {n_pred_nonempty}/{len(questions)}")
    print(f"Ĝ F1 vs gold: mean={summary['f1_ghat_vs_gold_mean']:.3f}")
    print(f"Targeted flip (answered): {n_flip_targeted}/{n_answered}")
    print(f"Random   flip (answered): {n_flip_random}/{n_answered * K}")
    if n_answered:
        ft = n_flip_targeted / n_answered
        fr = n_flip_random / (n_answered * K)
        print(f"CFScore: {ft - fr:+.3f}")


if __name__ == "__main__":
    main()