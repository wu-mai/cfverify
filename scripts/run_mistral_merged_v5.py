"""
run_mistral_merged_v5.py — Fix the protocol inconsistency in merged evaluation.

v1-v4 compared the MERGED call's answer (JSON-embedded, long-form) against
RANDOM-deletion answers from the SEPARATE simple prompt (short-form). String
comparing a long-form answer against short-form answers guarantees spurious
flips. v5 runs the random-deletion probes through the SAME merged prompt
(still asking for JSON), so the comparison is format-consistent.

This isolates: was the v1-v4 "calibration collapse" partly a measurement
artifact of cross-format string comparison?

Outputs:
    results/mistral7b_merged_v5.json
"""
import json
import random
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
ROOT = HERE.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

MODEL = "/root/autodl-tmp/gcy_cf"
K_DEFAULT = 3

FEWSHOT = (
    "Example 1:\n"
    "Evidence:\n[0] The Eiffel Tower is located in Paris, France.\n"
    "[1] The Eiffel Tower was completed in 1889.\n\n"
    "Question: In what year was the Eiffel Tower completed?\n"
    "Output: {\"answer\": \"1889\", \"supporting_indices\": [1]}\n\n"
    "Example 2:\n"
    "Evidence:\n[0] The Eiffel Tower is located in Paris, France.\n"
    "[1] The Eiffel Tower was completed in 1889.\n\n"
    "Question: What is the population of Lyon?\n"
    "Output: {\"answer\": null, \"supporting_indices\": []}\n\n"
)


def parse_json_answer(raw):
    text = raw.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            obj = json.loads(m.group(0))
            ans = obj.get("answer")
            if ans is None:
                ans = "insufficient evidence"
            ghat = [int(x) for x in obj.get("supporting_indices", [])
                    if isinstance(x, (int, float))]
            return str(ans), ghat
        except Exception:
            pass
    m2 = re.search(r"answer\s*[:=]\s*(null|.+?)(?:\n|$)", text, re.IGNORECASE)
    ans = m2.group(1).strip() if m2 else text.split("\n")[0].strip()
    ghat = []
    m3 = re.search(r"[Ii]ndices[:\s=]+\[([^\]]*)\]", text)
    if m3:
        try:
            ghat = sorted({int(x.strip()) for x in m3.group(1).split(",")
                           if x.strip().lstrip("-").isdigit()})
        except Exception:
            pass
    if ans.lower() in ("null", "none"):
        ans = "insufficient evidence"
    return ans, ghat


def ask_merged(question, evidence_sents, tok, model, max_new_tokens=200):
    ev_text = "\n".join(f"[{i}] {s}" for i, s in evidence_sents)
    prompt = (
        f"{FEWSHOT}"
        f"Now the actual task.\n\n"
        f"Evidence:\n{ev_text}\n\nQuestion: {question}\n\n"
        "If the evidence does not contain the answer, output exactly "
        "{\"answer\": null, \"supporting_indices\": []}. Otherwise output "
        "{\"answer\": \"<a few words>\", \"supporting_indices\": [<indices>]}. "
        "Keep the answer to a few words. Output only the JSON object."
    )
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                    tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    raw = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                     skip_special_tokens=True).strip()
    return raw, *parse_json_answer(raw)


def f1(ghat, gold):
    if not ghat and not gold: return 1.0
    s1, s2 = set(ghat), set(gold)
    if not s1 or not s2: return 0.0
    tp = len(s1 & s2)
    if tp == 0: return 0.0
    p = tp/len(s1); r = tp/len(s2)
    return 2*p*r/(p+r) if (p+r) else 0.0


def main():
    print(f"Loading {MODEL}...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                 device_map="auto")
    model.eval()
    print(f"Loaded {time.time()-t0:.1f}s")

    hot = json.loads((DATA / "hotpotqa_30_questions.json").read_text())
    wiki = json.loads((DATA / "2wiki_30_questions.json").read_text())
    questions = [("hotpotqa", q) for q in hot] + [("2wiki", q) for q in wiki]
    K = K_DEFAULT
    print(f"v5 (consistent merged protocol), K={K}, N={len(questions)}")

    records = []
    n_ans = n_ft = 0; n_fr = 0; f1s = []
    t0 = time.time()
    for qi, (ds, q) in enumerate(questions, 1):
        sents = []
        for d in q["documents"]:
            for s in d["sentences"]:
                sents.append(s)
        gold = set(q["gold_sentence_indices"])
        non_gold = [i for i in range(len(sents)) if i not in gold]
        n_remove = min(len(gold), len(non_gold))
        full_ev = [(i, s) for i, s in enumerate(sents)]

        # 1. Merged call on full evidence
        raw, ans, ghat = ask_merged(q["question"], full_ev, tok, model)
        ghat_valid = sorted({x for x in ghat if 0 <= x < len(sents)})
        target_remove = set(ghat_valid) if ghat_valid else gold

        # 2. Merged call on targeted deletion (SAME prompt format)
        tgt_ev = [(i, s) for i, s in enumerate(sents) if i not in target_remove]
        _, a_target, _ = ask_merged(q["question"], tgt_ev, tok, model)

        # 3. K merged calls on random deletions (SAME prompt format)
        rnd_flips = []
        for k in range(K):
            seed = q["question_id"] * 1009 + 31 * (k + 1) + 7
            rng = random.Random(seed)
            rr = set(rng.sample(non_gold, n_remove))
            rr_ev = [(i, s) for i, s in enumerate(sents) if i not in rr]
            _, a_rnd, _ = ask_merged(q["question"], rr_ev, tok, model)
            rnd_flips.append(a_rnd.strip().lower() != ans.strip().lower())

        f1s.append(f1(ghat_valid, sorted(gold)))
        abst = ans.lower().startswith("insufficient") or not ans.strip()
        if not abst:
            n_ans += 1
            if a_target.strip().lower() != ans.strip().lower(): n_ft += 1
            n_fr += sum(rnd_flips)
        records.append({"dataset": ds, "qid": q["question_id"],
                        "answer": ans, "ghat": ghat_valid, "f1": f1s[-1],
                        "target_answer": a_target,
                        "targeted_flipped": a_target.strip().lower() != ans.strip().lower(),
                        "rnd_flips": rnd_flips})
        if qi % 10 == 0 or qi == len(questions):
            print(f"  [{qi}/{len(questions)}] [{time.time()-t0:.0f}s] ans={n_ans} T={n_ft} R={n_fr}", flush=True)

    summary = {"model": "Mistral-7B-v5-consistent", "K": K, "n": len(records),
               "answered": n_ans,
               "f1_mean": sum(f1s)/len(f1s),
               "F_T": n_ft/n_ans if n_ans else 0,
               "F_R": n_fr/(n_ans*K) if n_ans else 0,
               "CFScore": (n_ft/n_ans - n_fr/(n_ans*K)) if n_ans else 0,
               "records": records}
    json.dump(summary, open(RESULTS / "mistral7b_merged_v5.json", "w"), indent=1)
    print(f"\nDONE {time.time()-t0:.0f}s: answered={n_ans}/60, F1={summary['f1_mean']:.3f}, "
          f"F_T={summary['F_T']:.1%}, F_R={summary['F_R']:.1%}, CFScore={summary['CFScore']:+.3f}")


if __name__ == "__main__":
    main()