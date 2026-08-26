"""
run_mistral_merged.py — Mistral-7B with merged self-rationale prompt.

The "merged" variant asks the model in a single call to (a) answer the question,
(b) output the JSON object listing the indices of evidence sentences it actually
used. This compresses CF-Verify's 3+K calls into 2+K.

For each of 60 questions:
  cond1_merged: (answer, ghat) in one call
  cond2_gold_removed: targeted deletion using ghat as G
  K × cond3: random-set deletion baseline

Outputs:
    results/mistral7b_merged.json
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
RESULTS.mkdir(exist_ok=True)

MODEL = "/root/autodl-tmp/gcy_cf"  # local path; weights moved to data disk
K_DEFAULT = 3  # baseline; K is not the focus of this experiment


def load_model():
    print("Loading Mistral-7B-Instruct-v0.3 (bf16)...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    print(f"Loaded in {time.time()-t0:.1f}s")
    return tok, model


def parse_answer_and_ghat(raw):
    """Parse merged output: 'answer\\n{...json...}' or 'answer {...json...}'.

    Returns (answer_str, ghat_list_of_int). Tolerates model noise.
    """
    text = raw.strip()
    # Try to find JSON object first
    m = re.search(r"\{[\s\S]*\}", text)
    ghat = []
    if m:
        try:
            obj = json.loads(m.group(0))
            ghat = obj.get("essential_indices", [])
            if not isinstance(ghat, list):
                ghat = []
            ghat = [int(x) for x in ghat if isinstance(x, (int, float))]
        except Exception:
            ghat = []
    # Answer = everything before the first '{'
    ans = text.split("{", 1)[0].strip()
    # Strip common lead-ins
    ans = re.sub(r"^(answer\s*:?\s*)", "", ans, flags=re.IGNORECASE).strip()
    ans = ans.split('\n')[0].strip().strip('"').strip('.')
    return ans, ghat


def ask_merged(question, evidence_sents, tok, model, max_new_tokens=200):
    """Single merged call returning (answer, ghat)."""
    ev_text = (
        "\n".join(f"[{i}] {s}" for i, s in evidence_sents)
        if evidence_sents else "(no evidence)"
    )
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
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    raw = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return parse_answer_and_ghat(raw)


def ask_simple(question, evidence_sents, tok, model, max_new_tokens=40):
    """Plain answer call (no rationale) for targeted and random passes."""
    ev_text = (
        "\n".join(f"[{i}] {s}" for i, s in evidence_sents)
        if evidence_sents else "(no evidence)"
    )
    prompt = (
        f"Evidence:\n{ev_text}\n\nQuestion: {question}\n\n"
        "Answer the question based ONLY on the evidence above. Give a concise answer "
        "(a few words), or say \"insufficient evidence\" if the evidence does not "
        "support an answer. Do not use outside knowledge.\n\nAnswer:"
    )
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    ans = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return ans.split('\n')[0].strip().strip('"').strip('.')


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
    tok, model = load_model()
    questions = load_questions()
    K = K_DEFAULT

    print(f"\n=== Merged-prompt variant, K={K}, N={len(questions)} ===\n")
    records = []
    t_global = time.time()
    f1s = []
    n_answered = 0
    n_flip_targeted = 0
    n_flip_random = 0
    for qi, (ds, q) in enumerate(questions, 1):
        all_sents = []
        for d in q["documents"]:
            for s in d["sentences"]:
                all_sents.append(s)
        gold = set(q["gold_sentence_indices"])
        non_gold = [i for i in range(len(all_sents)) if i not in gold]
        n_remove = min(len(gold), len(non_gold))
        full_ev = [(i, s) for i, s in enumerate(all_sents)]
        ghat_removed_ev = [
            (i, s) for i, s in enumerate(all_sents)
            if i not in {x for x in gold if 0 <= x < len(all_sents)}
        ]
        # Note: we delete ghat (not gold) for the targeted pass to test the merged
        # variant's predictions. If ghat is empty or invalid, fall back to gold.
        t = time.time()
        ans_merged, ghat = ask_merged(q["question"], full_ev, tok, model)
        # Build ghat_removed_ev based on predicted ghat (filter to valid indices)
        ghat_valid = sorted({x for x in ghat if 0 <= x < len(all_sents)})
        target_remove = set(ghat_valid) if ghat_valid else gold
        ghat_removed_ev = [
            (i, s) for i, s in enumerate(all_sents) if i not in target_remove
        ]
        a_target = ask_simple(q["question"], ghat_removed_ev, tok, model)

        # K random samples
        random_samples = []
        for k in range(K):
            seed = q["question_id"] * 1009 + 31 * (k + 1) + 7
            local_rng = random.Random(seed)
            rr = set(local_rng.sample(non_gold, n_remove))
            rr_ev = [(i, s) for i, s in enumerate(all_sents) if i not in rr]
            a3 = ask_simple(q["question"], rr_ev, tok, model)
            random_samples.append({
                "k_idx": k + 1, "seed": seed,
                "random_removed": sorted(rr),
                "answer": a3,
                "flipped": (a3.strip().lower() != ans_merged.strip().lower()),
            })

        # ghat F1 vs gold
        f1_score = f1(ghat_valid, sorted(gold))
        f1s.append(f1_score)

        # answered = substantive answer (not abstention)
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
            "cond1_merged_answer": ans_merged,
            "cond1_ghat": ghat_valid,
            "f1_vs_gold": f1_score,
            "target_removed_indices": sorted(target_remove),
            "cond2_target_removed_answer": a_target,
            "targeted_flipped": (a_target.strip().lower() != ans_merged.strip().lower()),
            "per_K_random": random_samples,
        })
        elapsed = time.time() - t
        print(
            f"  [{qi:>2}/{len(questions)}] {ds:>8} q{q['question_id']:>3} "
            f"[{elapsed:>4.1f}s] ghat={len(ghat_valid)} f1={f1_score:.2f} "
            f"ans='{ans_merged[:30]}' targ='{a_target[:30]}'"
        )

    summary = {
        "model": MODEL,
        "K": K,
        "n_questions": len(questions),
        "answered_count": n_answered,
        "f1_ghat_vs_gold_mean": sum(f1s) / len(f1s) if f1s else 0,
        "f1_ghat_vs_gold_median": sorted(f1s)[len(f1s) // 2] if f1s else 0,
        "answered_targeted_flip": n_flip_targeted,
        "answered_random_flip": n_flip_random,
        "answered_random_total": n_answered * K,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_seconds": time.time() - t_global,
        "records": records,
    }
    out_path = RESULTS / "mistral7b_merged.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}  (total {time.time()-t_global:.0f}s)")
    print(f"\n=== Summary ===")
    print(f"Answered (full): {n_answered}/{len(questions)}")
    print(f"Ĝ F1 vs gold: mean={summary['f1_ghat_vs_gold_mean']:.3f}")
    print(f"Targeted flip (answered): {n_flip_targeted}/{n_answered}")
    print(f"Random   flip (answered): {n_flip_random}/{n_answered * K}")


if __name__ == "__main__":
    main()