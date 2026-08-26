"""
run_llama_cfverify.py — CF-Verify headline protocol on Llama-3.1-8B-Instruct
(fifth model family, Round-4 review #1). Mirrors the Mistral protocol:
gold supports, K=1, N=60 (30 HotpotQA + 30 2Wiki).

Outputs:
    results/llama31_cfverify.json
"""
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
ROOT = HERE.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

MODEL = "/root/autodl-tmp/llama31-8b-instruct"
K = 1

SYSTEM_PROMPT = (
    "You are a question-answering assistant. Answer the question based ONLY "
    "on the evidence provided. Give a concise answer (a few words). If the "
    "evidence does not support an answer, say exactly 'insufficient evidence'. "
    "Do not use any other knowledge."
)


def norm(s):
    return s.strip().lower().strip(".").strip('"').strip()


def ask(model, tok, question, evidence_sents, max_new_tokens=40):
    ev_text = "\n".join(f"[{i}] {s}" for i, s in enumerate(evidence_sents)) if evidence_sents else "(no evidence)"
    prompt = (
        f"{SYSTEM_PROMPT}\n\nEvidence:\n{ev_text}\n\nQuestion: {question}\n\n"
        "Give a concise answer (a few words), or say \"insufficient evidence\".\n\nAnswer:"
    )
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                    tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:],
                      skip_special_tokens=True).split("\n")[0].strip().strip('"').strip(".")


def main():
    print(f"Loading {MODEL} (bf16)...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                 device_map="auto")
    model.eval()
    print(f"Loaded in {time.time()-t0:.1f}s")

    hot = json.loads((DATA / "hotpotqa_30_questions.json").read_text())
    wiki = json.loads((DATA / "2wiki_30_questions.json").read_text())
    questions = [("hotpotqa", q) for q in hot] + [("2wiki", q) for q in wiki]
    print(f"N={len(questions)}, K={K}")

    records = []
    n_ans = n_ft = n_fr = 0
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
        tgt_ev = [(i, s) for i, s in enumerate(sents) if i not in gold]

        full_a = ask(model, tok, q["question"], full_ev)
        tgt_a = ask(model, tok, q["question"], tgt_ev)
        seed = q["question_id"] * 1009 + 31 + 7
        rng = random.Random(seed)
        rr = set(rng.sample(non_gold, n_remove))
        rr_ev = [(i, s) for i, s in enumerate(sents) if i not in rr]
        rnd_a = ask(model, tok, q["question"], rr_ev)

        t_flip = norm(full_a) != norm(tgt_a)
        r_flip = norm(full_a) != norm(rnd_a)
        abst = full_a.lower().startswith("insufficient") or not full_a.strip()
        records.append({"dataset": ds, "qid": q["question_id"],
                        "question": q["question"], "gold_answer": q["answer"],
                        "full_answer": full_a, "target_answer": tgt_a,
                        "random_answer": rnd_a, "abstained": abst,
                        "targeted_flipped": t_flip, "random_flipped": r_flip})
        if not abst:
            n_ans += 1
            n_ft += t_flip
            n_fr += r_flip
        if qi % 10 == 0 or qi == len(questions):
            print(f"  [{qi}/{len(questions)}] [{time.time()-t0:.0f}s] answered={n_ans} T={n_ft} R={n_fr}",
                  flush=True)
            RESULTS.mkdir(exist_ok=True)
            json.dump({"model": "Llama-3.1-8B-Instruct", "K": K, "records": records},
                      open(RESULTS / "llama31_cfverify.json", "w"), indent=1)

    summary = {"model": "Llama-3.1-8B-Instruct", "K": K, "n_questions": len(records),
               "n_answered": n_ans,
               "F_T": n_ft / n_ans if n_ans else 0, "F_R": n_fr / n_ans if n_ans else 0,
               "CFScore": (n_ft - n_fr) / n_ans if n_ans else 0,
               "records": records}
    json.dump(summary, open(RESULTS / "llama31_cfverify.json", "w"), indent=1)
    print(f"\nDONE in {time.time()-t0:.0f}s")
    print(f"Answered: {n_ans}/{len(records)}")
    if n_ans:
        print(f"F_T={summary['F_T']:.1%} F_R={summary['F_R']:.1%} CFScore={summary['CFScore']:+.3f}")


if __name__ == "__main__":
    main()