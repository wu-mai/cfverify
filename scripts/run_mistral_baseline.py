#!/usr/bin/env python3
"""
Run CF-Verify with Mistral-7B-Instruct-v0.3 on HotpotQA + 2Wiki.
3 conditions per question: full / gold-removed / random-removed.
"""
import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

print("Loading Mistral-7B-Instruct...")
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="auto"
)
model.eval()
print(f"✓ Loaded in {time.time()-t0:.1f}s\n")

def ask(question, evidence_sents, max_new_tokens=40):
    """Ask the model a question with given evidence (list of (idx, sent))."""
    ev_text = "\n".join(f"[{i}] {s}" for i, s in evidence_sents) if evidence_sents else "(no evidence)"
    prompt = f"""Evidence:
{ev_text}

Question: {question}

Answer the question based ONLY on the evidence above. Give a concise answer (a few words), or say "insufficient evidence" if the evidence does not support an answer. Do not use outside knowledge.

Answer:"""
    messages = [{"role": "user", "content": prompt}]
    inputs = tok.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(model.device)
    with torch.no_grad():
        out = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
    ans = tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()
    # Clean up: take first line
    ans = ans.split('\n')[0].strip().strip('"').strip('.')
    return ans


def run_dataset(questions_file, label):
    """Run CF-Verify 3 conditions on a dataset."""
    print(f"\n=== {label} ===")
    with open(questions_file) as f:
        questions = json.load(f)

    import random
    random.seed(123)

    results = []
    for i, q in enumerate(questions):
        qid = q['question_id']
        # Build all sentences
        all_sents = []
        for doc in q['documents']:
            for s in doc['sentences']:
                all_sents.append(s)
        gold = set(q['gold_sentence_indices'])
        n_gold = len(gold)

        # Random removal (same protocol)
        non_gold = [i for i in range(len(all_sents)) if i not in gold]
        random_remove = set(random.sample(non_gold, min(n_gold, len(non_gold))))

        full_ev = [(i, s) for i, s in enumerate(all_sents)]
        gold_removed_ev = [(i, s) for i, s in enumerate(all_sents) if i not in gold]
        rand_removed_ev = [(i, s) for i, s in enumerate(all_sents) if i not in random_remove]

        t = time.time()
        a1 = ask(q['question'], full_ev)
        a2 = ask(q['question'], gold_removed_ev)
        a3 = ask(q['question'], rand_removed_ev)

        results.append({
            'qid': qid,
            'question': q['question'],
            'answer': q['answer'],
            'type': q.get('type', ''),
            'gold_indices': sorted(gold),
            'random_removed': sorted(random_remove),
            'cond1_full': a1,
            'cond2_gold_removed': a2,
            'cond3_random_removed': a3,
        })
        print(f"  Q{qid:2d} [{time.time()-t:.1f}s]: a1='{a1[:40]}' a2='{a2[:40]}' a3='{a3[:40]}'")

    return results


# Run HotpotQA
hotpotqa_results = run_dataset('results/hotpotqa_30_real_questions.json', 'HotpotQA distractor (N=30)')

# Run 2Wiki
twiki_results = run_dataset('results/2wiki_30_questions.json', '2WikiMultihopQA (N=30)')

# Save
output = {
    'model': 'mistralai/Mistral-7B-Instruct-v0.3',
    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    'hotpotqa': hotpotqa_results,
    'twiki': twiki_results,
}
with open('results/mistral7b_baseline.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\n✓ Saved results/mistral7b_baseline.json")
print(f"Total: {len(hotpotqa_results)} HotpotQA + {len(twiki_results)} 2Wiki = {len(hotpotqa_results)+len(twiki_results)} questions")
