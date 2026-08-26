"""
run_support_predictors.py — Compare different support-set predictors on the
same N=60 question pool.

Predictors:
  1. Gold (upper bound): human-annotated gold_sentence_indices
  2. LLM self-rationale: the model predicts essential_indices
  3. BGE top-k: pick k=|gold| sentences by cosine similarity to question
  4. Random: pick k=|gold| sentences uniformly

For each, run CF-Verify (full + targeted) on the original N=60.
Output: results/support_predictors.json
"""
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ["HF_HOME"] = "/root/autodl-tmp/migrate_backup/hf_cache"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).parent
ROOT = HERE.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

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


def call_llm(client, model, question, evidence_sents, max_output=80, ask_indices=False, max_retries=6):
    ev_text = "\n".join(f"[{i}] {s}" for i, s in enumerate(evidence_sents))
    if ask_indices:
        user = (
            f"Evidence:\n{ev_text}\n\nQuestion: {question}\n\n"
            "First, answer the question based ONLY on the evidence above. "
            "Give a concise answer (a few words), or say \"insufficient evidence\".\n\n"
            "Then, on a new line, output a JSON object with the field "
            "\"\"essential_indices\"\" listing the integer indices of the evidence "
            "sentences you actually used. Format: {\"\"essential_indices\"\": [0, 5]}\n\nAnswer:"
        )
    else:
        user = f"Question: {question}\n\nEvidence:\n{ev_text}\n\nAnswer:"
    import time as _t
    for attempt in range(max_retries):
        try:
            r = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                max_output_tokens=max_output,
            )
            break
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = min(60 * (attempt + 1), 300)
            print(f"    [retry {attempt+1}/{max_retries}] {str(e)[:80]} — sleeping {wait}s")
            _t.sleep(wait)
    raw = r.output_text.strip()
    if ask_indices:
        # parse answer + JSON
        import re
        m = re.search(r"\{[\s\S]*\}", raw)
        ghat = []
        if m:
            try:
                obj = json.loads(m.group(0))
                ghat = [int(x) for x in obj.get("essential_indices", []) if isinstance(x, (int, float))]
            except Exception:
                pass
        ans = raw.split("{", 1)[0].strip().split("\n")[0].strip().strip('"').strip(".")
        return ans.lower(), ghat
    return raw.split("\n")[0].strip().strip('"').strip(".").lower(), []


def norm(s):
    return s.strip().lower().strip(".").strip('"').strip()


def main():
    client = get_client()
    model = "gpt-5.4"

    print("Loading BGE-base-en-v1.5 from local cache...")
    bge = SentenceTransformer("BAAI/bge-base-en-v1.5",
                              cache_folder="/root/autodl-tmp/migrate_backup/hf_cache")

    hot = json.loads((DATA / "hotpotqa_30_questions.json").read_text())
    wiki = json.loads((DATA / "2wiki_30_questions.json").read_text())
    questions = [("hotpotqa", q) for q in hot] + [("2wiki", q) for q in wiki]

    records = []
    t0 = time.time()
    for qi, (ds_label, q) in enumerate(questions, 1):
        all_sents = []
        for d in q["documents"]:
            for s in d["sentences"]:
                all_sents.append(s)
        gold = set(q["gold_sentence_indices"])
        non_gold = [i for i in range(len(all_sents)) if i not in gold]
        n_remove = len(gold)
        full_ev = [(i, s) for i, s in enumerate(all_sents)]

        # 1. Full answer (one per question)
        full_a, _ = call_llm(client, model, q["question"], full_ev)
        abstained = full_a.startswith("insufficient") or not full_a.strip()

        # 2. LLM self-rationale
        _, ghat = call_llm(client, model, q["question"], full_ev, ask_indices=True)
        ghat_valid = sorted({x for x in ghat if 0 <= x < len(all_sents)})

        # 3. BGE top-k: most similar to question
        embs = bge.encode([q["question"]] + all_sents, normalize_embeddings=True,
                          show_progress_bar=False)
        q_emb = embs[0]
        s_embs = embs[1:]
        sims = [(i, float(np.dot(q_emb, s_embs[i]))) for i in range(len(all_sents))]
        sims.sort(key=lambda x: -x[1])
        bge_top = sorted([i for i, _ in sims[:n_remove]])

        # 4. Random
        rng = random.Random(q["question_id"] * 1009 + 7)
        rnd_sel = sorted(rng.sample(range(len(all_sents)), n_remove))

        predictor_results = {}
        for pred_name, pred_set in [
            ("gold", sorted(gold)),
            ("llm_rationale", ghat_valid if ghat_valid else sorted(gold)),
            ("bge_topk", bge_top),
            ("random", rnd_sel),
        ]:
            target_ev = [(i, s) for i, s in enumerate(all_sents) if i not in set(pred_set)]
            tgt_a, _ = call_llm(client, model, q["question"], target_ev)
            tgt_flipped = (norm(full_a) != norm(tgt_a))
            # F1 vs gold
            ps = set(pred_set); gs = set(gold)
            if not ps and not gs:
                f1 = 1.0
            elif not ps or not gs:
                f1 = 0.0
            else:
                tp = len(ps & gs)
                p = tp / len(ps); r = tp / len(gs)
                f1 = 2*p*r/(p+r) if (p+r) else 0.0
            predictor_results[pred_name] = {
                "predicted_indices": pred_set,
                "target_answer": tgt_a,
                "targeted_flipped": tgt_flipped,
                "f1_vs_gold": f1,
            }

        records.append({
            "dataset": ds_label,
            "qid": q["question_id"],
            "full_answer": full_a,
            "abstained": abstained,
            "gold_size": len(gold),
            "predictors": predictor_results,
        })
        elapsed = time.time() - t0
        if qi % 5 == 0 or qi == len(questions):
            print(f"  [{qi:>2}/{len(questions)}] {ds_label:>8} q{q['question_id']:>3} "
                  f"[{elapsed:>5.0f}s]")

    # Aggregate
    summary = {"model": model, "n_questions": len(questions), "records": records}
    pred_summary = {}
    for pred in ("gold", "llm_rationale", "bge_topk", "random"):
        ans = [r for r in records if not r["abstained"]]
        ft = sum(r["predictors"][pred]["targeted_flipped"] for r in ans) / len(ans)
        f1s = [r["predictors"][pred]["f1_vs_gold"] for r in records]
        pred_summary[pred] = {
            "n_answered": len(ans),
            "F_T": ft,
            "mean_F1_vs_gold": sum(f1s) / len(f1s),
        }
    summary["summary_by_predictor"] = pred_summary

    out_path = RESULTS / "support_predictors.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n=== Support predictor comparison (answered subset) ===")
    print(f"{'Predictor':<15} {'N answered':>10} {'F_T':>8} {'F1 vs gold':>10}")
    for pred, s in pred_summary.items():
        print(f"{pred:<15} {s['n_answered']:>10} {s['F_T']:>8.1%} {s['mean_F1_vs_gold']:>10.3f}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()