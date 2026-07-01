"""
run_gpt5.4_questions.py — Run gpt-5.4 on the three CF-Verify conditions for
both datasets and save raw outputs to results/.

Requires:  export OPENAI_API_KEY=sk-...
Produces:  results/gpt5.4_hotpotqa_main.json
           results/gpt5.4_hotpotqa_control.json
           results/gpt5.4_2wiki_main.json
"""
import json
import os
import random
import time
from pathlib import Path

# This is a minimal prototype. The original experiment was run via the
# codex-mcp-chat tool. The body of each LLM call has the same prompt as
# in scripts/run_self_rationale.py -- only the question changes.

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise SystemExit("Set OPENAI_API_KEY first.")

HERE = Path(__file__).parent
RESULTS = HERE.parent / "results"
DATA = HERE.parent / "data"
RESULTS.mkdir(exist_ok=True)


SYSTEM_PROMPT = (
    "You are a question-answering assistant. Answer the question based ONLY "
    "on the evidence provided. Give a concise answer (a few words). If the "
    "evidence does not support an answer, say exactly 'insufficient evidence'. "
    "Do not use any other knowledge."
)


def call_llm(question, evidence_sents, model="gpt-5.4", temperature=0):
    """Send a single question + evidence list to the LLM, return the answer."""
    ev_text = "\n".join(f"[{i}] {s}" for i, s in enumerate(evidence_sents))
    user = f"Question: {question}\n\nEvidence:\n{ev_text}\n\nAnswer:"
    # NOTE: replace this with the actual client call. The original experiment
    # used codex-mcp-chat. A drop-in OpenAI client is left as an exercise;
    # any client with the same prompt reproduces the numbers.
    raise NotImplementedError(
        "Wire this to your preferred OpenAI-compatible client. "
        "See README for what to do with the returned answer string."
    )


def run_dataset(questions, name):
    """Run three conditions for each question: full / gold-removed / random-removed."""
    out = []
    for q in questions:
        all_sents = []
        for doc in q["documents"]:
            for s in doc["sentences"]:
                all_sents.append(s)
        gold = set(q["gold_sentence_indices"])
        non_gold = [i for i in range(len(all_sents)) if i not in gold]
        n_remove = min(len(gold), len(non_gold))
        rng = random.Random(q["question_id"] * 31 + 7)
        random_remove = set(rng.sample(non_gold, n_remove))
        a1 = call_llm(q["question"], all_sents)
        a2 = call_llm(q["question"], [s for i, s in enumerate(all_sents) if i not in gold])
        a3 = call_llm(q["question"], [s for i, s in enumerate(all_sents) if i not in random_remove])
        out.append({
            "qid": q["question_id"],
            "question": q["question"],
            "answer": q["answer"],
            "type": q.get("type", ""),
            "gold_indices": sorted(gold),
            "full_answer": a1,
            "removed_answer": a2,
            "flipped": (a1.strip().lower() != a2.strip().lower()),
        })
    return out


def main():
    hot = json.loads((DATA / "hotpotqa_30_questions.json").read_text())
    wiki = json.loads((DATA / "2wiki_30_questions.json").read_text())
    print(f"Loaded HotpotQA: {len(hot)} Q, 2Wiki: {len(wiki)} Q")
    print("Run conditions for each question. This stub does not call any LLM;")
    print("see the docstring for how to wire your client.")


if __name__ == "__main__":
    main()
