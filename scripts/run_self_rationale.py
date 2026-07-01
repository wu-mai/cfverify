"""
run_self_rationale.py — Get LLM self-rationale (predicted support set Ĝ)
for each question, then re-run CF-Verify with Ĝ instead of true G.

Requires:  export OPENAI_API_KEY=sk-...
Produces:  results/self_rationale_ghat.json
"""
import json
import os
from pathlib import Path

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise SystemExit("Set OPENAI_API_KEY first.")

HERE = Path(__file__).parent
DATA = HERE.parent / "data"
RESULTS = HERE.parent / "results"
RESULTS.mkdir(exist_ok=True)

SELF_RATIONALE_PROMPT = """You are a careful question-answering system. Given a question and a set of evidence sentences (each numbered), identify which evidence sentences are NECESSARY to answer the question.

Question: {question}

Evidence:
{evidence}

List the indices of the evidence sentences that are necessary to answer the question. Output strictly as a JSON object with a single field "essential_indices" containing the list of integers. Only include sentences that are strictly necessary; if a sentence is just additional context but not directly needed, do not include it.

Output format: {{"essential_indices": [list of integers]}}"""


def call_llm(prompt, model="gpt-5.4"):
    """Single gpt-5.4 call returning the response text. Replace with your client."""
    raise NotImplementedError(
        "Wire this to your preferred OpenAI-compatible client. "
        "Return the assistant message text."
    )


def main():
    hot = json.loads((DATA / "hotpotqa_30_questions.json").read_text())
    wiki = json.loads((DATA / "2wiki_30_questions.json").read_text())
    out = {}
    for ds_name, ds in [("hotpotqa", hot), ("2wiki", wiki)]:
        for q in ds:
            all_sents = []
            for doc in q["documents"]:
                for s in doc["sentences"]:
                    all_sents.append(s)
            ev_text = "\n".join(f"[{i}] {s}" for i, s in enumerate(all_sents))
            prompt = SELF_RATIONALE_PROMPT.format(question=q["question"], evidence=ev_text)
            response = call_llm(prompt)
            try:
                indices = json.loads(response)["essential_indices"]
            except Exception:
                indices = []
            out[f"{ds_name}_{q['question_id']}"] = {
                "qid": q["question_id"],
                "dataset": ds_name,
                "essential_indices": indices,
            }
    (RESULTS / "self_rationale_ghat.json").write_text(json.dumps(out, indent=2))
    print(f"Wrote {len(out)} entries to results/self_rationale_ghat.json")


if __name__ == "__main__":
    main()
