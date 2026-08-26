"""
run_no_evidence_gate.py — Parametric-answerability gate for CF-Verify (fix B).

Hypothesis: unconditional detection degrades on the N=300 paired set because
many questions are parametrically answerable (model answers with or without
evidence). A cheap gate — asking the model the question with NO evidence —
identifies these: if the model produces a substantive answer with no evidence,
the question is parametrically answerable and behavioural verification does
not apply; if it abstains, CF-Verify is applicable.

For each of the 300 questions (60 orig + 240 ext), run ONE no-evidence probe.
Then evaluate: AUROC of CFScore restricted to questions where the gate
abstains.

Outputs:
    results/no_evidence_gate.json
"""
import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

HERE = Path(__file__).parent
ROOT = HERE.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

GATE_PROMPT_SYS = (
    "You are a question-answering assistant. Answer the question from your own "
    "knowledge. Give a concise answer (a few words). If you do not know the "
    "answer, say exactly 'insufficient knowledge'."
)


def norm(s):
    return s.strip().lower().strip(".").strip('"').strip()


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Set OPENAI_API_KEY first.")
    client = OpenAI(api_key=api_key, base_url=os.environ.get("OPENAI_BASE_URL"))
    model = "gpt-5.4"

    # Load all 300 questions (orig 60 + ext 240) with their CF data
    orig = json.loads(open(RESULTS / "grounded_ungrounded.json").read())
    ext = json.loads(open(RESULTS / "paired_set_ext.json").read())

    questions = []
    for r in orig["records"]:
        questions.append({"src": "orig", "ds": r["dataset"], "qid": r["qid"],
                          "question": r["question"]})
    for r in ext["records"]:
        questions.append({"src": "ext", "ds": r["dataset"], "qid": r["qid"],
                          "question": r["question"]})
    print(f"Total questions: {len(questions)}")

    out_path = RESULTS / "no_evidence_gate.json"
    done = {}
    if out_path.exists():
        ckpt = json.loads(out_path.read_text())
        done = {(r["ds"], r["qid"]): r for r in ckpt["records"]}
        print(f"Resuming: {len(done)} done")

    records = list(done.values())
    t0 = time.time()
    n_new = 0
    for qi, q in enumerate(questions, 1):
        k = (q["ds"], q["qid"])
        if k in done:
            continue
        for attempt in range(10):
            try:
                r = client.responses.create(
                    model=model,
                    input=[
                        {"role": "system", "content": GATE_PROMPT_SYS},
                        {"role": "user", "content": f"Question: {q['question']}\n\nAnswer:"},
                    ],
                    max_output_tokens=40,
                    timeout=120.0,
                )
                break
            except Exception as e:
                if attempt == 9:
                    raise
                time.sleep(min(30 * (attempt + 1), 300))
        ans = r.output_text.strip().split("\n")[0].strip().strip('"').strip(".")
        abst = ans.lower().startswith("insufficient") or not ans.strip()
        records.append({"ds": q["ds"], "qid": q["qid"], "question": q["question"],
                        "no_evidence_answer": ans, "gate_abstains": abst})
        n_new += 1
        if n_new % 20 == 0:
            json.dump({"records": records}, open(out_path, "w"), indent=1)
            print(f"  [{qi}/{len(questions)}] [{time.time()-t0:.0f}s] checkpointed", flush=True)

    json.dump({"records": records}, open(out_path, "w"), indent=1)
    n_abst = sum(1 for r in records if r["gate_abstains"])
    print(f"\nDONE: {len(records)} probes; gate abstains on {n_abst}/{len(records)} = {n_abst/len(records):.1%}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()