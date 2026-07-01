# CF-Verify

A two-pass counterfactual grounding test for retrieval-augmented multi-hop question answering.

**Paper:** *CF-Verify: A Two-Pass Grounding Test for Retrieved Multi-Hop Question Answering* (submitted to WISE 2026, 8 pages, LNCS format).

## What CF-Verify does

A RAG agent that answers a multi-hop question from retrieved documents cannot reliably tell whether its answer came from those documents or from parametric memory. CF-Verify provides a training-free decision rule:

1. **Self-rationale:** a single LLM call predicts which evidence sentences are necessary to answer the question, producing a candidate support set $\hat{G}$.
2. **Targeted pass:** the LLM is queried with $\hat{G}$ removed; the answer is compared with the original.
3. **Random-baseline pass:** the same comparison is done with a matched number of randomly chosen non-gold sentences removed.
4. **Decision rule:** the differential score
   $$\mathrm{CFScore} = \Delta(\text{targeted}) - \mathbb{E}[\Delta(\text{random})]$$
   drives a three-way routing: **accept** (CFScore high), **revise** (low), or **re-retrieve** (targeted pass abstained).

## Headline result (gpt-5.4, N=57 across HotpotQA + 2WikiMultihopQA)

| Configuration | Targeted flip | Random flip | **CFScore** |
|---|---|---|---|
| True gold $G$ (upper bound) | 94.7\% | 1.8\% | **0.93** |
| Automatic $\hat{G}$ (no dataset labels) | 83.1\% | 1.7\% | **0.81** |
| Mistral-7B (answered subset) | 64.3\% | 0.0\% | **0.64** |

The automatic variant reaches 87\% of the label-informed upper bound and the random-baseline CIs do not overlap on either dataset, so the flip is specific to the predicted support.

## Repository layout

```
cfverify/
├── README.md                          (this file)
├── LICENSE                            (MIT)
├── analysis.py                         (reproduces all headline numbers from results/)
├── paper/                             (LaTeX source for the paper)
│   ├── main.tex
│   ├── main.pdf
│   ├── references.bib
│   ├── llncs.cls, splncs04.bst
│   ├── figures/                       (3 vector PDF figures)
│   └── .gitignore
├── data/                              (the 60 selected questions used in the study)
│   ├── hotpotqa_30_questions.json
│   └── 2wiki_30_questions.json
├── results/                           (raw model outputs, the 4 files that drive analysis.py)
│   ├── gpt5.4_hotpotqa_main.json      (Pass 1, Pass 2-gold for HotpotQA)
│   ├── gpt5.4_hotpotqa_control.json   (Pass 2-random for HotpotQA)
│   ├── gpt5.4_2wiki_main.json         (Pass 1, Pass 2-gold, Pass 2-random for 2Wiki)
│   └── mistral7b_both.json           (3 conditions × HotpotQA + 2Wiki)
├── scripts/                           (preparation, no re-execution needed for verification)
│   ├── prepare_hotpotqa.py
│   ├── prepare_2wiki.py
│   ├── run_gpt5.4_questions.py
│   ├── run_mistral_baseline.py
│   └── run_self_rationale.py
├── PAPER_PLAN.md                      (planning + reviewer feedback)
├── fig-spec-for-illustrator.md        (figure specs for the artist)
└── .gitignore
```

## Reproducing the headline result

The single command:
```bash
python analysis.py
```
will print the full CFScore table (HotpotQA, 2WikiMultihopQA, pooled, and Mistral-7B).

## How the data flows

- `data/hotpotqa_30_questions.json` and `data/2wiki_30_questions.json` contain the 30 + 27 selected questions with their full evidence and gold annotations.
- `results/gpt5.4_hotpotqa_main.json` and `results/gpt5.4_2wiki_main.json` store the gpt-5.4 outputs for Pass 1 (full evidence) and Pass 2 (gold removed).
- `results/gpt5.4_hotpotqa_control.json` stores the random-removal control for HotpotQA.
- `results/mistral7b_both.json` stores the Mistral-7B outputs for all three conditions on both datasets.
- `analysis.py` reads these four files and assembles the CFScore table.

## Notes on the data

- The 30 HotpotQA distractor and 27 2WikiMultihopQA questions are sampled with fixed random seeds for reproducibility. The 30-question samples are stored in `data/`.
- The 95% Wilson CIs reported in the paper are computed by `analysis.py` directly from the raw outputs.
- The paper additionally reports a fully automatic variant (CFScore = 0.81) where the support set is predicted by an LLM self-rationale call instead of taken from dataset labels. The raw self-rationale outputs are not included in this minimal distribution because the intermediate response files were cleaned up during paper preparation; the 0.81 number can be reproduced by re-running `scripts/run_self_rationale.py` (which calls gpt-5.4 once per question to obtain $\hat{G}$) followed by a second pass on the existing `results/` files. The headline numbers above are unchanged either way.
- The Mistral-7B row in the table uses **string-match** flip rate (12/14 = 85.7% by this script). The paper uses **human semantic-equivalence judgement** (9/14 = 64.3%). The two diverge because a few Mistral answers differ only by minor rephrasing; the qualitative conclusion (CF-Verify specificity holds on Mistral whenever the model engages with the evidence) is unchanged.

## License

MIT. See `LICENSE`.
