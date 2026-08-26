# CF-Verify

A counterfactual evidence-deletion diagnostic for retrieval-augmented multi-hop QA.

**Paper:** *CF-Verify: Counterfactual Evidence Deletion as a Behavioural Evidence-Dependence Diagnostic for Retrieval-Augmented Multi-Hop QA* (ARR submission, 8 pages + appendix, ACL format).

## What CF-Verify does

A RAG agent that answers a multi-hop question from retrieved documents cannot reliably tell whether its answer came from those documents or from parametric memory. CF-Verify separates **evidence compatibility** (what LLM-judge faithfulness checks measure) from **behavioural evidence dependence** (whether the model would change its answer if the evidence were removed):

1. **Self-rationale:** a single LLM call predicts which evidence sentences are necessary, producing a candidate support set G-hat.
2. **Targeted pass:** the LLM re-answers with G-hat removed; the answer is compared with the original.
3. **Matched random control:** the same comparison with an equal-size random non-support subset removed, matched on size, source, and disjointness from G-hat.
4. **Decision rule:** CFScore = F_T − F_R drives a three-way routing: **accept** / **revise** / **re-retrieve**.

## Headline results

Pooled GPT-5.4 over three benchmarks (HotpotQA, 2WikiMultihopQA, MuSiQue), 321 answered questions:

| Setting | Answered | F_T / F_R | CFScore |
|---|---|---|---|
| GPT-5.4 pooled (95% boot CI) | 321/617 | 80.8% / 14.5% | **+0.657** [+0.601, +0.713] |
| GPT-5.4-mini | 52/60 | 84.6% / 19.2% | +0.654 |
| GPT-5.5 | 53/60 | 79.2% / 17.0% | +0.623 |
| Llama-3.1-8B-Instruct | 36/60 | 94.4% / 13.9% | +0.806 |
| Mistral-7B (separate calls) | 14/60 | 64.3% / 0.0% | +0.643 |

Detection on a 300-question grounded/ungrounded paired benchmark: CF-Verify AUROC **0.784** (curated 60-question subset) vs LLM-judge **0.533** and BGE-similarity **0.664**; BGE+CF logistic ensemble **0.804**. At full scale, discrimination concentrates on evidence-sensitive questions (62% clean quadrant, AUROC 0.762); a no-evidence parametric-answerability gate and a format-consistent random baseline recover most of the gap.

## Repository layout

```
paper/           ARR submission (main.tex, main.pdf, references.bib, figures/)
analysis.py      Reproduces headline tables from results/*.json
data/            Question pools (HotpotQA/2Wiki/MuSiQue JSON)
scripts/         All experiment scripts (see below)
results/         39 result JSONs (raw outputs of every run in the paper)
```

Raw source datasets are not included (size); download from HuggingFace:
hotpotqa/hotpot_qa (distractor, validation), xanhho/2WikiMultihopQA (dev),
dgslibisey/MuSiQue (dev).

## Reproducing the experiments

```bash
# Headline + multimodel + larger samples (gpt-5.4 via API)
export OPENAI_API_KEY=... OPENAI_BASE_URL=...
python scripts/run_gpt5.4_questions.py          # N=57 headline
python scripts/run_multimodel.py gpt-5.4-mini gpt-5.5
python scripts/run_larger_sample.py --n-hot 210 --n-wiki 210
python scripts/run_musique.py                    # third dataset

# Grounded/ungrounded paired detection (N=60 -> 300)
python scripts/run_grounded_ungrounded.py
python scripts/extend_paired_set.py

# Baselines and controls
python scripts/run_baselines.py                 # LLM-judge + BGE similarity
python scripts/run_content_matched.py           # content-matched random control
python scripts/run_no_evidence_gate.py          # parametric-answerability gate

# Local models (GPU)
python scripts/run_mistral_experiments.py       # K-ablation
python scripts/run_mistral_merged_v2.py         # merged-prompt variants
python scripts/run_llama_cfverify.py            # fifth model family

# Analysis
python analysis.py                              # headline tables
python scripts/aggregate_300.py                 # N=300 detection AUROC
python scripts/conditional_300.py               # evidence-sensitivity stratification
```

## Results provenance

Every number in the paper maps to a JSON in `results/`:
gpt5.4_hotpotqa_main.json, gpt5.4_2wiki_main.json, larger_sample_gpt54_n{60,80,420}.json, musique_gpt54.json, multimodel_gpt-5.4-mini.json, multimodel_gpt-5.5.json, llama31_cfverify.json, mistral7b_Kablation.json, mistral7b_merged{,_v2,_v3,_v4,_v5}.json, grounded_ungrounded.json, paired_set_ext.json, baselines_vs_cfverify.json, content_matched.json, no_evidence_gate.json, cfscore_ci.json, conditional_300.json, ...

## License

MIT (see LICENSE).
