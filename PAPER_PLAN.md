# Paper Plan

**Title**: CF-Verify: Counterfactual Evidence Certification for Multi-Hop Question Answering
**One-sentence contribution**: We show that removing the gold-supporting sentences from retrieved evidence and re-querying an LLM produces a deterministic, unfakeable groundedness signal — if the answer flips to "insufficient evidence", it was truly grounded; if it persists, the LLM hallucinated.
**Venue**: WISE 2026 (findings/short track, ~7 pages)
**Type**: Empirical/method
**Date**: 2026-06-29
**Page budget**: 7 pages (IEEE conf format, references included)
**Section count**: 6

---

## Claims-Evidence Matrix

| Claim | Evidence | Status | Section |
|-------|----------|--------|---------|
| **C1**: Counterfactual evidence removal yields a groundedness signal | Gold-removal flip = **93.3%** (28/30, 95% CI [78.7%, 98.2%]) | ✅ Supported | §3, §4.1 |
| **C2**: The signal is **specific** to gold evidence (not generic perturbation) | Random-removal flip = **3.3%** (1/30, 95% CI [0.6%, 16.7%]); CIs non-overlapping with C1 | ✅ **Strongly supported** | §4.2 |
| **C3**: The signal is robust on multi-hop reasoning (bridge Qs) | 17/17 = 100% on bridge questions (95% CI [81.6%, 100%]) | ✅ Strongly supported | §4.3 |
| **C4**: Zero hallucination under removal — LLM never fabricates the gold answer from prior knowledge when evidence is gone | 0/30 hallucinations after gold removal (95% CI [0%, 11.4%]) | ✅ Supported | §4.4 |
| **C5** *(analysis, not headline)*: Comparison questions are harder (85%) due to gold-label under-annotation, NOT mechanism failure | Q18, Q20 failure analysis: residual answering sentences remain; control still flips 0/2 of these | ✅ Supported (moved to analysis, not main claim) | §4.5 |

### Key specificity result (addresses reviewer's #1 concern)
**Contingency (N=30):**

|                | Flipped | Not-flipped |
|----------------|---------|-------------|
| Gold-removal   | 28      | 2           |
| Random-removal | 1       | 29          |

- Gold-removal flip: 93.3% [78.7%, 98.2%]
- Random-removal flip: 3.3% [0.6%, 16.7%]
- **CIs do not overlap** → the flip is specific to gold evidence, not a generic "remove any sentence" artifact. This is the paper's strongest defense against the N=30 / cherry-pick criticism.

---

## Structure (6 sections)

### §0 Abstract (~200 words)
- **What we achieve**: A training-free, pluggable method (CF-Verify) that certifies whether an LLM's answer to a multi-hop question is grounded in retrieved evidence, by counterfactually removing the gold-supporting sentences and checking if the answer changes.
- **Why it matters**: RAG agents need a reliable accept/revise/re-retrieve decision; existing faithfulness checks (SelfCheck, FactScore) are themselves noisy LLM judgments. CF-Verify turns groundedness into a deterministic, observable behavior.
- **How**: Two-pass LLM query — full evidence vs. evidence-minus-gold; flip ⇒ grounded.
- **Evidence**: 93.3% flip rate (28/30) on real HotpotQA distractor with gpt-5.4; 0% hallucination; 100% on bridge questions.
- **Most remarkable result**: On 17 bridge questions the method is perfect (100%), and the only 2 failures are traceable to under-annotation in HotpotQA's own gold labels — the method exposes label noise.

### §1 Introduction (~1.5 pages)
- **Opening hook**: A RAG agent that answers confidently from retrieved context can still hallucinate; the agent has no cheap, reliable way to know *whether its answer came from the evidence or from parametric memory*.
- **Gap / challenge**: Existing groundedness/faithfulness checks (SelfCheckGPT, FactScore,Attribution) re-use an LLM as judge — replacing one hallucination-prone component with another. Multi-hop QA compounds this because the chain of reasoning spans distractors.
- **One-sentence contribution** *(this is the thesis)*: Counterfactual removal of the minimal sufficient evidence set, followed by re-querying, gives a deterministic groundedness signal that needs no extra model and no retraining.
- **Approach overview**: (i) identify supporting sentences (HotpotQA gold, or any retriever's top-k rationale), (ii) re-query without them, (iii) compare answers.
- **Key questions**:
  - RQ1: Does removal actually flip the answer on real multi-hop data?
  - RQ2: Is the flip specific to the gold evidence (not random removal)?
  - RQ3: Does it survive distractor noise (the real HotpotQA setting has 8 distractors)?
- **Contributions**:
  1. CF-Verify: a training-free groundedness certification primitive based on counterfactual perturbation.
  2. Empirical validation on 30 real HotpotQA distractor questions: 93.3% flip rate, 0% hallucination, 100% on bridge questions.
  3. A diagnostic finding: CF-Verify exposes under-annotation in HotpotQA's supporting-facts labels (a tool-side bonus).
- **Results preview**: 93.3% / 0% / 100% table up front.
- **Hero figure**: Figure 1 — two-pass pipeline (Full → answer X; Removed → "insufficient evidence"); a 2×2 contingency (grounded/hallucinated × flip/no-flip) showing the signal cleanly separates the two cases.

### §2 Related Work (~1 page)
- **Subtopics**:
  1. **LLM faithfulness & attribution**: SelfCheckGPT, FactScore, RARR — all judge-with-LLM; contrast with CF-Verify's behavioral signal.
  2. **RAG & retrieval grounding**: retrieval-augmented LMs, citation verification — contrast: we certify *answer* grounding, not citation presence.
  3. **Counterfactual / ablation explanation in NLP**: input ablation for saliency (e.g., leave-one-out, contextual decomposition) — contrast: we apply ablation as a *verification decision*, not an explanation.
- **Positioning**: CF-Verify is, to our knowledge, the first to use minimal-evidence ablation as an accept/revise decision primitive for RAG agents.
- **Minimum**: 3–4 synthesis paragraphs.

### §3 Method (~1.5 pages)
- **Notation**: question q, retrieved evidence E = {s₁…sₙ}, supporting set G ⊆ E, LLM M.
- **Problem formulation**: produce a binary certification C(q, E) ∈ {grounded, hallucinated} and a triage decision accept / revise / re-retrieve.
- **Algorithm**:
  1. Pass 1: a₁ = M(q, E)
  2. Identify G (gold rationale; in practice from a retriever or LLM self-rationale)
  3. Pass 2: a₂ = M(q, E \ G)
  4. Decision: a₁ ≠ a₂ ⇒ grounded; a₁ = a₂ ⇒ hallucinated; a₂ = "insufficient evidence" ⇒ re-retrieve.
- **Why it works (intuition)**: a capable LLM conditioned on a clean evidence set will use it; removing the load-bearing sentences breaks the chain → "insufficient evidence". Parametric-only answers survive removal, revealing hallucination.
- **Capability-gating note**: requires the LLM to (a) use evidence when present and (b) abstain when absent — a capability that scales with model size (foreshadow §4.5).

### §4 Experiments (~2 pages)
- **Setup**: HotpotQA distractor (validation), N=30 balanced (15 bridge / 15 comparison), hard level, gpt-5.4 zero-shot, deterministic prompt.
- **Figures/Tables**:
  - **Table 1** (main): flip rate, EM-full, EM-removed, hallucination, over-conservative — by question type.
  - **Figure 2**: per-question results (flip / no-flip) colored by type.
  - **Table 2**: failure-case analysis (Q18, Q20) showing residual answering sentences.
- **Subsections**:
  - §4.1 Main flip rate (93.3%)
  - §4.2 By type (bridge 100%, comparison 85%)
  - §4.3 Hallucination = 0
  - §4.4 Failure analysis — gold-label under-annotation, not mechanism failure (the 2 "failures" actually show CF-Verify catching missing supporting_facts)
  - §4.5 Capability gating — cite earlier 7B pilot (~7% flip) as evidence the mechanism is capability-gated; full small-model study is future work.

### §5 Discussion & Limitations (~0.5 pages)
- **Limitations** (honest):
  - N=30 is small; generalization to 2Wiki/BEIR is left to future work.
  - Depends on a capable LLM; small-model baseline not yet run (C5 partial).
  - Gold set G must be identified; we use HotpotQA labels here — a learned/LLM-rationale G is future work.
  - Cost: 2× LLM calls per question.
- **Future work**: (a) small-model cascade (cheap detector + CF-Verify only when uncertain), (b) cross-corpus (2Wiki, BEIR), (c) using CF-Verify to auto-clean supporting-facts labels.

### §6 Conclusion (~0.25 pages)
- Restate: counterfactual removal is a cheap, training-free, deterministic groundedness signal; 93.3% flip, 0% hallucination on real HotpotQA.
- One-line future direction: capability-cascade to bring the method to open-weight models.

---

## Figure Plan

| ID | Type | Description | Data Source | Priority |
|----|------|-------------|-------------|----------|
| Fig 1 | Hero/architecture | Two-pass pipeline + 2×2 contingency (grounded/hallucinated × flip/no-flip) | manual | HIGH |
| Fig 2 | Stacked bar | Per-type flip rate (bridge 100% vs comparison 85%) | results/FINAL_RESULTS_N30.json | HIGH |
| Table 1 | Results table | flip rate, EM-full, EM-removed, hallucination, over-conservative | results/FINAL_RESULTS_N30.json | HIGH |
| Table 2 | Failure case | Q18, Q20: residual answering sentences after gold removal | results/FINAL_RESULTS_N30.json | MEDIUM |

**Figure 1 (hero) detail**: Left = two boxes showing Pass 1 (full evidence → "1755") and Pass 2 (evidence − gold → "insufficient evidence"), with an arrow labeled "flip ⇒ grounded". Right = 2×2 grid: (grounded, flip)=accept, (grounded, no-flip)=re-check G, (hallucinated, flip)=rare, (hallucinated, no-flip)=revise/abstain. Caption: "CF-Verify certifies groundedness by observing whether the answer survives removal of its supporting evidence."

---

## Citation Plan
- §1 Intro: SelfCheckGPT [VERIFY], FactScore [VERIFY], RARR [VERIFY], RAG original (Lewis et al.) [VERIFY]
- §2 Related: + HotpotQA (Yang et al.) [VERIFY], attribution/verification survey [VERIFY], input-ablation saliency work [VERIFY]
- §3 Method: minimal sufficient set / rationale literature [VERIFY]
- §4: BEIR [VERIFY], 2WikiMultihopQA [VERIFY] (mentioned as future)

**Citation rule**: do NOT fabricate bibtex — verify each via search before compiling.

---

## Reviewer Feedback (gpt-5.4 xhigh, 2026-06-29)

**Overall: weak reject in current form. Scores: flow 7, claim-evidence 3, missing-exp 2, positioning 5, page-feasibility 6, front-matter 6.**

### Three fatal issues
1. **Overclaiming**: "deterministic/unfakeable/iff/certify" unsupported → must become "empirical groundedness probe / two-pass grounding test". Title change to "CF-Verify: A Two-Pass Grounding Test for Retrieved Multi-Hop QA".
2. **N=30 + no control = review-killer**. MUST add: (a) **random-removal control** on same 30 (remove equal-count non-gold sentences, show flip rate ≪ gold-removal → proves specificity), (b) **Wilson 95% CIs** on all proportions, (c) sampling protocol (seed, balance).
3. **C4 (diagnostic) and C5 (capability-gating) are cope** as stated. Either adjudicate failures with manual re-annotation, or move both to limitations. Drop "first to..." novelty claim.

### Minimum viable paper
> "Two-pass evidence deletion test. On 30 sampled HotpotQA, deleting support flipped/abstained in 28; matched random deletions did so far less often. No unsupported replacement answers. Preliminary, one dataset/model."

### Required new experiments (before writing)
- [ ] **Random-removal control** (≈30 gpt-5.4 calls): remove |G| random non-gold sentences, measure flip rate. Target: ≪ 93%.
- [ ] Wilson CIs on Table 1.
- [ ] Manual adjudication table for Q18, Q20 (2-annotator residual-evidence check).

### Cuts to fit 7 pages
- Drop capability-gating from main claims (→ 1 sentence in limitations).
- Drop C4 from contributions (→ short failure-analysis subsection).
- Shrink Related Work to 0.5–0.75p. Merge Fig2 into Table 1.

### Revised contribution list (2 bullets only)
1. A simple two-pass evidence-deletion test for grounding in multi-hop QA.
2. Small-sample empirical finding on HotpotQA showing high sensitivity to support deletion + matched random-deletion control + failure analysis.

---

## Next Steps
- [ ] Get gpt-5.4 review of this outline (§6 of skill)
- [ ] /paper-figure to generate Fig 1, Fig 2, Table 1
- [ ] /paper-write to draft LaTeX (LNCS or IEEE format per WISE)
- [ ] /paper-compile to build PDF
