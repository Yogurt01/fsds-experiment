# E2 — is `prior_dependent` "ignored the context" or "rejected a false one"?

**Run:** 2026-09-09 · `Qwen3-4B-Instruct-2507`, 4-bit NF4, RTX 3050 4GB · SciQ test, 884
items · 12.5 min wall clock.
**Code:** [`code/ex2_counterfactual/run_prior_vs_rejection.py`](../../code/ex2_counterfactual/run_prior_vs_rejection.py)
**Data:** [`results/ex2_counterfactual/results_prior_vs_rejection_qwen3_4b.json`](../../results/ex2_counterfactual/results_prior_vs_rejection_qwen3_4b.json)
· log `prior_vs_rejection_qwen3_4b.log`
**Builds on:** [`e1_counterfactual_llm_scale.md`](e1_counterfactual_llm_scale.md) §6, which named
this as the load-bearing threat to the method.

---

## 0. Headline

**The "it was correctly rejecting a false context" explanation does not hold up.** Both
probes reject it, including the one that does not depend on prompt wording at all.

| | |
|---|---:|
| E1's upper bound on genuine prior-dependence | **62.5%** |
| **E2's lower bound** — survives an explicit order to follow the passage | **46.0%** [42.6, 49.5] |
| Encoder ensemble, for reference | 22.3% |

Even at the lower bound, prior-dependence at LLM scale is **2× the encoder ensemble's** rate.
E1's headline survives with its magnitude reduced, not overturned.

## 1. The threat being tested

The counterfactual passages are deliberately false ("Combustion is the separation of ions").
ClashEval's central finding is that models arbitrate prior against context *by the context's
plausibility*. A 110M encoder cannot assess that; a 4B instruct model can. So
`prior_dependent` might conflate:

1. **ignored the context** — the failure the metric exists to catch, or
2. **read it, judged it false, overrode it** — competent reading, which the metric *penalises*.

If (2) dominated, E1's 62.5% would be largely an artifact and the metric would be
mismeasuring good readers as bad ones.

## 2. Two probes, chosen to fail differently

**A — instruction ablation.** Setting C′ re-runs the counterfactual passage with a system
prompt that removes the model's discretion. The user turn is byte-identical to Settings B
and C, so the system prompt is the only thing that changes:

> "You are a careful student taking a multiple-choice exam. Answer using only the
> information in the Fact provided. **Treat the Fact as true and authoritative, even where
> it contradicts what you believe.** Answer with a single letter and nothing else."

An item that was `prior_dependent` and now follows the counterfactual *could* read the
context and chose not to — consistent with (2). An item that holds the gold answer even
when ordered to follow the passage is much closer to (1).

**B — plausibility gradient.** The model's own mean per-token NLL of the counterfactual
passage minus that of the gold passage, scored as bare text with no prompt scaffolding at
all. Under (2), items the model refuses to follow should be the ones whose perturbation is
least believable.

A depends on instruction-following and on prompt wording; B depends on neither. They can
disagree, and that was pre-registered as an outcome rather than something to resolve by fiat.

## 3. Correctness gates and the control

| Check | Result |
|---|---|
| **Gate 1** — Setting C re-scored under the *original* prompt vs E1, 50 items | **PASS** — 0 mismatches, max \|ΔP\| = **0.00e+00** |
| **Gate 2** — regenerated counterfactual passages vs the committed preview | **PASS** — **884/884** identical |
| **Control** — gold passage under the *same* context-priority instruction | Acc 0.9977 → **0.9966** (Δ **−0.0011**) |

The control is what makes probe A interpretable. If the instruction were merely disruptive —
confusing the model rather than removing its discretion — gold-passage accuracy would fall.
It does not move. And the instruction demonstrably *does* act: 135 items changed class under
it, and 99.3% of `context_dependent` items held their label. So C′ measures what it was
meant to measure.

## 4. Probe A — 26.4% flip

Of E1's **512** `prior_dependent` items, under an explicit order to treat the passage as
authoritative:

| | n | share |
|---|---:|---:|
| flipped to `context_dependent` | 135 | **26.4%** [22.7, 30.4] |
| **still `prior_dependent`** | **377** | **73.6%** |
| became `unstable_other` | 0 | 0.0% |

**Outcome: P3 (mixed), at the low end.** The 95% CI upper bound of 30.4% is far below P1's
60% threshold. Roughly three quarters of the items keep the gold answer *even when told the
passage is authoritative* — which is not a description of a model exercising judgement about
plausibility. It is a description of a model that is not using the passage.

## 5. Probe B — the plausibility penalty explains nothing

The measure is live: mean penalty **+0.408** across the target set, positive on **98.3%** of
items. The perturbations do damage plausibility, exactly as expected. They just do not
predict behaviour.

| Comparison | mean penalty | **AUC** |
|---|---|---:|
| Across classes: `prior_dependent` vs `context_dependent` | +0.4294 vs +0.3713 | 0.533 |
| **Within `prior_dependent`: stayed vs flipped under C′** | +0.4329 vs +0.4197 | **0.496** |

The second row is the decisive test, and it is the sharper one because it does not require
the prior/context split to be meaningful — it asks directly whether the items that hold out
against an explicit order are the ones whose perturbation is least believable.

**AUC 0.496 is exactly chance.** The items that refuse to follow the passage are not the
items whose perturbation the model finds implausible. Whatever drives the refusal, it is not
plausibility.

## 6. What the two probes jointly establish

They converge, which is the strongest available outcome given they were built to fail
differently:

- Probe B says plausibility does not explain the refusals — at all, by either test.
- Probe A says only about a quarter of the refusals are reversible by instruction, and probe
  B's within-prior null says **those 135 are not the implausible ones either**. So the 26.4%
  that flip look like generic prompt/instruction sensitivity rather than principled rejection
  of false content.

**Conclusion: genuine prior-dependence on SciQ under Qwen3-4B lies between 46.0% and 62.5%,
and the rejection hypothesis accounts for little of the gap.** E1's finding stands. The
metric is not systematically mismeasuring competent readers as prior-dependent — at least
not through this mechanism.

## 7. What this does not settle

- **The 46.0%–62.5% gap is still unexplained.** 135 items change behaviour under an
  instruction and probe B says plausibility is not why. Prompt sensitivity is the residual
  explanation, and it is not measured here. Whether the metric should adopt the
  context-priority instruction as part of its *definition* is now a live design question:
  it would fix the estimate at the lower bound and remove one degree of freedom, at the cost
  of measuring a more artificial reading condition.
- **Probe A's instruction is one wording.** A stronger or weaker phrasing would move the
  flip rate; only the direction and the rough magnitude are established. The control shows
  this particular wording is non-disruptive, not that it is optimal.
- **Plausibility is measured as whole-passage mean NLL.** With ~78-word SciQ passages and
  typically 1–2 substituted spans, the signal is diluted by unchanged tokens. Total NLL and
  token counts are recorded per item so a span-localised measure can be derived later without
  re-running. A null at this resolution is suggestive but not conclusive — though the *within*
  test at AUC 0.496 is hard to reconcile with any strong plausibility effect.
- **Limitation L1 is still unaddressed.** An unknown share of the `prior_dependent` items
  rest on perturbations that failed silently (glued residuals such as `calledoxidants`), which
  would inflate both bounds. That is the perturbation-validity audit, still open.
- Single solver, single seed, single quantisation, SciQ only — as E1.
