# E1 — Setting C under a saturated solver

**Run:** 2026-09-09 · `Qwen3-4B-Instruct-2507`, 4-bit NF4, RTX 3050 4GB · SciQ test, 884 items ·
9.5 min wall clock.
**Code:** [`code/ex2_counterfactual/run_counterfactual_qwen.py`](../../code/ex2_counterfactual/run_counterfactual_qwen.py)
**Data:** [`results/ex2_counterfactual/results_counterfactual_qwen3_4b.json`](../../results/ex2_counterfactual/results_counterfactual_qwen3_4b.json)
· log `counterfactual_qwen3_4b.log`
**Answers:** README §6.5's named open question — *does the counterfactual intervention still
separate prior from context when the solver is strong enough to be saturated?*

---

## 0. Headline

**Yes — the intervention survives saturation, and it exposes a prior inflation 15× larger
than the encoder ensemble showed.** But the Setting-C label turns out to be a property of
the *solver*, not of the *question* (κ = 0.046 against the encoder ensemble), which is a
direct problem for RQ3.

| | encoder ensemble (`KDA_small`) | **Qwen3-4B** |
|---|---:|---:|
| `both_correct` ∩ CF-eligible — the target set | 381 | **819** |
| `context_dependent` | 75.3% | **37.1%** [33.9, 40.5] |
| **`prior_dependent`** | 22.3% | **62.5%** [59.2, 65.8] |
| `unstable_other` | 2.4% | **0.4%** [0.1, 1.1] |
| Raw Acc$_{wf}$ on eligible | 0.9058 | **0.9977** |
| **Context-verified Acc$_{wf}$** | 0.8581 | **0.3988** |
| **Prior inflation** | **+4.0 pp** | **+59.9 pp** |

Intervals are Wilson 95%.

## 1. Correctness gates

Both passed before any number above was written.

| Gate | Check | Result |
|---|---|---|
| 1 | Setting A/B predictions vs the published ex1b run, 50 samples | **PASS** — 0 mismatches, max \|ΔP\| = **0.00e+00** (bit-exact) |
| 2 | Regenerated counterfactual passages vs the committed dry-run preview | **PASS** — **884/884** identical |

Gate 1 matters because the three-setting prompt builder is re-implemented here rather than
imported ([`kda_qwen_eval.build_prompt`](../../code/ex1_reproduce_KDA/kda_qwen_eval.py) is
boolean, and ex1b's results must stay reproducible). Bit-exact agreement means Settings A
and B are the *same measurement* ex1b published. Gate 2 means Setting C perturbs the
passages identically to what the encoder runs saw — so the cross-solver comparison in §3
differs in the solver and nothing else.

## 2. The pre-registered outcome

Four outcomes were named in the script's docstring before the run, with the H1/H2 cut at
**63.9%** — the closed-book `t5-small-ssm-nq`'s prior-dependence rate, the top of the
encoder suite's monotone trend.

**Result: the H1/H2 boundary.** `prior_dependent` = **62.5%**, 95% CI **[59.2%, 65.8%]**,
which *contains* 63.9%. The monotone extrapolation — *the more a solver knows a priori, the
less its with-fact correctness means* — is **not distinguishable from the weaker H2 reading
at this n**, but Qwen3-4B lands statistically indistinguishable from the most
parametric-knowledge-heavy solver in the encoder suite and far above every other member
(8.2%–31.0%). The theory's prediction is upheld.

> A labelling bug in the pre-registration was found and fixed after the run: the H2 branch
> condition was `< 0.64` while its label read "15–60%", so a 62.5% result was auto-labelled
> H2 despite sitting above the stated band. The bands are now contiguous, the cut is stated
> as 63.9%, and a result whose CI spans it is reported as a boundary case rather than forced
> into one side. No scored value changed — the re-derivation asserts every count and accuracy
> is identical (`summary.reanalysis_note`).

**H3 (context sycophancy) is rejected.** Only 37.1% of the target set follows the perturbed
passage; the model does *not* blindly defer to supplied context. **H4 is rejected
decisively** — `unstable_other` is 3 items (0.37%), against 2.4% for the encoders.

## 3. The finding that was not pre-registered: the label is solver-specific

Both solvers classified the same 860 eligible SciQ items, on byte-identical passages.

| | n | raw agreement | **Cohen's κ** |
|---|---:|---:|---:|
| All three classes | 860 | 0.412 | **0.046** |
| Restricted to items neither called `unstable_other` | 776 | 0.456 | **0.064** |

**κ = 0.046 is chance agreement.** The dominant cell is the disagreement: of the **669**
items the encoder ensemble certified as `context_dependent`, Qwen3-4B calls **392 (58.6%)**
`prior_dependent`.

This matters beyond E1. RQ3's *Our Disentangled Filtering* arm (README §6.5) assumes
Setting-C labels can filter a question bank — that "this question is context-dependent" is a
property of the question. At κ = 0.046 across two solvers it is not; it is a property of
whichever solver you asked. **A filtered bank built on the encoder ensemble and a filtered
bank built on Qwen3-4B would share little more than chance overlap.** Any RQ3 design must
either fix the solver as part of the metric's definition and say so, or aggregate over an
ensemble broad enough that the label stabilises — and that stability has to be measured, not
assumed.

## 4. Supporting numbers

| | value |
|---|---:|
| Acc, Setting A (no passage) | 0.9536 |
| Acc, Setting B (gold passage) | 0.9977 |
| Acc, Setting C **vs gold** | 0.6023 |
| Acc, Setting C **vs counterfactual target** | 0.3884 |
| Baseline buckets | `both_correct` 843 · `wrong_to_correct` 39 · `both_wrong` 2 |
| $P(R^q{=}1) \geq 0.999$ | 827/884 = 93.6% |
| Mean KDA weight $1 - P(R^q{=}1)$ | 0.0458 |

**The target set is 2.15× the encoder run's** (819 vs 381) precisely *because* the solver is
saturated: 843 of 884 items land in `both_correct`, the bucket baseline KDA cannot see. The
RQ1 failure mode makes the RQ2 experiment better powered, not worse.

## 5. What this run deliberately does not report

**$KDA_{cont}$ and the three adjusted estimators.** With one solver $|M| = 1$, the weight
$(1 - P(R^q{=}1))$ cancels between numerator and denominator and $KDA_{cont}$ degenerates to
$P(R^{q+f}{=}1)$ — [`kda_tiny.py`](../../code/ex1_reproduce_KDA/kda_tiny.py) refuses that
configuration by design. Publishing an "LLM-scale adjusted KDA" from a single solver would
be the degenerate quantity under a better name. The classification split, context-verified
accuracy and prior inflation need no ensemble and are what is reported.

That layer becomes available with a second LLM solver. E1 therefore exports its 860
counterfactual passages standalone to
[`counterfactual_passages_for_cloud.json`](../../results/ex2_counterfactual/counterfactual_passages_for_cloud.json)
so the self-contained Kaggle script can run Setting C with no repo import.

## 6. The limitation that bounds every number above

**At LLM scale, `prior_dependent` conflates "ignored the context" with "correctly detected
that the context is false."**

The counterfactual passages are deliberately *false* — "Combustion is the separation of
ions" for a passage about dissociation. A 110M-parameter encoder cannot assess that claim's
plausibility and can only pattern-match. Qwen3-4B can. ClashEval's central finding is that
models arbitrate prior against context *by the context's plausibility*, so a strong model
rejecting an implausible perturbation is behaving correctly — and this pipeline scores it as
`prior_dependent`.

So **62.5% is an upper bound on genuine prior-dependence**, and the 58.6% encoder→Qwen
disagreement in §3 is exactly what that hypothesis predicts.

> **Tested, 2026-09-09 — the hypothesis does not hold up.**
> [`e2_prior_vs_rejection.md`](e2_prior_vs_rejection.md) probed it two independent ways.
> Under an explicit order to treat the passage as authoritative only **26.4%** of the
> `prior_dependent` items flip, and the model's own plausibility penalty for the perturbed
> passage fails to predict which items refuse — **AUC 0.496**, exactly chance, on the
> decisive within-class test. The lower bound on genuine prior-dependence is therefore
> **46.0%** [42.6, 49.5], still 2× the encoder ensemble's 22.3%. The finding below stands
> with its magnitude reduced, not overturned. This is the LLM-scale analogue
of the OBQA problem already recorded as
[`counterfactual_obqa_analysis.md`](counterfactual_obqa_analysis.md) §5 limitation 4
("genuinely ignoring context, vs. sensibly declining to follow an incoherent string"), and
it is now the load-bearing threat to the method rather than a dataset-specific caveat.

Distinguishing the two is the natural next experiment, and it is cheap: the existing runner
already records the full Setting-C distribution, so a plausibility-graded perturbation (or
asking the solver to answer *from the passage alone*) would separate them without new
annotation.

Further caveats carried from the design: single solver, single seed, single quantisation, no
option-order debiasing; SciQ only (OBQA was not run — it is secondary and carries L1); and
limitation L1 itself is unaddressed, so an unknown share of the `prior_dependent` items rest
on perturbations that failed silently (`counterfactual_passage.py` leaves glued residuals
such as `calledoxidants`).
