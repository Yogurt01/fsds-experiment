# Experiment 6 — the P/S/F/D dependency score

**Written:** 2026-09-14. **Status:** prototype, evaluated, **not adopted**.
**Scope:** a parallel RQ2 formulation, computed from committed Setting-A/B/C outputs and compared
against the existing estimators. It **does not modify** the hard / soft / sample-exclusion adjusted
KDA estimators in `code/ex2_counterfactual/`, and no RQ1 experiment was reopened.

**No model inference was run.** Every number below is derived from probability vectors already
stored in `results/ex2_counterfactual/results_counterfactual_*.json`. Nothing here is fabricated,
and nothing here required a GPU.

| | |
|---|---|
| Code | [`code/ex6_psfd_score/psfd.py`](../../code/ex6_psfd_score/psfd.py) · [`compute_psfd_scores.py`](../../code/ex6_psfd_score/compute_psfd_scores.py) · [`evaluate_psfd.py`](../../code/ex6_psfd_score/evaluate_psfd.py) |
| Outputs | `results/ex6_psfd_score/psfd_scores_{sciq_test_full,obqa_test_full,obqa_test_exact_tier,qwen3_4b_sciq}.json` · `psfd_evaluation.json` |
| Inputs | the four committed Setting-C runs; `results/ex5_failure_analysis/rq1_flagged_questions{,_rebuilt_obqa}.json` |
| Ordinal | `ex6` claimed on the build-order convention; the RQ3 filtering study's reservation moved to `ex7` (PROJECT_READING_GUIDE §6.4 item 13, `NEXT_PHASE_HANDOFF.md` §3.3 item 10) |

```bash
python code/ex6_psfd_score/compute_psfd_scores.py && python code/ex6_psfd_score/evaluate_psfd.py
```

---

## 0. Executive summary

> **Two claims, deliberately kept apart. They are not the same claim and only the second is
> negative.**
>
> **Claim 1 — the P/S/F decomposition has diagnostic value, and this is supported.** Working from
> Settings A/B/C, the decomposition sees inside the `both_correct` bucket that `KDA_cont` is
> structurally blind to: within those 381 SciQ items, `D` separates context- from prior-dependent
> at **AUC 0.772** against `KDA_cont`'s **0.617**, and across all 860 eligible items **0.824 vs
> 0.549**. The three signals carry real information that the baseline estimator does not.
>
> **Claim 2 — the specific multiplicative combination `(1−P)·S·max(F,0)` is *not* supported as the
> way to combine them.** Pre-registered criterion 5 fails on **every** split, by −0.075 to −0.399
> AUC, and `F` alone outperforms both `D` and `D'` throughout. This is a finding about the *product
> form*, not about the underlying signals.
>
> **The correct reading is "the product form needs revisiting; the underlying signals still show
> value" — not "the P/S/F idea failed."**
>
> **Claim 3 — and Tier 1 can only support Claim 1 at family granularity (§14).** Tier 1 is valid for
> "**uses Setting-C information at all**" (`F`, `D`, `D'`) versus "**does not**" (`KDA_cont`, `P`,
> `S`, `MarginB`) — that is the 0.824-vs-0.549 result, and it holds. It is **not** valid for ranking
> *within* the C-derived family, including `F` against `D`: those differences track how directly a
> quantity re-encodes the label's own definition. Choosing among C-derived candidates requires a
> validation set whose labels are not derived from Setting C.

1. **The product form is not supported.** On SciQ, `F` separates `context_dependent` from
   `prior_dependent` at **AUC 0.899**, against `D`'s 0.824, `D'`'s 0.857, and `KDA_cont`'s **0.549**;
   multiplying `F` by `S` and by `(1 − P)` makes the score worse on every split tested.
   ⚠️ **But §12 and §14 remove the ability to read anything more into that ordering.** `F` decomposes
   exactly into `0.5·(MarginB + MarginC)`, where `MarginC` reproduces the Tier-1 label at **AUC
   1.0000 by construction** and the non-circular half, `MarginB`, sits at **chance** (0.454 pooled,
   0.538 within `both_correct`). So `F`'s own 0.899 is substantially definitional, and **"`F` alone
   is the metric" is NOT a supportable claim.** What survives is only that the product form scores
   lower — a fact about the combination rule, not evidence that `F` is the right answer.
2. **`S` alone runs below chance on Tier 1 (0.452 SciQ, 0.288 OBQA), but this is a composition
   effect, not an inverse relationship.** §11 shows the sign **reverses** inside `both_correct`
   (AUC 0.542, Cohen's *d* = **+0.165**): `prior_dependent` items are simply concentrated in the
   high-`S` bucket (22.8% of `both_correct`'s classified items vs 8.0% of `wrong_to_correct`'s).
   The original wording of this point overstated it and has been corrected.
3. **The `(1 − P)` gate reproduces the RQ1 saturation failure.** On the encoder ensemble it costs a
   uniform ~41% of the score with little discriminative return; on saturated Qwen3-4B it annihilates
   it — mean `D` **0.0356** with **88.1% of items exactly zero**, against `D'` **0.3906**, and Tier-1
   AUC collapses from **0.997** (`D'`) to **0.600** (`D`). See §5.
4. **`D` still comfortably beats `KDA_cont`** at the job `KDA_cont` was supposed to do — this is
   Claim 1 above, and it is the part of the result that survives every caveat in §9 and §12.
5. **OpenBookQA remains unusable** for this family. 29.0% eligible, two thirds of that on the
   degraded `partial` tier, and `KDA_cont`'s discrimination there is *inverted* (0.355). Reported,
   not headlined.
6. **The `reasoning_shortcut` "miss" in §8.1 was a spec error, not a metric failure.** §13 reads the
   eight highest-`F` items: the taxonomy labels are sound, and they label *item-level
   exploitability* while `F` measures *solver-level behaviour*. An item can be shortcut-solvable and
   still have the solver read the passage. Phase-1 row 9 conflated the two.

---

## 1. The formulation

Per solver `k` and question `q`, with `g` the gold option index and `c` the counterfactual target:

$$P = p(\text{gold} \mid A) \qquad S = p(\text{gold} \mid B)$$

$$F = \tfrac{1}{2}\Big[\underbrace{\big(p(\text{gold}|B) - p(\text{gold}|C)\big)}_{\text{term}_1:\ \text{mass leaves gold}} + \underbrace{\big(p(c|C) - p(c|B)\big)}_{\text{term}_2:\ \text{mass arrives on the cf target}}\Big]$$

Two **named variants**, reported side by side everywhere, neither treated as the other's fallback:

$$D = (1-P)\cdot S \cdot \max(F,0) \qquad\qquad D' = S \cdot \max(F,0)$$

`D` is the formulation as specified. `D'` drops the prior gate. §5 is the comparison.

### 1.1 Field mapping (verified, 0 missing records across all four runs)

| Symbol | Source |
|---|---|
| `P` | `per_model[k].probabilities_no_passage[g]` |
| `S` | `per_model[k].probabilities_original_passage[g]` |
| `p(gold\|C)` | `per_model[k].probabilities_counterfactual_passage[g]` |
| `p(cf\|C)` | `per_model[k].probabilities_counterfactual_passage[c]` |
| **`p(cf\|B)`** | `per_model[k].probabilities_original_passage[c]` — **the one quantity no run stores as a scalar** |

The Qwen3-4B run uses a flat layout (`probabilities_*` on the record itself, |M| = 1); `psfd.solver_views`
normalises both.

### 1.2 Faithfulness checks (all passed; the scripts exit non-zero otherwise)

- Recomputed `P`, `S`, `p(gold|C)`, `p(cf|C)` agree with the scalars each run already stored, to
  within 1e-9, on every (solver, question) pair.
- `n_eligible` matches each file's own `counterfactual_generation.n_eligible`: **860/884** SciQ,
  **145/500** OBQA `partial`, **42/500** OBQA `exact`, **860/884** Qwen SciQ.
- `counterfactual_target_idx == answer_idx` occurs **0** times on the eligible set.
- Independently recomputed means reproduce the published README §6.3 figures exactly:
  `KDA_cont` **0.4767**, hard **0.3357**, soft **0.2641**.

### 1.3 Aggregation

Per-solver `D_k` is computed first and **then** averaged — never the reverse. Pooling probabilities
first destroys the prior signal: the pooled ensemble `P` on SciQ spans only **[0.100, 0.662]**
(median 0.316, **zero** items above 0.9), because `kda-albert-xlarge-v2-race` is near-uniform on
**607/860** items and drags the mean to ≈0.33.

Each item carries mean, SD, min, max and a bootstrap interval over solvers. **The interval is over
four members and must not be read as a population CI** — it is reported because the Setting-C label
agrees across solvers at only Cohen's **κ = 0.046**, so between-solver spread is a result, not noise.

---

## 2. Phase 1 — the acceptance spec (what a correct metric must do)

The table each result in §3–§5 is judged against. Counts are ensemble-level, SciQ `test`.

### 2.1 Setting-C classes and the four contingency buckets

| # | Failure mode | What `KDA_cont` does | Required P/S/F/D behaviour | Outcome |
|---|---|---|---|---|
| 1 | `both_correct` ∧ `prior_dependent` (85 of 381 ensemble) | Blind spot: down-weights via `(1−P)`, still credits `S` | `F ≤ 0` → `D = 0`. The `max(F,0)` clamp is the mechanism | ✅ AUC 0.772 within the bucket (§4.2) |
| 2 | `both_correct` ∧ `context_dependent` (287) | Credits it, but by correlation not causation | High `S`, `F > 0`, top of range | ✅ |
| 3 | `both_correct` ∧ `unstable_other` (76 ensemble; **677 pairs**) | Credits `S` in full | `F` should not credit it (strict convention) | ❌ **Diverges — see §6** |
| 4 | `wrong_to_correct` (388) — KDA's only visible bucket | Credits fully | Should be *split* by `F` | ✅ D 0.197 vs 0.123 on `both_correct` |
| 5 | `both_wrong` (82) | Near-zero, for the wrong reason (solver weakness) | `D ≈ 0` but flagged as uninformative | ✅ D 0.060; stratified, never pooled |
| 6 | `correct_to_wrong` (9) — context harm | **Silently dropped** | Visible via `S < P` | ⚠️ D 0.020, but indistinguishable from #5 by `D` alone |
| 7 | `common_knowledge` (179) | Weight shrinks, item still scores | `(1−P)` suppresses regardless of `S` | ⚠️ Suppresses, but weakly (§4.3) |
| 8 | `parametric_knowledge` (48) | Same | Same as #7 | ⚠️ Same |
| 9 | `reasoning_shortcut` (42) | Fully credited; `P` can be low | `(1−P)` cannot fire → **`F` must** | ⚠️ Barely below control |
| 10 | `option_leakage` (45) | Fully credited (anchor #587 has the split's **max** `KDA_cont` 0.8419) | Same as #9 | ✅ Lowest `F` of the flagged categories |
| 11 | `material_not_necessary` (8) | Credited when A happens to fail | High `S`, `F ≈ 0` | ✅ Lowest `D` of all categories |
| 12 | **`KDA_cont` carries no prior/context information** (retained-mean exclusion inert at 100.5%; AUC 0.553) | — | D must clear a bar `KDA_cont` fails | ✅ **0.824 vs 0.549** |
| 13 | **Saturation collapse of the ignorance weight** | The RQ1 finding | D must not reproduce it | ❌ **It does — §5** |
| 14 | Setting-C label is solver-dependent (κ = 0.046) | — | Ensemble mean + spread mandatory | ✅ Implemented; spread 0.176 SciQ, 0.272 OBQA |
| 15 | Ensemble `P` never saturates (max 0.662) | — | Aggregate `D_k`, not `p` | ✅ Implemented |
| 16 | Corpus defect: 12 SciQ ids, all Setting-C ineligible | — | Already excluded by the 860 filter | ✅ Verified, no extra filter |
| 17 | CF-ineligibility as silent selection (2.7% SciQ vs **71.0%** OBQA) | — | Report `n_eligible/n` beside every figure | ✅ §7 |

---

## 3. Phase 3 Tier 1 — Setting-C discrimination

**AUC separating `context_dependent` (positive) from `prior_dependent` (negative), ensemble level.**

> ⚠️ **Partly definitional.** `F` and the Setting-C class are both functions of Setting C. This is a
> *calibration* check — does the continuous score track the hard label — not independent validation.
> Tier 2 supplies the independence. The effect is extreme on the saturated single solver, where the
> hard label is nearly a deterministic function of `F`'s sign (AUC 0.999).

| Signal | SciQ (669 / 115) | OBQA `partial` (96 / 26) | OBQA `exact` (24 / 10) | Qwen3-4B (334 / 518) |
|---|---:|---:|---:|---:|
| **`F` alone** | **0.8987** | **0.7568** | **0.8625** | **0.9989** |
| `D'` = `S·F` | 0.8574 | 0.6354 | 0.8083 | 0.9970 |
| **`D`** = `(1−P)·S·F` | 0.8241 | 0.6550 | 0.7833 | 0.5996 |
| `(1 − P)` alone | 0.7095 | 0.7548 | 0.7958 | 0.5958 |
| `S` alone | **0.4524** | **0.2877** | 0.3958 | 0.4944 |
| `KDA_cont` | 0.5494 | **0.3550** | 0.5792 | n/a (\|M\|=1) |

Three things to read off this table:

- **`S` is anti-predictive.** Below 0.5 on three of four splits, and 0.288 on OBQA. High Setting-B
  confidence is weak evidence *against* context-dependence. Every point of `S` in the product is
  working against `F`.
- **`KDA_cont` is at chance on SciQ (0.549) and inverted on OBQA (0.355)**, closely tracking the
  committed exclusion-estimator figures (0.553 / 0.346). Those are a *different* comparison —
  Q_ctx-kept vs Q_ctx-dropped, not context- vs prior-dependent — so this is corroboration from a
  neighbouring contrast, not a re-derivation of the same number.
- **Ordering is stable**: `F` > `D'` > `D` on every split but OBQA `partial`, where `D` (0.655)
  edges `D'` (0.635) because `(1−P)` happens to be the strongest single signal there (0.755).

Excluding the near-uniform albert member changes essentially nothing (SciQ: `D` 0.8244, `D'` 0.8582,
`F` 0.8988) — the pooled-probability compression, not albert's presence in the *score* mean, was the
issue.

### 3.1 Per-solver `D` AUC — the κ = 0.046 problem, at the score level

| Solver | SciQ | OBQA `partial` | OBQA `exact` |
|---|---:|---:|---:|
| `Riiid/kda-mpnet-base-race` | 0.9459 | 0.9181 | 0.9250 |
| `Riiid/kda-scibert-uncased-race` | 0.7831 | 0.7435 | 0.6590 |
| `google/t5-small-ssm-nq` | 0.7710 | 0.6462 | 0.7980 |
| `Riiid/kda-albert-xlarge-v2-race` | 0.7695 | 0.7979 | 0.7857 |
| **Spread** | **0.176** | **0.272** ⚠️ | **0.266** ⚠️ |

SciQ's spread stays inside the pre-registered 0.2 tolerance; both OBQA runs breach it. On OBQA the
score is more a property of the solver than of the question — the same conclusion the κ = 0.046
result reached about the hard label.

---

## 4. Phase 3 Tier 2 — structure from Settings A and B alone

Independent of how `F` is defined.

### 4.1 The four contingency buckets (SciQ, ensemble)

| Bucket | n | mean `D` | mean `D'` | mean `KDA_cont` |
|---|---:|---:|---:|---:|
| `wrong_to_correct` | 388 | **0.1972** | 0.2746 | 0.5054 |
| `both_correct` | 381 | 0.1234 | 0.2726 | 0.4985 |
| `both_wrong` | 82 | 0.0599 | 0.0876 | 0.2612 |
| `correct_to_wrong` | 9 | 0.0199 | 0.0333 | 0.2768 |

`KDA_cont` puts `wrong_to_correct` and `both_correct` within 0.007 of each other; `D` separates them
by 0.074 (AUC 0.714). Note the separation is carried almost entirely by `(1−P)` (AUC 0.986, by
construction — the buckets are defined on Setting-A correctness), and `D'` is at chance (0.494) here.
**This criterion is largely definitional too**, and is reported for completeness rather than as
evidence.

### 4.2 Inside `both_correct` — the bucket `KDA_cont` is blind to

The core value proposition: 381 items correct in both A and B, where the ignorance weight is small
and baseline KDA credits the item anyway.

| Signal | AUC (287 context-dep vs 85 prior-dep) |
|---|---:|
| `F` | **0.9121** |
| `D'` | 0.8892 |
| **`D`** | **0.7723** ✅ (criterion 2 bar: 0.70) |
| `(1 − P)` | 0.6250 |
| `KDA_cont` | 0.6169 |
| `S` | 0.5424 |

`D` clears the bar. `F` clears it by a much wider margin.

### 4.3 Rank agreement and scale

Spearman ρ against `KDA_cont` on 860 items: **`D` 0.711**, `D'` 0.653, `F` 0.605, `S` 0.844. `D` is
not a monotone repackaging of `KDA_cont` (criterion 3 bar: < 0.90) ✅ — but note `S`, the component
that contributes nothing to discrimination, is the one most correlated with `KDA_cont`. That is
consistent: `KDA_cont` is a weighted average of `S`.

Mean values on the 860: `KDA_cont` 0.4767, hard 0.3357, soft 0.2641, **`D` 0.1495, `D'` 0.2534**.
`D` and `KDA_cont` are **not on a common scale** and the ratio is not a retention figure comparable
to hard/soft.

---

## 5. Q2 — the `D` vs `D'` comparison

### 5.1 What the `(1 − P)` gate costs

| Split | mean `D` | mean `D'` | gate attenuation `D/D'` | items with `D = 0` | items with `(1−P) < 0.01` |
|---|---:|---:|---:|---:|---:|
| SciQ, 4 encoders | 0.1495 | 0.2534 | 0.590 | 0.1% | — |
| SciQ, ex-albert | 0.1957 | 0.3330 | 0.588 | — | — |
| OBQA `partial` | 0.0494 | 0.0877 | 0.564 | 0.7% | — |
| OBQA `exact` | 0.0904 | 0.1454 | 0.622 | 2.4% | — |
| **Qwen3-4B, \|M\|=1** | **0.0356** | **0.3906** | **0.091** | **88.1%** | **94.2%** |

Per-solver on SciQ, the gate's bite tracks exactly how much the solver already knows:

| Solver | mean `D` | mean `D'` | mean `(1−P)` | frac `(1−P) < 0.01` |
|---|---:|---:|---:|---:|
| `kda-mpnet-base-race` | 0.3746 | 0.6570 | 0.590 | 0.1% |
| `t5-small-ssm-nq` | 0.1342 | 0.2299 | 0.616 | **11.2%** |
| `kda-scibert-uncased-race` | 0.0784 | 0.1123 | 0.718 | 0.0% |
| `kda-albert-xlarge-v2-race` | 0.0109 | 0.0144 | 0.749 | 0.0% |
| **Qwen3-4B** | **0.0356** | **0.3906** | **0.047** | **94.2%** |

### 5.2 Framing for the mentor update

> The literal formulation's `(1 − P)` gate reproduces, inside the proposed metric, the exact failure
> RQ1 identified in `KDA_cont`: on the saturated Qwen3-4B run **94.2% of counterfactual-eligible
> items have `(1 − P) < 0.01`**, driving **88.1% of `D` values to exactly zero** and collapsing
> Tier-1 discrimination from **AUC 0.997** (`D' = S·max(F,0)`) to **0.600** (`D`) — while on the
> weaker encoder ensemble the same gate costs a fairly uniform ~41% of the score with modest
> discriminative return (`D` 0.824 vs `D'` 0.857). The tension is real and, I think, philosophical
> rather than a bug: `D` is arguably *right* that no knowledge-dependency claim survives when the
> solver already knew the answer, but that correctness makes it **practically inert precisely in the
> modern-LLM regime RQ2 exists to serve**, where almost every item is already known. This is a
> comparison finding between two named variants, not a verdict that `D` failed — `D` still beats
> `KDA_cont` decisively (0.824 vs 0.549) and sees the `both_correct` bucket `KDA_cont` cannot. The
> formulation is not locked pending your read.

---

## 6. Q3 — the `unstable_other` divergence, quantified (no variant implemented)

Per the strict convention, all four committed estimators score `unstable_other` as earning no
credit. `F` does not implement that, because probability can leave the gold answer without arriving
on the counterfactual target. Counts over **(solver, question) pairs whose own Setting-C class is
`unstable_other`**:

| Split | pairs | `F_raw > 0` | **term1-only (`term1>0`, `term2≤0`)** | rate of pairs | rate of `F>0` |
|---|---:|---:|---:|---:|---:|
| SciQ | 677 | 514 | **48** | **7.1%** | 9.3% |
| OBQA `partial` | 196 | 124 | **27** | **13.8%** | 21.8% |
| OBQA `exact` | 58 | 40 | **7** | **12.1%** | 17.5% |
| Qwen3-4B | 8 | 7 | **0** | 0.0% | 0.0% |

For context, the same statistic outside `unstable_other` is far smaller: SciQ `context_dependent`
0.7%, `prior_dependent` 1.8%. The divergence is concentrated where the spec predicted.

> **⚠️ One number that changes the decision.** `F_strict = min(term1, term2)` would fix only the
> term1-only slice. Decomposing SciQ's 514 positive-`F` `unstable_other` pairs: **435 have *both*
> terms positive**, 48 term1-only, 31 term2-only. So `min()` would still credit **435 pairs (64.3%
> of all `unstable_other` pairs)**; OBQA `partial` **81 (41.3%)**, OBQA `exact` **32 (55.2%)**.
> `min()` is not the fix for this divergence — the divergence is that `F` is a graded quantity and
> the strict convention is a hard rule about `argmax`. Reconciling them needs a decision about which
> is authoritative, not a different clamp.

No `F_strict` variant was implemented, as instructed. The numbers are above for your call.

---

## 7. Phase 4 — OpenBookQA counterfactual adequacy

Quantified from the committed artifacts; no dry run was needed, since
`summary.counterfactual_generation` and the per-item `substitution_tier` already carry it.

| Metric | SciQ | OBQA `partial` | OBQA `exact` |
|---|---:|---:|---:|
| **Eligible** | **860/884 = 97.3%** | **145/500 = 29.0%** | **42/500 = 8.4%** |
| `exact` tier | 813 (92.0%) | 42 (8.4%) | 42 |
| `partial` tier | 36 (4.1%) | 97 (19.4%) | 0 |
| `none` (ineligible) | 24 (2.7%) | **355 (71.0%)** | 458 |
| **`partial` as share of eligible** | **4.2%** | **66.9%** | 0% |
| Mean substitutions / item | 1.894 | 1.048 | 1.024 |
| Glued residual among eligible | 16 | 20 (13.8%) | 0 |
| Passage length, median words | **58** | **8** | 8 |

Fidelity on the `partial` tier ([`counterfactual_obqa_analysis.md`](../ex2_counterfactual/counterfactual_obqa_analysis.md) §1.3):
coverage = 1.0 on **1.0%** of items, coverage < 0.5 on **79.4%**, inserted target differs from the
matched span by > 1 token on **81.4%**. The single `glued` match is a documented false positive.

**Verdict: inadequate, and this experiment confirms it downstream.** OBQA `partial` is the only split
where `KDA_cont`'s discrimination is *inverted* (0.355), where `S` is most strongly anti-predictive
(0.288), and where the per-solver `D` AUC spread breaches tolerance (0.272). **P/S/F/D must not be
reported as an OpenBookQA headline.** The 42-item `exact` subset behaves like a small SciQ (`F` 0.863)
and is the only OBQA figure worth quoting, with its n.

### 7.1 Alternative constructions — proposals only, nothing implemented

Building on the three already set out in `counterfactual_obqa_analysis.md` §6; ranked, with the
trade-off note this pass adds.

| Strategy | Fidelity | Cost | Leakage risk | Note |
|---|---|---|---|---|
| **(1) Antonym / scalar swap on the rule's head term**, controlled lexicon (`increase↔decrease`, `hot↔cold`) | High — preserves clause shape exactly, grammatical by construction | Low; deterministic, offline, no GPU | **Low** — no distractor string is spliced in, so no surface cue for the solver to exploit | ⚠️ Needs an **answer-remapping step**: under the inverted rule the correct answer is not automatically one of the four options, and for some items no option is consistent. `n_remappable` must be reported as a second eligibility rate. A lexicon-match coverage estimate over the 500 `fact1` strings is cheap and is the right next step |
| **(2) LLM-generated single-clause counterfactual rule**, gated by round-trip entailment plus "some option is correct under the perturbed rule" | Highest coverage | GPU/API, plus its own validation pass | **Highest** — a generator conditioned on the option list can echo distractor phrasing, manufacturing exactly the cue Setting C must not contain. Mitigate by generating blind to the options and remapping separately | Cannot be trusted without a κ-validated human check on a sample. This repo's own Task A result — the LLM judge over-credits **13:1** against a human — is direct local evidence that an unvalidated LLM in the loop biases predictably |

Rule-level **negation** (that doc's option 1) ranks below (1): same remapping problem, worse control
over whether the negated rule remains meaningful.

---

## 8. The pre-registered verdict table

Five criteria, fixed in the RQ2 planning document before any of these numbers existed. Criterion 5's
margin (0.02 AUC) was the one free parameter and was fixed in `evaluate_psfd.py` at implementation
time, before first run. Adjudicated on `D`, the specified formulation, primary split **SciQ**.

| # | Criterion | SciQ | OBQA `partial` | OBQA `exact` | Qwen3-4B |
|---|---|:--:|:--:|:--:|:--:|
| 1 | Tier-1 AUC ≥ 0.75 **and** > `KDA_cont`'s | ✅ 0.824 > 0.549 | ❌ 0.655 (> 0.355) | ✅ 0.783 > 0.579 | ❌ 0.600 (no `KDA_cont`) |
| 2 | Within-`both_correct` AUC ≥ 0.70 | ✅ 0.772 (n=381) | ✅ 0.808 (n=41) | ✅ 0.804 (n=16) | ❌ **0.565** (n=819) |
| 3 | Spearman(`D`, `KDA_cont`) < 0.90 | ✅ 0.711 | ✅ 0.772 | ✅ 0.851 | — n/a |
| 4 | Mean `D` below control on ≥ 3 of 5 categories *(corroboration only)* | ✅ **5/5** | ❌ 2/4 populated | ✅ 3/4 populated | ✅ 5/5 |
| 5 | `D` beats its best single component by ≥ 0.02 AUC | ❌ **−0.075** (`F` 0.899) | ❌ −0.102 (`F` 0.757) | ❌ −0.079 (`F` 0.863) | ❌ −0.399 (`F` 0.999) |
| | **Passed** | **4 / 5** | 2 / 5 | 4 / 5 | 1 / 4 decided |

**Overall: NEEDS ANOTHER ITERATION — the product form is not justified; a single component beats `D`
on every split.** On the two OBQA runs the per-solver AUC spread additionally breaches 0.2, so the
κ = 0.046 solver-dependence dominates there.

> **What this verdict does and does not say.** It is a verdict on **the combination rule**, not on
> the decomposition. Criteria 1–4 — the ones that ask whether P/S/F carry information `KDA_cont`
> lacks — pass on 4 of 5 for SciQ and OBQA `exact`. Criterion 5, the one that fails everywhere, asks
> only whether *multiplying* the three signals beats using one of them. So the finding is
> **"`(1−P)·S·max(F,0)` is the wrong way to combine three signals that individually do carry
> value"**, and the next iteration is a combination-rule question, not a return to square one.
> §12 shows the obvious next move — fitting the combination — is blocked by circularity, so the
> real next step is an independently-labelled validation set, not a better functional form.

Criterion 5 is the one that decides this, and it fails by a wide and consistent margin: −0.075 to
−0.399 AUC against a required **+0.02**.

**Criterion 2 is where the saturation cost shows up most sharply.** `D` clears the bar on all three
encoder splits and fails on Qwen3-4B — 0.565, barely above chance — on the 819-item `both_correct`
pool that is 95% of that run. `D'` scores **1.000** on the identical items. The gate does not merely
shrink `D` at LLM scale; inside the one bucket RQ2 most needs to see, it removes the signal:

| Within `both_correct` | SciQ (381) | OBQA `partial` (41) | OBQA `exact` (16) | **Qwen3-4B (819)** |
|---|---:|---:|---:|---:|
| `F` | 0.9121 | 0.9706 | 1.0000 | **1.0000** |
| `D'` | 0.8892 | 0.9599 | 1.0000 | **1.0000** |
| **`D`** | 0.7723 | 0.8075 | 0.8036 | **0.5647** |
| `(1 − P)` | 0.6250 | 0.5267 | 0.6964 | 0.5630 |
| `KDA_cont` | 0.6169 | 0.6123 | 0.6429 | n/a |

(The 1.000s on the two smallest and the saturated splits are the circularity of §3 at its most
extreme, not a real perfect separation.)

### 8.1 Tier 3 detail — corroboration only

> **⚠️ Not a validation gate.** The five failure-category labels are single-annotator, single-pass,
> with no rubric written in advance and **no measured inter-annotator agreement (no κ)**; each item
> carries exactly one forced label ([`failure_taxonomy_methodology.md`](../ex5_failure_audit/failure_taxonomy_methodology.md) §0).
> The flagged pool contains **only** failure cases, so `unflagged_eligible` is a remainder, **not a
> verified-clean control**. Read the ordering, never the values.

SciQ, 322 of 328 flagged items joined (6 are Setting-C ineligible):

| Category | n | mean `D` | mean `D'` | mean `(1−P)` | mean `F` | mean `KDA_cont` | below control? |
|---|---:|---:|---:|---:|---:|---:|:--:|
| `common_knowledge` | 179 | 0.1134 | 0.2665 | 0.575 | 0.314 | 0.4883 | ✅ |
| `parametric_knowledge` | 48 | 0.1249 | 0.2430 | 0.615 | 0.292 | 0.5189 | ✅ |
| `reasoning_shortcut` | 42 | 0.1161 | 0.2709 | 0.570 | 0.318 | 0.4993 | ✅ |
| `option_leakage` | 45 | 0.1003 | 0.2251 | 0.578 | 0.264 | 0.4969 | ✅ |
| `material_not_necessary` | 8 | 0.0952 | 0.2084 | 0.567 | 0.251 | 0.4915 | ✅ |
| *`unflagged_eligible`* | *538* | *0.1713* | *0.2516* | *0.721* | *0.313* | *0.4654* | — |

All five categories sit below the control on `D` — but the mechanism is **not** the one the spec
predicted. `(1−P)` on the flagged categories is 0.57–0.62 against the control's 0.721, so the
suppression is coming almost entirely from the prior gate, and it fires *uniformly* rather than
selectively: `common_knowledge` and `parametric_knowledge` are not suppressed any harder than
`reasoning_shortcut`. On `D'` (gate removed) only **3 of 5** stay below control, and
`reasoning_shortcut` rises *above* it. Meanwhile `KDA_cont` is **higher** on every flagged category
than on the control — the taxonomy's own restatement of finding #12.

The two categories the spec said `F` alone would have to catch — `reasoning_shortcut` and
`option_leakage` — split: `option_leakage` has the second-lowest `F` (0.264) ✅, `reasoning_shortcut`
has the *highest* `F` of any category (0.318, above the control's 0.313) ❌.

> **⚠️ Superseded by §13.** Reading the eight highest-`F` `reasoning_shortcut` items shows the
> taxonomy labels are correct and `F` is behaving correctly; the ❌ above is an error in the
> Phase-1 acceptance spec, which assumed a shortcut-exploitable *item* implies a shortcut-using
> *solver*. Row 9 of §2.1 should be read as withdrawn pending a restatement.

OBQA (49 of 100 flagged items joined; the other 51 are Setting-C ineligible) reaches only 2 of 4
populated categories below control, and is too small and too degraded to interpret.

---

## 9. Limitations

1. **Tier 1 is partly definitional.** `F` and the Setting-C label are both functions of Setting C.
   The AUCs are calibration, not independent validation, and this is most extreme on Qwen (0.999).
2. **Tier 2's bucket separation is also partly definitional** — the buckets are defined on Setting-A
   correctness, which is `P`.
3. **Tier 3 has no κ and no clean class.** See the box in §8.1.
4. **The bootstrap CI is over four solvers.** Not a population interval.
5. **`D` and `KDA_cont` are not on a common scale.** No ratio between them is a retention figure.
6. **OBQA results are reported, not relied on.** 29.0% eligible, 66.9% of that degraded.
7. **The Qwen run is |M| = 1**, so it has no ensemble and no `KDA_cont`. It is a saturation
   demonstration only. A non-degenerate |M| ≥ 2 LLM reading needs the Qwen2.5-7B run
   (`RUN_QWEN2.5.md`), which is out of scope here.
8. **Nothing here is adopted.** The ex2 estimators are untouched and remain the committed method.

---

## 10. Scope only — deferred, not attempted

- **OBQA antonym/scalar counterfactual builder** (§7.1 strategy 1). The lexicon-coverage estimate is
  cheap; the builder plus answer-remapping is not.
- **LLM-assisted counterfactual generation** (§7.1 strategy 2) and any validation pass it needs.
- **Any new model run**, including a |M| ≥ 2 LLM ensemble for a non-degenerate LLM-scale reading.
- **`F_strict = min(term1, term2)`** — quantified in §6, deliberately not implemented.
- **A small double-annotated subset for κ on the five-category failure taxonomy.** This is the real
  fix for Tier 3's weakness and the highest-value human task the evaluation exposes
  (`failure_taxonomy_methodology.md` §7 step 3 already specifies it, with the
  `common_knowledge` ↔ `parametric_knowledge` boundary as the pre-registered concern). Noted as a
  possible follow-up; **no annotation was started.**
- **All RQ3 / downstream-filtering work**, now reserved as `ex7`.

---

---

# Addendum — three follow-up diagnostics (2026-09-14)

Analysis only. No new formulation, no change to `psfd.py`'s compute logic, no new inference.
Script: [`code/ex6_psfd_score/diagnose_psfd.py`](../../code/ex6_psfd_score/diagnose_psfd.py) ·
Output: `results/ex6_psfd_score/psfd_diagnostics.json`

**Two of the three change how the headline should be worded.** §12 and §13 are flagged accordingly.

---

## 11. Why is `S` anti-predictive?

**Hypothesis under test:** `S` is near-saturated in both Setting-C classes by construction, so the
below-chance AUC is noise rather than a real inverse relationship.

**First, a premise correction.** The hypothesis assumed both classes require `both_correct`. They do
not: `classify_counterfactual` (`run_counterfactual_experiment.py:145-153`) keys off the Setting-C
argmax alone, so the class is assigned to **every** counterfactual-eligible item — 860 on SciQ, of
which only 381 are `both_correct`. Tier 1 therefore runs over the full eligible set. The ceiling
argument is tested both ways below.

### 11.1 All eligible items

| Split | context `S` mean (sd) | prior `S` mean (sd) | AUC | Cohen's *d* | frac `S` > 0.9 |
|---|---|---|---:|---:|---|
| SciQ | 0.5478 (0.1305), n=669 | 0.5736 (0.1162), n=115 | 0.4524 | **−0.201** | ctx 0.000 / pri 0.000 |
| OBQA `partial` | 0.3422 (0.1490), n=96 | 0.4458 (0.1204), n=26 | 0.2877 | **−0.722** | 0.000 / 0.000 |
| OBQA `exact` | 0.4496 (0.1370), n=24 | 0.4953 (0.0955), n=10 | 0.3958 | **−0.361** | 0.000 / 0.000 |
| **Qwen3-4B** | **0.9970 (0.0547)**, n=334 | **0.9992 (0.0176)**, n=518 | 0.4944 | −0.060 | **0.997 / 0.998** |

### 11.2 Restricted to `both_correct` — where the ceiling argument would actually bite

| Split | context `S` mean (sd) | prior `S` mean (sd) | AUC | Cohen's *d* |
|---|---|---|---:|---:|
| SciQ | 0.6114 (0.0962), n=287 | 0.5949 (0.1129), n=85 | 0.5424 | **+0.165** |
| OBQA `partial` | 0.5103 (0.0817), n=22 | 0.4866 (0.0957), n=17 | 0.5882 | **+0.269** |
| OBQA `exact` | 0.5397 (0.0967), n=7 | 0.5098 (0.0875), n=8 | 0.6071 | **+0.326** |
| **Qwen3-4B** | **1.0000 (0.0000)**, n=304 | **1.0000 (0.0000)**, n=512 | 0.4944 | −0.100 |

### 11.3 Which case is it? — **both, on different splits**

- **Qwen3-4B: the noise case. Hypothesis confirmed.** `S` is exactly at ceiling — 99.7% / 99.8% of
  items above 0.9 overall, and **sd = 0.0000 within `both_correct`** where every value is 1.0.
  AUC 0.494 is a coin flip on a constant. Nothing can be read from `S` on this run.
- **The three encoder splits: NOT the ceiling case.** Pooled ensemble `S` never exceeds 0.9 on a
  single item (frac > 0.9 = 0.000 everywhere, because the near-uniform albert member caps the pooled
  mean), and sd is 0.10–0.15. There is genuine variance.
- **But it is also not a real inverse relationship.** The sign **flips** under stratification:
  negative over all eligible items (*d* = −0.20 to −0.72), positive inside `both_correct`
  (*d* = +0.17 to +0.33). This is a composition effect — a small Simpson's paradox. `prior_dependent`
  items are concentrated in the high-`S` bucket:

  | SciQ bucket (860 eligible) | mean `S` | context-dep | prior-dep | prior share of the two |
  |---|---:|---:|---:|---:|
  | `both_correct` | **0.6045** | 287 | 85 | **22.8%** |
  | `wrong_to_correct` | 0.5245 | 332 | 29 | 8.0% |
  | `both_wrong` | 0.2874 | 47 | 0 | 0.0% |
  | `correct_to_wrong` | 0.2706 | 3 | 1 | 25.0% |

  `prior_dependent` is nearly three times as concentrated in `both_correct`, which is also the
  highest-`S` bucket. Pooling the buckets makes high `S` look like evidence of prior-dependence when
  within every bucket it is weak evidence of the opposite.

**Wording change required.** §0 point 2 previously read *"`S` is the culprit, and it is
anti-predictive … a solver being confident with the gold passage is weak counter-evidence that it
was reading the passage."* That is **not supported** and has been corrected in place. `S` is
uninformative-to-weakly-positive within stratum, and its below-chance pooled AUC is an artifact of
bucket composition. The conclusion that `S` contributes nothing useful to the product stands; the
stated mechanism was wrong.

---

## 12. Ceiling check — is any combination of P, S, F better than `F` alone?

Logistic regression on (P, S, F) predicting `context_dependent` vs `prior_dependent`, features
standardised, IRLS with a 1e-4 ridge. **Not a metric candidate** — it is fitted on the labels it is
scored against.

| Split | logistic (in-sample) | logistic (5-fold CV) | `F` alone | `D` | `D'` | `S` alone | `P` alone |
|---|---:|---:|---:|---:|---:|---:|---:|
| SciQ (n=784) | 0.9903 | **0.9894** | 0.8987 | 0.8241 | 0.8574 | 0.4524 | 0.2905 |
| OBQA `partial` (n=122) | 0.9531 | **0.9271** | 0.7568 | 0.6550 | 0.6354 | 0.2877 | 0.2452 |
| OBQA `exact` (n=34) | 1.0000 | **1.0000** | 0.8625 | 0.7833 | 0.8083 | 0.3958 | 0.2042 |
| Qwen3-4B (n=852) | 1.0000 | **1.0000** | 0.9989 | 0.5996 | 0.9970 | 0.4944 | 0.4042 |

Fitted coefficients on standardised features:

| Split | `P` | `S` | `F` |
|---|---:|---:|---:|
| SciQ | −0.474 | **−6.218** | **+10.338** |
| OBQA `partial` | −0.651 | **−3.574** | +4.156 |
| OBQA `exact` | +6.296 | **−32.196** | +37.404 |
| Qwen3-4B | −4.534 | **−3.075** | +90.603 |

`S` takes a **large negative** weight on every split, consistent with §11's finding that it adds
nothing and with the product form's failure — the product multiplies by `S` where a fitted
combination subtracts it.

### 12.1 ⚠️ The gain is an algebraic artifact, and this is the important part

CV AUC 0.989 against `F`'s 0.899 looks like headroom. It is not. Expanding the definition of `F`:

$$F_{\text{raw}} = \tfrac{1}{2}\big[(S - p(\text{gold}|C)) + (p(c|C) - p(c|B))\big]
\;\Longrightarrow\;
2F_{\text{raw}} - S + p(c|B) \;=\; p(c|C) - p(\text{gold}|C)$$

**Verified: the identity holds to 1e-9 on all 784 SciQ items.** And among items classed
`context_dependent` or `prior_dependent`, the label *is* the sign of that margin — the two classes
are exactly `argmax(C) = c` and `argmax(C) = gold`, so `p(c|C) − p(gold|C)` separates them at
**AUC 1.0000** by construction.

So a linear model given `S` and `F` is rotating toward a known identity. The only reason it stops at
0.989 rather than 1.000 is that it is not given the fourth term, `p(c|B)`. **This is not evidence
that a better combination of the three signals exists** — it is a demonstration that Tier 1 is far
more definitional than §3 conceded.

**Wording change required.** §3's warning called Tier 1 *"partly definitional"*. For any combination
that uses both `S` and `F` it is **almost entirely** definitional, and the same caution now attaches
to the comparison in §0 point 1: "`F` alone captures essentially all the separable signal" cannot be
concluded from Tier 1, because Tier 1 cannot distinguish a good metric from an algebraic
re-derivation of its own label. The honest statement is narrower: **among the three raw signals, `F`
is the strongest on Tier 1, and the product form is worse than `F` — but Tier 1 cannot adjudicate
what the best combination is.** Establishing that needs labels not derived from Setting C.

---

## 13. The `reasoning_shortcut` "miss", read item by item

The eight SciQ `reasoning_shortcut` items (of 42 eligible) with the highest ensemble `F`. All eight
are `substitution_tier = exact` and all eight are ensemble `context_dependent`.

| # | `F` | `D` | `S` | `P` | bucket | Question → gold / cf-target | Shortcut cue visible in the stem? |
|---|---:|---:|---:|---:|---|---|---|
| 529 | 0.590 | 0.118 | 0.787 | 0.653 | both_correct | *What crucial role does beneficial fungi play?* → `balance of ecosystems` / `cleaning the soil` | ✗ no stem↔option echo; label looks weakest of the eight |
| 323 | 0.535 | 0.374 | 0.714 | 0.264 | wrong_to_correct | *Water seeps through **permeable** material and stops when it reaches what?* → `impermeable rock` / `Bed Rock` | ✓ "permeable" → "impermeable" |
| 779 | 0.514 | 0.335 | 0.737 | 0.274 | wrong_to_correct | *…using stained **gel**, can separate dna fragments…* → `gel electrophoresis` / `microwave electrophoresis` | ✓ "gel" — only option containing it |
| 753 | 0.512 | 0.101 | 0.626 | 0.533 | both_correct | *atoms tend to have **eight** electrons…* → `octet rule` / `coupling rule` | ✓ "eight" → "octet" |
| 617 | 0.491 | 0.195 | 0.642 | 0.427 | both_correct | *carbohydrate formed by **two mono**saccharides* → `disaccharide` / `Nitrate` | ✓ "two"→"di", "saccharide" echo |
| 485 | 0.488 | 0.141 | 0.586 | 0.457 | both_correct | *cold front … **cold air mass** runs into what?* → `warm air mass` / `dry air mass` | ~ odd-one-out among air masses |
| 865 | 0.472 | 0.079 | 0.541 | 0.555 | both_correct | *removing **phosphorylated** amino acids* → `phosphatase` / `sucrose` | ✓ "phosphorylated" → "phosphatase" |
| 831 | 0.451 | 0.127 | 0.555 | 0.387 | both_correct | *what kind of muscle is the **heart** composed of?* → `cardiac muscle` / `idealized muscle` | ✓ "heart" → "cardiac" |

Two worked examples of the solver behaviour, all four solvers:

**#779** — B: *"**Gel** electrophoresis is an analytical technique used to separate DNA fragments…"*
C: *"**Microwave** electrophoresis is an analytical technique used to separate DNA fragments…"*

| Solver | pred B | pred C | class |
|---|---|---|---|
| `kda-mpnet-base-race` | gel electrophoresis | **microwave electrophoresis** | context_dependent |
| `kda-scibert-uncased-race` | gel electrophoresis | **microwave electrophoresis** | context_dependent |
| `t5-small-ssm-nq` | gel electrophoresis | **microwave electrophoresis** | context_dependent |
| `kda-albert-xlarge-v2-race` | gel electrophoresis | gel electrophoresis | prior_dependent |

**#831** — B: *"…the **cardiac muscle** needs to contract in an organized way."*
C: *"…the **idealized muscle** needs to contract in an organized way."* All four solvers move
B `cardiac muscle` → C `idealized muscle`.

### 13.1 What this shows

**The taxonomy labels are sound.** Seven of eight carry a clear stem↔option lexical or morphological
echo, exactly the reconstructed `reasoning_shortcut` criterion
([`failure_taxonomy_methodology.md`](../ex5_failure_audit/failure_taxonomy_methodology.md) §4.1). Only
#529 looks doubtful, and #485 is borderline against `common_knowledge`. This is a hand-check of eight
items by one reader and carries every caveat §8.1 already states.

**`F` is also behaving correctly** — and #779 is the clean demonstration. The stem contains "gel", so
the shortcut points at `gel electrophoresis`. The counterfactual passage says "Microwave
electrophoresis". Three of four solvers answer **microwave** — they *abandoned* the shortcut cue and
followed the passage. High `F` is the right reading of that.

**So the ❌ in §8.1 was an error in the acceptance spec, not in the metric.** Phase-1 row 9 asserted
that `reasoning_shortcut` items must show `F ≈ 0`. That conflates two different objects:

| | Object | Answers |
|---|---|---|
| `failure_category` | the **item** | "is this question *exploitable* by a shortcut?" |
| `F` | the **(solver, question) pair** | "did *this solver* follow the passage *this time*?" |

An item can be shortcut-solvable **and** have the solver read the passage — especially on SciQ, where
the passage is extractive and states the gold answer verbatim in 94.8% of items. There is no
contradiction, and no reason to expect low `F` on shortcut-exploitable items.

**Wording change required.** §2.1 row 9 is withdrawn as written, and the ❌ in §8.1 is annotated. The
correct expectation for a shortcut-exploitable item is a claim about *Setting A* — the solver should
answer it without any passage — which is `P`, not `F`. The data are consistent with that
restatement: these eight items carry `P` of 0.264–0.653 (mean **0.444**) against an eligible-set mean
of **0.332**, and the whole `reasoning_shortcut` category averages `P` = **0.430** against the
unflagged control's **0.279**. A properly specified test of row 9 would be a `P`-based one, and
nothing in this experiment ran it.

---

## 14. `F` splits into a definitional half and a chance-level half

A direct algebraic follow-on to §12. Regroup `F_raw`'s four terms **by setting** instead of by
option — the same four numbers, added in a different order:

$$F_{\text{raw}} = \tfrac{1}{2}\big[\underbrace{(p(\text{gold}|B) - p(\text{gold}|C))}_{\text{term}_1} + \underbrace{(p(c|C) - p(c|B))}_{\text{term}_2}\big] = \tfrac{1}{2}\big[\underbrace{(p(\text{gold}|B) - p(c|B))}_{\textbf{MarginB}} + \underbrace{(p(c|C) - p(\text{gold}|C))}_{\textbf{MarginC}}\big]$$

- **`MarginB`** is a pure Setting-B quantity. It uses **no Setting-C information at all**, so its
  Tier-1 AUC is the one number in this entire experiment that is not circular.
- **`MarginC`** is a pure Setting-C quantity, and among the two classes being separated its **sign
  *is* the label**: `context_dependent` and `prior_dependent` are exactly `argmax(C) = c` and
  `argmax(C) = gold`, so `p(c|C) > p(gold|C)` reproduces the label with no error.

**Identity verified to 1e-9 on 784/784 SciQ, 122/122 OBQA `partial`, 34/34 OBQA `exact`, 852/852
Qwen items.**

### 14.1 Tier-1 AUC of each half

| Split | **(a) `MarginC`** | **(b) `MarginB`** | (c) `F_raw` | `F` (clamped) | `MarginB` within `both_correct` |
|---|---:|---:|---:|---:|---:|
| SciQ (784) | **1.0000** | **0.4540** | 0.8923 | 0.8987 | 0.5379 |
| OBQA `partial` (122) | **1.0000** | **0.2688** | 0.7712 | 0.7568 | 0.5508 |
| OBQA `exact` (34) | **1.0000** | **0.3625** | 0.8708 | 0.8625 | 0.4821 |
| Qwen3-4B (852) | **1.0000** | **0.4550** | 0.9989 | 0.9989 | 0.4613 |

### 14.2 What this says — including where it contradicts the expectation

**(a) does dominate, and it is not merely "very high" — it is exactly 1.0000 on every split.**
`MarginC` reproduces the Tier-1 label perfectly, by construction. **`F`'s own Tier-1 AUC of 0.899 is
therefore substantially definitional**, not just combinations of P/S/F. `F` is a 50/50 blend of a
signal that reproduces the label perfectly and one that does not; it scores 0.892 rather than 1.000
*because* the other half dilutes it. **On Tier 1, a higher `F`-family AUC means "closer to
re-deriving the label", not "better metric".** That inverts how §3's table should be read within the
C-derived family.

**(b) does NOT match the expectation, and this is the sharpest negative result in the branch.**
`MarginB` was expected to be the informative non-circular quantity. It is at or **below** chance on
every split — 0.454, 0.269, 0.363, 0.455. Stratifying by bucket shows the below-chance values are the
same composition effect §11 identified (`prior_dependent` concentrated in the high-`S`
`both_correct` bucket): within `both_correct` alone, `MarginB` returns to **0.538 / 0.551 / 0.482 /
0.461** — i.e. **chance**. It is not anti-predictive; it is *uninformative*.

> **The conclusion this forces.** The only quantity in this experiment whose Tier-1 evaluation is not
> circular carries **no Tier-1 signal**. Every quantity that scores well on Tier 1 does so in
> proportion to how much Setting-C information it contains, up to `MarginC`, which contains nothing
> else and scores 1.0. **Tier 1 has essentially no power to rank members of the C-derived family**
> — it measures proximity to its own definition.

### 14.3 What Tier 1 can and cannot be used for

| Comparison | Valid on Tier 1? |
|---|---|
| C-derived family (`F`, `D`, `D'`, `MarginC`) vs non-C signals (`KDA_cont`, `P`, `S`, `MarginB`) | ✅ Yes — this is the `KDA_cont` 0.549 vs `D` 0.824 result, and it survives. Using Setting C at all beats not using it |
| **Ranking *within* the C-derived family** — `F` vs `D` vs `D'` vs a fitted combination | ❌ **No.** Differences here track how directly a quantity encodes `sign(MarginC)`, not metric quality |
| Criterion 5 as originally stated ("`D` beats its best single component") | ⚠️ The *failure* still stands as a fact about the product form, but the *margin* is not interpretable as metric quality |

Nothing in §3–§8 is withdrawn; the Tier-1 numbers are correct as computed. What changes is what may
be concluded from differences **inside** the C-derived family. The two conclusions that survive
intact are Tier 2's within-`both_correct` result (`D` 0.772 vs `KDA_cont` 0.617 — a family-vs-family
comparison) and the §5 saturation finding (a comparison of the same quantity across solvers, not a
ranking of quantities).

## Related

- [`README.md`](../../README.md) §2.4, §6.3 — the hard/soft/exclusion estimators this is compared against
- [`docs/ex2_counterfactual/counterfactual_experiment_methodology.md`](../ex2_counterfactual/counterfactual_experiment_methodology.md) — Setting C
- [`docs/ex2_counterfactual/unstable_other_convention.md`](../ex2_counterfactual/unstable_other_convention.md) — the strict convention §6 diverges from
- [`docs/ex2_counterfactual/e1_counterfactual_llm_scale.md`](../ex2_counterfactual/e1_counterfactual_llm_scale.md) — the Qwen3-4B run and κ = 0.046
- [`docs/ex5_failure_audit/failure_taxonomy_methodology.md`](../ex5_failure_audit/failure_taxonomy_methodology.md) — Tier 3's labels and their limits
