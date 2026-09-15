# How `unstable_other` is scored, and why

**Decided:** 2026-09-09.
**Decision:** the **strict** convention — a Setting-C prediction that is neither the gold
answer nor the counterfactual target earns **no credit in any of the four estimators**.
**Implemented at:** `UNSTABLE_CONVENTION` in
[`code/ex2_counterfactual/run_counterfactual_experiment.py`](../../code/ex2_counterfactual/run_counterfactual_experiment.py).
**Numbers behind it:** [`code/ex2_counterfactual/unstable_other_conventions.py`](../../code/ex2_counterfactual/unstable_other_conventions.py)
→ [`results/ex2_counterfactual/unstable_other_conventions.json`](../../results/ex2_counterfactual/unstable_other_conventions.json).

---

## 1. The problem this resolves

Setting C sorts each (model, question) pair into three classes
([`run_counterfactual_experiment.py`](../../code/ex2_counterfactual/run_counterfactual_experiment.py),
`classify_counterfactual`). Two have an obvious reading. The third does not:

| Class | Setting-C argmax | Reading |
|---|---|---|
| `context_dependent` | the counterfactual target | followed the passage — credit |
| `prior_dependent` | the gold answer | overrode the passage — no credit |
| `unstable_other` | a third option | **no obvious reading** |

Until 2026-09-09 the estimators **disagreed** about the third class, and no document said so:

| Estimator | Treated `unstable_other` as | i.e. same as |
|---|---|---|
| $KDA_{adj}^{hard}$, $KDA_{adj}^{soft}$ | gate = 0 → **no credit** | `prior_dependent` |
| $KDA_{adj}^{excl}$ | retained at full $KDA_{cont}$ → **full credit** | `context_dependent` |
| $\text{Acc}^{verified}_{wf}$ | not counted → **no credit** | `prior_dependent` |

So the same items were scored one way by three estimators and the opposite way by the
fourth.

## 2. How much was at stake

Exposure is much larger at the **(model, question) pair** level — which is what the hard
and soft gates actually operate on — than at the ensemble level the reports used to quote:

| Run | unstable pairs | unstable ensemble items |
|---|---:|---:|
| SciQ `partial` | 677/3,440 = **19.7%** | 76/860 = 8.8% |
| OBQA `partial` | 196/580 = **33.8%** | 23/145 = 15.9% |
| OBQA `exact` | 58/168 = **34.5%** | 8/42 = 19.0% |

Headline movement across the three conventions (`hard` / `soft` shown as retention vs. the
unadjusted eligible baseline):

| Run | convention | hard | soft | excl (retained) | excl (zero-filled) | Acc$^{ver}$ |
|---|---|---:|---:|---:|---:|---:|
| SciQ | **strict** | 0.3357 · 70.4% | 0.2641 · 55.4% | 0.4793 | 0.4269 | 0.7849 |
| SciQ | lenient | 0.3964 · 83.2% | 0.3800 · 79.7% | 0.4781 | 0.4308 | 0.7953 |
| SciQ | abstain | 0.4385 · 81.0% | 0.3309 · 61.2% | 0.4793 | 0.4314 | 0.7932 |
| OBQA `partial` | **strict** | 0.1275 · 41.8% | 0.1145 · 37.6% | 0.2969 | 0.2580 | 0.3793 |
| OBQA `partial` | lenient | 0.2164 · 71.0% | 0.2259 · 74.1% | 0.2964 | 0.2617 | 0.3931 |
| OBQA `partial` | abstain | 0.2101 · 61.0% | 0.1599 · 46.4% | 0.2969 | 0.2616 | 0.3846 |
| OBQA `exact` | **strict** | 0.1902 · 49.9% | 0.1734 · 45.5% | 0.3891 | 0.3057 | 0.5000 |
| OBQA `exact` | lenient | 0.2987 · 78.4% | 0.2991 · 78.5% | 0.3831 | 0.3101 | 0.5238 |
| OBQA `exact` | abstain | 0.3128 · 70.8% | 0.2422 · 54.8% | 0.3891 | 0.3132 | 0.5122 |

**The choice moves the hard retention by up to 29 percentage points** (OBQA `partial`,
41.8% → 71.0%). That is larger than several of the effects this project reports, so it is
not a footnote.

## 3. The three conventions

**strict** — credit requires positive evidence of context-following. `unstable_other`
earns nothing. *(What hard, soft and context-verified accuracy already did.)*

**lenient** — credit anything that is not prior-driven; the diagnostic question is "did the
answer come from the prior?", and scattering says it did not. *(What sample exclusion
already did.)* Soft's analogue gates on $1 - P^C(\text{gold})$ rather than
$P^C(\text{cf target})$.

**abstain** — the perturbation returned no verdict for that pair, so drop it from **both**
the numerator and the denominator.

## 4. Why strict

**1. Certification requires positive evidence.** The estimator's claim is that a question's
with-fact correctness was *earned from the passage*. `unstable_other` is evidence against
prior-dependence (the model did abandon the gold answer) but is not evidence for
context-following. "We could not tell" must not score the same as "we verified it did."

**2. `abstain` reintroduces the exact pathology this project exists to document.** RQ1's
central finding is that $KDA_{disc}$ emits a confident number from a silently shrinking
denominator — 41 of 884 questions at Qwen3-4B scale (README §6.4). Adopting `abstain` would
let the *adjusted* estimator shrink its own denominator whenever the perturbation confuses
the solver, and it would shrink most on exactly the data where the perturbation is least
trustworthy: 34.5% of OBQA `exact` pairs would vanish. An estimator built to fix
denominator collapse must not be permitted to cause it.

**3. `lenient` credits a broken perturbation as knowledge dependency.** Standing limitation
L1 ([`counterfactual_experiment_methodology.md`](counterfactual_experiment_methodology.md)
§9) establishes that ~81% of OBQA `partial`-tier perturbations are ungrammatical. On that
data "the model scattered" very often means "the passage became word salad." Crediting that
as evidence the question requires its material is indefensible.

The empirical consequence of (3) is decisive: **`lenient` erases a finding the project has
independently corroborated.** The SciQ-vs-OBQA gap in hard retention is 28.6 pp under strict
(70.4% vs 41.8%) and only 12.2 pp under lenient (83.2% vs 71.0%) — lenient more than halves
the clearest dataset-level effect in Experiment 2, by crediting OBQA for perturbations that
`counterfactual_obqa_analysis.md` §2.2 shows are incoherent.

## 5. What changed in the code

Only the **sample-exclusion** variant changed; the other three already followed strict, so
their published values are unaffected ($KDA_{adj}^{hard}$ = 0.3357 and
$KDA_{adj}^{soft}$ = 0.2641 on SciQ, unchanged).

$\mathcal{Q}_{ctx}$ now drops `both_correct` **&** (`prior_dependent` **or**
`unstable_other`), via `_fails_context_check`. On SciQ that moves the drop count from 85 to
**94** (85 prior + 9 unstable).

| SciQ | before | after |
|---|---:|---:|
| $KDA_{adj}^{excl}$, retained mean | 0.4781 (100.3%) | **0.4793 (100.5%)** |
| $KDA_{adj}^{excl}$, zero-filled | 0.4308 (90.4%) | **0.4269 (89.6%)** |
| dropped | 85 | **94** |

The pre-change condition is preserved as the `ensemble_both_correct_prior_only` entry of
the `q_ctx_sensitivity` block, so the convention change stays auditable and
`recompute_adjusted_kda.py` verifies against it.

**Exposure is now reported unconditionally** in every run's summary and log
(`unstable_other_exposure`), at both the pair and ensemble level. Because strict scores
these items as failures, how much of a result rests on that choice must never be invisible
— which is the same standard this project applies to $KDA_{disc}$'s denominator.

## 6. What this does not settle

- **Whether an `unstable_other` item is a broken perturbation or a genuine solver failure**
  is still unseparable. Under strict both are scored as failures, which is conservative for
  the metric but means OBQA's numbers are depressed by perturbation quality as well as by
  solver behaviour — `counterfactual_obqa_analysis.md` §5 limitation 4, unchanged.
- **The `both_correct` conditioning in $\mathcal{Q}_{ctx}$** is untouched here. It leaves 29
  SciQ items that are `wrong_to_correct` **&** `prior_dependent` in the retained set, some
  of which are demonstrably failed perturbations (`counterfactual_passage.py` leaves glued
  residuals such as `calledoxidants`). That is a separate open item — the perturbation-
  validity audit — not an `unstable_other` question.
