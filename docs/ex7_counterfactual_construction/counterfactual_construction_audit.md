# Experiment 7 — auditing the counterfactual construction, and what should replace it

**Written:** 2026-09-14. **Status:** audit complete; construction method designed, not built.
**Scope:** data quality of the Setting-C intervention. **Not** in scope: the ex6 §14 Tier-1
circularity finding. A perfectly constructed counterfactual would still produce labels
near-tautologically determined by `MarginC`; construction quality and label informativeness are
separate problems and nothing here resolves the second.

**No model inference, no LLM calls, no annotation.** Every number is derived from committed ex2
Setting-C runs, their dry-run previews, and ex6 score files. `code/ex2_counterfactual/` and all
ex2/ex6 artifacts are unmodified.

| | |
|---|---|
| Code | [`audit_construction_fidelity.py`](../../code/ex7_counterfactual_construction/audit_construction_fidelity.py) · [`classify_ineligibility.py`](../../code/ex7_counterfactual_construction/classify_ineligibility.py) · [`antonym_coverage_dryrun.py`](../../code/ex7_counterfactual_construction/antonym_coverage_dryrun.py) |
| Outputs | `results/ex7_counterfactual_construction/{construction_fidelity_audit,leakage_flags_*,obqa_ineligibility_causes,antonym_coverage_dryrun}.json` + `ineligibility_handcheck_sheet_BLIND.csv` |

---

## 0. Executive summary

1. **SciQ's construction is sound; the audit found positive evidence for it.** `exact` (813/884)
   has 810/810 coverage 1.0 and a surface-leakage asymmetry of **−3.4 pp** — the rewrite makes the
   counterfactual target *less* surface-guessable than the gold was. And the leakage probe does
   **not** predict which items solvers call `context_dependent` (AUC **0.547**, n+=669/n−=115),
   which is evidence *against* the hypothesis that SciQ's Setting-C labels are surface artifacts.
2. **SciQ is not uniformly clean.** Its 36 `partial` items carry the same defects as OBQA's, at
   4.2% prevalence: only 4/36 coverage 1.0, 69.4% keep unmatched gold words in the perturbed
   passage, F AUC 0.750 against `exact`'s 0.904. `morphological` (16 items across both datasets)
   is clean — verified by inspection; the coverage metric is simply invalid for that tier.
3. **OBQA's ineligibility has four causes, not one** — two of which were not in the plan.
   Stratum-weighted from a 48-item hand read (§4.3.1): **(A) rule-to-instance ~38%**,
   **(B) paraphrase ~31%**, **(C) fact-insufficient ~25%**, **(D) inference-licensed ~5%**. Only
   (B) is reachable by better matching; (A)+(D) need rewriting; **(C) is out of reach of any
   construction method and should be excluded.** Ceiling on OBQA eligibility ≈ **82%**.
4. **OBQA `partial` manufactures a surface cue: +43.0 pp.** The gold is surface-findable in
   Setting B for 55.5% of items; the counterfactual target is findable in Setting C for **98.5%**.
   Splicing a whole distractor into an 8-word fact makes the target trivially guessable. Not
   previously measured anywhere.
5. **OBQA `exact`'s residual instability remains unexplained — four causes ruled out, none
   sufficient.** The clean 42-item subset still shows **34.5%** pair-level `unstable_other` (SciQ
   `exact`: 19.5%). Ruled out: passage length, substitution fidelity, and cause (C) (**refuted** —
   16/16 items read have a determining fact). Counterfactual-target **incoherence** contributes
   but is weak: φ = 0.253 across all 42, and coherent targets still yield 30.2% instability, so
   fixing every incoherent target buys ≈ **4 pp of the 34.5%** (§4.3.2). The same weak mechanism
   operates on SciQ (φ = 0.174) and cannot explain the cross-dataset gap either (§4.3.3). ~30 pp
   is still unaccounted for.
6. **The antonym/scalar strategy is formally rejected** (§4.1): coverage rises to 47.2–56.4%, but
   only **4.7%** of reachable items admit answer remapping.
7. **Recommended: target-anchored rewriting** (§4.2) — choose the distractor first, then rewrite
   the fact to entail it. Satisfies the validity condition by construction, which is exactly what
   every fact-anchored strategy fails.

### 0.1 Two corrections to the planning document

> **(i) My speculated direction of bias was wrong.** The plan said the surface artifact would
> inflate `context_dependent` and thereby understate prior-dependence. Measured: leaking OBQA items
> have *lower* `context_dependent` (60.7% vs 69.7%), *identical* `prior_dependent` (17.9% vs
> 18.0%), and *higher* `unstable_other` (**21.4% vs 12.4%**). The defective rewrites **break** the
> solver rather than steering it. Excluding them therefore *raises* OBQA's ex6 AUCs (§3.3) — the
> published numbers are depressed by low fidelity, not inflated by leakage.
>
> **(ii) Two further ineligibility causes exist.** The plan had a two-way (A)/(B) split; the hand
> read found **(C) fact-insufficient** (~25%) and **(D) inference-licensed** (~5%).
>
> **(iii) Cause (C) is NOT the mechanism behind finding 5 — I proposed it and it is refuted.**
> §4.3.2 tested it directly: in all 16 OBQA `exact` items read, `fact1` determines the answer.
>
> **(iv) …and my proposed replacement was overstated.** I reported target type-incoherence as a
> near-deterministic substitute (6/8 vs 0/8). Extending to all 42 items with the unread ones
> classified blind gives **φ = 0.253** and an expected gain of only **~4 pp** on the 34.5%. The
> 6/8-vs-0/8 figure was **extreme-group sampling inflation** — I sampled the 8 most-unstable and 8
> most-stable items, which maximises separation by construction. Finding 5 is back to
> **unexplained**, and the §5.1 re-opening of criterion 6 is narrowed accordingly.

---

## 1. Block A — fidelity by substitution tier

| | SciQ `exact` | SciQ `morph` | **SciQ `partial`** | OBQA `exact` | OBQA `morph` | **OBQA `partial`** |
|---|---:|---:|---:|---:|---:|---:|
| n | 813 | 11 | 36 | 42 | 5 | 97 |
| coverage = 1.0 | 810/810 | *n/a* | **4/36** | 41/41 | *n/a* | **5/97** |
| coverage < 0.5 | 0 | *n/a* | 6 | 0 | *n/a* | **67** |
| token mismatch > 1 | 2 | 0 | **10** | 0 | 0 | **79** |
| glued residual | 7 | 0 | **9** | 0 | 0 | **20** |
| over-substitution (>2 sites) | 170 † | 0 | **2** | 0 | 0 | 0 |
| unmatched gold remainder survives | — | — | **25/36 (69.4%)** | — | — | 9/97 (9.3%) |
| **whole-phrase gold leakage** | **0** | **0** | **0** | **0** | **0** | **0** |
| distractor already in passage | 0 | 0 | 0 | 0 | 0 | 0 |

† Multi-site substitution is **benign on `exact`** — a long extractive passage naturally repeats the
answer, and every site is a correct whole-word match. It is a defect only when the match is a
*fragment*, which is why the two SciQ `partial` cases matter: `percent` replaced at 6 sites and
`cells` at 5, corrupting the passage globally. Read this row for `partial` only.

*Coverage is word-set overlap and is **invalid for the `morphological` tier**, where the match is an
inflection of the gold by construction. All 16 morphological matches across both datasets were
inspected individually (`lamprey`↔`lampreys`, `favorable trait`↔`favorable traits`, …) and every one
is a legitimate inflection. The tier is clean.*

**Whole-phrase gold leakage is zero everywhere and no chosen distractor already occurs in its
passage.** Those two channels are genuinely well-controlled; the defect is fragment-level.

### 1.1 The five mechanical failure modes of `partial` (all 36 SciQ items inspected)

| Mode | Example |
|---|---|
| Fragment→full-phrase splice | gold `transporting electrcal energy`, matched only `transporting` → *"storing electrical energy electrcal energy"* |
| Over-substitution | matched `percent` replaced at **6** sites; matched `cells` at **5** |
| Remainder leakage | gold `three percent`, matched `percent` → *"three thirty percent"* — the gold's other word survives adjacent |
| Self-contradiction | matched `polar` → inserted `secular and nonpolar` |
| Duplication | matched `size and number of` → *"increase size and number of genes cells"* |

OBQA `partial` shows the same modes, more severely: *"wind carries **sand is always moving** from one
place to another place"*; *"no **they are easier to make** shines through an opaque object"*;
*"humans changing an **the trail is expanded** sometimes causes that **the trail is expanded** to be
destroyed"*.

---

## 2. Block B — surface-cue leakage

Model-free probe: pick the option with the highest share of its content words present in the
passage (1/k credit on k-way ties; chance = 25%). The quantity of interest is the **asymmetry** —
does Setting C make the target more surface-guessable than Setting B made the gold?

| Split | B picks **gold** | C picks **cf-target** | Δ | vs ±5 pp bar |
|---|---:|---:|---:|:--:|
| SciQ ALL | 91.1% | 88.8% | **−2.3 pp** | ✅ |
| SciQ `exact` | 92.3% | 89.0% | **−3.4 pp** | ✅ |
| SciQ `morphological` | 40.9% | 86.4% | +45.5 pp | ⚠️ *metric artifact* |
| SciQ `partial` | 79.9% | 85.6% | +5.8 pp | ❌ |
| OBQA ALL | 66.8% | 97.9% | +31.1 pp | ❌ |
| OBQA `exact` | 97.0% | 98.2% | **+1.2 pp** | ✅ |
| OBQA `morphological` | 40.0% | 100.0% | +60.0 pp | ⚠️ *metric artifact* |
| **OBQA `partial`** | **55.5%** | **98.5%** | **+43.0 pp** | ❌ |

The two `morphological` rows (n=11, n=5) are the same metric limitation as §1: the probe's
Setting-B term fails to find the gold because the passage carries a *different inflection*, so the
asymmetry is manufactured by the probe, not by the rewrite. Consistent with the tier being clean on
inspection; disregard both rows.

**Both datasets' `exact` tiers pass. Both `partial` tiers fail, OBQA catastrophically.**

---

## 3. Block D — does the probe track real solver behaviour? *(Addition 1)*

### 3.1 The direct contingency is null

| | OBQA `partial` (n=82) | SciQ `partial` (n=30) |
|---|---:|---:|
| P(`context_dependent` \| leakage created) | 0.821 | 0.800 |
| P(`context_dependent` \| no leakage) | 0.837 | 0.800 |
| φ | **−0.022** | **0.000** |
| AUC(leakage Δ → `context_dependent`) | 0.553 | 0.622 |

### 3.2 On OBQA that null is uninterpretable — range restriction

The probe picks the counterfactual target in Setting C for **96.9%** of OBQA `partial` items and
**97.6%** of `exact`. There is no unexposed comparison group, so **the test cannot be run on OBQA**.
A near-constant cannot correlate with anything.

### 3.3 On SciQ the test *is* powered, and it comes back at chance

SciQ `exact` has real variance — 20.8% of items (169) have the target *not* fully surface-findable.
Over all 860 eligible items, n+=669 / n−=115:

| Predictor | AUC → `context_dependent` |
|---|---:|
| probe picks cf-target in C | **0.5470** |
| leakage Δ (item level) | 0.5418 |

**Verdict on criterion 5: it remains a plausible proxy, NOT a validated predictor of solver
susceptibility.** Stated plainly as the addition asked.

But the null is informative in the other direction, and this is the audit's most reassuring result:
**if SciQ's `context_dependent` labels were surface artifacts, the probe would predict them. It does
not.** That is positive evidence that SciQ's Setting-C labels reflect something other than lexical
matching. The same cannot be said for OBQA, where the test is simply unavailable — the +43 pp
asymmetry shows the rewrite *creates* a cue, and we cannot show whether solvers use it.

Criterion 5 therefore stands as a **construction-hygiene** requirement — *do not manufacture cues* —
rather than as a measured predictor of harm.

### 3.4 The ex6 discount *(Q5)*

Re-running the ex6 Tier-1 AUCs with surface-leaking items excluded. Per-item flags are in
`results/ex7_counterfactual_construction/leakage_flags_{sciq,obqa}_test_full.json`; ex2 and ex6 are
untouched.

| | **OBQA** published | OBQA leak-free | | **SciQ** published | SciQ leak-free |
|---|---:|---:|---|---:|---:|
| items | 145 | **89** (56 dropped) | | 860 | 838 (22 dropped) |
| `F` | 0.7568 | **0.8327** | | 0.8987 | 0.9008 |
| `D` | 0.6550 | **0.7409** | | 0.8241 | 0.8240 |
| `D'` | 0.6354 | **0.7349** | | 0.8574 | 0.8609 |
| `KDA_cont` | 0.3550 | **0.4163** | | 0.5494 | 0.5507 |

**How to read this for the mentor update.** SciQ's numbers need **no discount** — they move by
≤0.004. OBQA's published ex6 numbers are **not inflated by the artifact; they are depressed by
it.** Dropping the 56 defective items raises every signal by 0.06–0.10 AUC. So the honest statement
is *"the published OBQA figures understate what a clean construction would give, and rest on a
sample where 39% of items carry a manufactured surface cue"* — not *"the OBQA figures are inflated."*

⚠️ Two limits on that reading: the leak flag is confounded with substitution tier (most leaking items
are `partial`), and dropping 39% of a 145-item split is a large composition change. Treat the
leak-free column as an indication of direction and rough magnitude, not a corrected result.

---

## 4. Root causes and the construction decision

### 4.1 Antonym / scalar swap — **formally rejected** *(Q2)*

| Lexicon | facts with ≥1 term | projected eligibility | **options admit remapping** |
|---|---:|---:|---:|
| core (36 pairs) | 128/500 = 25.6% | 236/500 = **47.2%** | **6 (4.7%)** |
| permissive (65 pairs) | 191/500 = 38.2% | 282/500 = **56.4%** | 11 (5.8%) |

Coverage nearly doubles. It does not matter. Setting C requires *some option to be correct under the
perturbed passage*, and only **4.7%** of reachable items contain a polar contrast among their four
options. For the other ~95%, inverting the rule leaves no option correct — no counterfactual target,
and Setting C degenerates into "did the solver stop picking gold", which is a weaker, different
signal and a manufactured source of `unstable_other`. Effective yield ≈ **6 items**.

**This strategy is not carried forward as an alternative.** *(The remapping test is a necessary
surface condition, so the true rate is bounded below by 4.7%, not equal to it; it would have to be
wrong by more than an order of magnitude to change the verdict.)*

### 4.2 Recommended: target-anchored rewriting

The current method is **source-anchored** — find the gold in the passage, replace it. That fails
whenever the gold is not in the passage, which is 71.0% of OBQA. Invert the pipeline: **choose the
counterfactual target first, then rewrite the fact so that target follows.**

| | Q: *"Which animal lays eggs?"* · options `emus`(gold), `skunk`, … · `fact1`: *"birds lay eggs"* |
|---|---|
| Source-anchored (current) | `emus` not in fact → **ineligible** |
| **Target-anchored** | target = `skunk` → *"mammals lay eggs"* → **`skunk` is now supported** |

Why it follows from the audit:

- Solves cause **(A)** directly — never needs the gold in the fact, the structural blocker for 33.5%
  of ineligible items.
- **The validity condition holds by construction**, not by luck — precisely the 4.7% failure above.
- Preserves one-clause form, avoiding every §1.1 splice mode.
- Inserts a *category term* (`mammals`), not the option string (`skunk`), so it should not reproduce
  the +43 pp leakage. **Must be measured (criterion 5), not assumed.**

**What stays fragile:**
1. It does not address cause **(C)** or finding 5. Expect OBQA instability to persist.
2. LLM generation ⇒ Task A's 13:1 over-crediting precedent applies. Human spot-check is a
   **prerequisite**, not a follow-up.
3. Choosing the category term is knowledge-intensive; some rewrites will be false in ways that make
   *no* option correct.
4. It is a larger intervention than a one-word swap — see the standing caveat in §6.

### 4.3 OBQA ineligibility causes

Heuristic split of the 355, **unvalidated** (transparent rules, no human labels yet):

| Cause | Heuristic n | Share | Reachable by a better matcher? |
|---|---:|---:|---|
| (A) rule-to-instance | 119 | 33.5% | ❌ — needs the *rule* perturbed → target-anchored rewriting |
| (B) paraphrase / synonym | 79 | 22.3% | ✅ — a semantic matcher would find these |
| unclassified | 157 | 44.2% | — the heuristic has no (C) detector |

#### 4.3.1 Properly-sized hand read — **use these numbers, not the heuristic's**

The heuristic above is not accurate enough to cite. A single-pass hand read (one reader, no second
rater, no κ — exploratory sizing only) of **36 of the 157 `unclassified` items**, plus a **6-item
spot check of each heuristic stratum**, gives:

| Stratum | n | read | (A) rule→instance | (B) paraphrase | **(C) fact-insufficient** | (D) inference-licensed |
|---|---:|---:|---:|---:|---:|---:|
| `unclassified` | 157 | **36** | 7 (19.4%) | 12 (33.3%) | **13 (36.1%)** | 4 (11.1%) |
| heuristic `A` | 119 | 6 | 4 | 1 | **1** | 0 |
| heuristic `B` | 79 | 6 | 2 | 3 | **1** | 0 |

**The ~⅓ cause-(C) estimate from the earlier 18-item read holds up: 13/36 = 36.1%** in the
well-sized stratum. But **(C) is not confined to the `unclassified` bucket** — it appears at ~1/6 in
both heuristic strata (e.g. heuristic-A item 178, *"fossils are formed when layers of sediment cover
the remains of organisms"* → `may end up fueling a car`). Projecting stratum-weighted onto all 355:

| Cause | Projected n | Share of the 355 | Reachable how? |
|---|---:|---:|---|
| (A) rule-to-instance | ~136 | ~38% | target-anchored rewriting |
| (B) paraphrase / synonym | ~112 | ~31% | a semantic matcher |
| **(C) fact-insufficient** | **~90** | **~25%** | ❌ **no construction method — exclude** |
| (D) inference-licensed | ~17 | ~5% | target-anchored rewriting |

So **reachable ≈ 43% by rewriting (A+D) and ≈ 31% by better matching (B); ≈ 25% is unreachable.**
Maximum achievable OBQA eligibility ≈ 145 + 112 + 154 = **411/500 ≈ 82%**, comfortably above
criterion 1's 70% bar — but only if *both* the matcher and the rewriter are built.

A fourth cause, **(D) inference-licensed**, emerged in this read and was not in the plan: `fact1`
*does* determine the answer through a short inference, but no span matches (*"fossil fuels form over
300000000 years, a very long time to a human"* → `significant supplies accumulated prior`). It
groups with (A) for construction purposes.

> ⚠️ **Precision.** The `unclassified` row rests on 36 items; the two heuristic rows on **6 each**,
> where one item moves the rate by 16.7 pp. Treat (C) ≈ 25% as a point estimate in a roughly
> 20–30% band. The 60-item blind sheet is the instrument that fixes this. Separately, the spot check
> shows the heuristic itself is only ~7/12 accurate, so **the 33.5% / 22.3% heuristic split should
> not be cited anywhere.**

#### 4.3.2 Cause (C) is refuted — and its replacement is weaker than first reported

The plan proposed cause (C) as the mechanism behind finding 5 (34.5% pair-level `unstable_other`
surviving a clean `exact` swap). **Tested and refuted.** A side-by-side read of the 8 most-unstable
and 8 most-`context_dependent` of the 42 OBQA `exact` items found that **in all 16, `fact1`
determines the gold answer** — not one is a cause-(C) fact. Cause (C) is a real and large
*ineligibility* cause (§4.3.1) but it does not reach the eligible set.

That read suggested a replacement: the **type-coherence of the inserted counterfactual target**. It
looked near-deterministic at 6/8 vs 0/8. **It is not.** Extending to **all 42 items**, with the 26
previously unread ones classified *blind to their outcome* and the original 16 re-labelled under a
single stated criterion (INCOHERENT = the target is of the wrong semantic type for the slot, or
domain-absurd in it):

| | unstable ≥2/4 | unstable ≤1/4 | total | P(unstable ≥2) |
|---|---:|---:|---:|---:|
| target **INCOHERENT** | 8 | 5 | 13 | **61.5%** |
| target coherent | 10 | 19 | 29 | **34.5%** |
| total | 18 | 24 | 42 | |

**φ = 0.253.** Pair-level instability 44.2% (incoherent) vs 30.2% (coherent); mean 1.77 vs 1.21
unstable pairs per item.

> **The effect is real but modest, and the 6/8-vs-0/8 figure I reported last round was inflated by
> extreme-group sampling.** Reading the 8 most-unstable and 8 most-stable items maximises apparent
> separation by construction. At full n the relationship is a ~27 pp shift in item-level
> instability, not a near-deterministic one.

**The decisive number: coherent targets still produce 30.2% pair-level instability.** If every
incoherent target were replaced by a coherent one, OBQA `exact` would fall from **34.5% to ≈30.2%** —
a ~4 pp improvement. **Target incoherence is a contributing factor, not the driver.**

Two things nonetheless hold. `choose_distractor`
([`counterfactual_passage.py:129-160`](../../code/ex2_counterfactual/counterfactual_passage.py)) ranks
by (1) not already in passage, (2) no overlap with the answer, (3) word-count proximity, (4) option
index — **no semantic or plausibility check at any stage**, which is a genuine gap producing genuine
defects (*"a paper clip is often made of **large** metals"*, *"**insects** means breaking down
surface materials"*, *"**luck** transports materials through the plant"*). And surface properties do
not predict the split at all — mean \|gold words − target words\| is **0.00** for the unstable group
and **0.08** for the stable one. The signal is semantic, and nothing in the current pipeline looks
at it.

#### 4.3.3 The same mechanism operates on SciQ, equally weakly

Testing whether "SciQ escapes this by luck of option homogeneity" is evidence or just a
plausible-sounding claim. 28 SciQ `exact` items — 14 with ≥3/4 unstable pairs, 14 with 0 — shuffled
and classified **blind to outcome** under the same criterion:

| | unstable ≥3/4 | unstable = 0 | total |
|---|---:|---:|---:|
| target **INCOHERENT** | 4 | 2 | 6 |
| target coherent | 10 | 12 | 22 |

**φ = 0.174** — same direction as OBQA, weaker. The incoherent cases are recognisable
(`marsupial`→`gastrointestinal`, `solution`→`link`, `darwin`→`cannon`, `converging lenses`→`powering
lenses`), but **10 of the 14 unstable SciQ items have perfectly coherent targets**
(`atoms`→`ions`, `carbon`→`oxygen`, `nephrons`→`dendrites`, `biology`→`chemistry`).

So the mechanism is **cross-dataset consistent in direction and weak in both**. That is partial
support for a shared distractor-plausibility filter — the defect is real in both datasets and the
pipeline checks for it in neither.

But it does **not** support the "SciQ escapes by option homogeneity" claim as an explanation of the
cross-dataset gap. The incoherent share is 21.4% in the SciQ sample against 31.0% across OBQA's whole
`exact` tier, and those are not comparable figures — the SciQ sample is a deliberate 50/50
extreme-group draw, not a random one. More decisively: since the mechanism is weak in both datasets,
it cannot account for SciQ `exact` sitting at 19.5% pair-level instability while OBQA `exact` sits at
34.5%. **That gap remains unexplained.**

#### 4.3.4 Where finding 5 now stands

| Candidate explanation for OBQA `exact`'s 34.5% | Status |
|---|---|
| Passage length | ❌ Ruled out — within-SciQ control tops out at 23.4% (§1.4) |
| Substitution fidelity | ❌ Ruled out — these are clean `exact` swaps |
| Cause (C), fact-insufficient | ❌ **Refuted** — 16/16 read items have a determining fact |
| Counterfactual-target incoherence | ⚠️ **Contributing, ~4 pp of 34.5%.** Not the driver |
| **Remainder** | ❓ **Unexplained.** ~30 pp of pair-level instability on clean, coherent, determining items |

This is the honest state. Stating it plainly matters more than closing the loop: **ex7 has ruled out
four candidate causes and identified no sufficient one.**

---

## 5. Success criteria under the "methods contribution" scope *(Q1b)*

Fixing the scope now so it cannot move later.

**OBQA is a methods contribution: the deliverable is a construction method that demonstrably meets
its own quality bar, NOT an improved OBQA Setting-C result.** Concretely:

| # | Criterion | Bar | Expected under target-anchored rewriting |
|---|---|---|---|
| 1 | Eligibility on OBQA | ≥ 70% | ✅ **Expected to clear.** Reaches (A)+(B) = ~56% by heuristic, plus items currently eligible; (C) should be *excluded*, not rewritten |
| 2 | Validity — some option correct under the perturbed passage | ≥ 90% of eligible | ✅ **Expected to clear** — holds by construction; this is the criterion the antonym strategy fails at 4.7% |
| 3 | Fidelity — well-formed single clause | ≥ 95% | ✅ Expected to clear; generation replaces splicing |
| 4 | Whole-phrase gold leakage | 0 | ✅ **Expected to clear** — already 0 today; must not regress |
| 5 | Surface-cue leakage \|Δ\| | ≤ 5 pp | ✅ **Expected to clear** — inserts a category term, not the option string. The main empirical risk |
| 6 | **Pair-level `unstable_other`** | ≤ 34.5% | ⚠️ **Re-opened — see §5.1.** The original excuse (cause C) is refuted; the real driver is fixable |
| 7 | SciQ non-regression | eligibility ≥ 97.3%, \|Δ\| within ±5 pp | ✅ Required |

**So "success" = criteria 1, 2, 3, 4, 5 and 7 clear.** If **2 or 5** fails, the strategy is rejected
the way the antonym one was — those two are the load-bearing claims.

**What this scope explicitly does not promise:** stronger OBQA Tier-1/Tier-2 numbers, a lower
per-solver AUC spread, or OBQA becoming a results contribution. Claiming any of those later would be
moving the target.

### 5.1 Criterion 6 — the re-opening is narrowed, not withdrawn

Last round I re-opened criterion 6 on the strength of a 6/8-vs-0/8 read. **At full n that evidence
is much weaker** (§4.3.2): φ = 0.253, and coherent targets still produce 30.2% pair-level
instability against the 34.5% baseline. A distractor-plausibility filter is therefore expected to
buy roughly **4 pp of 34.5%** — worth having, not a fix.

Where that leaves the three consequences I raised:

1. ~~Criterion 6 should probably become a real bar.~~ **Withdrawn.** On current evidence no
   available intervention gets OBQA `exact` materially below ~30%, because the dominant cause is
   unidentified. Criterion 6 should stay **waived** — but for an honest reason (*"the driver is
   unknown"*) rather than the original wrong one (*"it is cause (C), which is unreachable"*).
2. **A distractor-plausibility filter is still worth building, on hygiene grounds.** It removes
   real defects (*"luck transports materials through the plant"*) and the pipeline checks for none
   of them. It applies to **both** datasets — the mechanism is present in SciQ at φ = 0.174 — but
   it should be justified as construct validity, not as an instability fix.
3. **It still decouples from the rewriter** and can be evaluated on the existing 42 `exact` items
   with no generator.

**Net effect on the Q1b scope: it reverts to what you originally pinned** — criteria 1, 2, 3, 4, 5,
7 clear; criterion 6 permitted to fail. The reasoning is different and weaker than the version in
the plan, and the unexplained ~30 pp is now an acknowledged open question rather than an attributed
one.
---

## 6. Standing caveat — cross-dataset comparability *(Q3)*

> **Once target-anchored rewriting is used on OpenBookQA, SciQ and OBQA Setting-C results are no
> longer on a common scale.** SciQ's Setting C is a minimal one-word lexical swap; OBQA's would be a
> generated single-clause rule rewrite — a materially larger intervention. Any figure that places
> the two side by side (prior-dependence shares, context-verified accuracy, adjusted-KDA retention,
> P/S/F/D) must carry this caveat. This applies going forward to README §6.3, the ex2 methodology
> docs, and any ex6 table reporting both datasets.

This joins, and does not replace, the existing caveat that absolute KDA values are not comparable
across datasets without controlling for target-fact style (README §6.6).

---

## 7. Validation protocol for the generation step *(Q4, Addition 2)*

Approved as a **one-off instrument-validation gate**, not a new annotation programme.

- **Sample:** 60 items, stratified — 30 OBQA cause-(A) rule-to-instance, 15 OBQA cause-(B)
  paraphrase, 15 SciQ control.
- **Double-annotated**, blind, reusing `build_blind_corpus_sheet.py` + `kappa_corpus_defect.py`
  rather than a hand-built sheet.
- **Gate: Cohen's κ ≥ 0.70** — the bar `compute_kappa.py` already pre-registers — plus criterion 2
  validity ≥ 90%. **If either fails, the pipeline does not scale.**

**Validity is recorded by failure type, not as a binary** *(Addition 2)*, so a failure says what to
fix:

| Code | Failure type | What it implicates |
|---|---|---|
| `V0` | Valid — exactly one option correct under the perturbed passage | — |
| `V1` | Wrong category term chosen — perturbed fact is well-formed but selects nothing | prompt / term selection |
| `V2` | **Multiple** options become correct | target selection; the item may be unusable |
| `V3` | **No** option correct — the antonym strategy's failure mode | the validity gate itself |
| `V4` | Ungrammatical or not a single clause | generation quality (criterion 3) |
| `V5` | Perturbed fact is trivially false / incoherent rather than counterfactual | fidelity of the intervention |
| `V6` | Fact still does not determine any answer — cause (C) leaked into the eligible set | the eligibility filter, not the generator |

`V3` and `V6` are the two that would force a redesign rather than a prompt fix.

---

## 8. Do now / defer

**Done this round (all three, plus Q5 promoted):**
`audit_construction_fidelity.py` (Blocks A–D + per-item leakage artifacts + ex6 discount) ·
`classify_ineligibility.py` (cause split + 60-item blind sheet) · `antonym_coverage_dryrun.py`.

**Deferred, unchanged:** the target-anchored generator; the 60-item double-annotated κ gate;
a semantic/entailment matcher for cause (B); an explicit cause-(C) detector and exclusion filter;
any re-run of ex1/ex2/ex6; anything touching ex6 §14 circularity or the labelled-validation-set
programme.

---

## 9. Limitations

1. The cause split rests on a **48-item single-pass hand read by one reader** (36 `unclassified` +
   6 in each heuristic stratum), no second rater, no κ. The two heuristic rows are **n=6 each**,
   where one item moves the rate by 16.7 pp. The automated heuristic is ~7/12 accurate and its
   33.5%/22.3% split should not be cited. §4.3.2's cause-(C) refutation rests on **16 items**; its
   type-coherence contingency covers all 42 OBQA `exact` items (26 classified blind, 16
   re-labelled) and §4.3.3's SciQ check is 28 items classified blind — all one reader, no κ. The
   coherence criterion (wrong semantic type for the slot, or domain-absurd in it) involves
   judgement calls, and several items are borderline.
2. The leakage probe is **content-word overlap only** — it misses semantic cues and is invalid on
   the `morphological` tier.
3. Criterion 5 is **not validated** as a predictor of solver susceptibility (§3.3); on OBQA the test
   is unavailable through range restriction.
4. The ex6 discount is **confounded with substitution tier** and drops 39% of the OBQA split.
5. The antonym rejection rests on a **necessary surface condition**, bounding the usable rate below.
6. Nothing here addresses ex6 §14. The labels remain near-tautologically determined by `MarginC`
   however well the passage is built.

---

---

## 10. Closing synthesis — what ex7 settled, and what it did not

The audit branch is closed here. No filter, no rewriter, no changes to
`counterfactual_passage.py`. This section is the handover.

### 10.1 Solid — safe to state to a mentor without hedging

| Finding | Evidence |
|---|---|
| **SciQ's construction is sound and needs no discount.** | `exact` 810/810 coverage 1.0; leakage asymmetry **−3.4 pp**; excluding leaking items moves ex6 AUCs by ≤0.004 |
| **SciQ's Setting-C labels are not surface artifacts.** | Well-powered probe test: AUC **0.547**, n+=669/n−=115. If they were, the probe would predict them |
| **SciQ is not uniformly clean.** | 36 `partial` items: 4/36 coverage 1.0, 69.4% keep unmatched gold words, F AUC 0.750 vs `exact`'s 0.904. Same disease as OBQA at 4.2% prevalence |
| **`morphological` is clean; the coverage metric is not.** | All 16 inspected; every match a legitimate inflection |
| **OBQA `partial` manufactures a surface cue.** | B 55.5% → C **98.5%**, **+43.0 pp**. Not previously measured anywhere |
| **OBQA's published ex6 numbers are depressed, not inflated.** | Leaking items have *higher* `unstable_other` (21.4% vs 12.4%), identical `prior_dependent`. Dropping them raises `D` 0.655→0.741 |
| **The antonym/scalar strategy fails.** | Coverage 47.2–56.4%, but only **4.7%** admit answer remapping. Formally rejected |
| **OBQA ineligibility has four causes; ~25% is unreachable.** | 48-item hand read: (A) ~38%, (B) ~31%, **(C) ~25%**, (D) ~5%. Eligibility ceiling ≈ **82%** |
| **`choose_distractor` has no semantic check.** | Code inspection, `counterfactual_passage.py:129-160` |

### 10.2 Provisional — carries real caveats

| Finding | Caveat |
|---|---|
| Cause split A/B/C/D | 48 items, **one reader, no κ**; the two heuristic strata are **n=6** each. (C) ≈ 25% is a point estimate in a ~20–30% band. The automated heuristic is ~7/12 accurate and **must not be cited** |
| Target incoherence contributes to instability | φ = 0.253 (OBQA, n=42) / 0.174 (SciQ, n=28). Real, weak, worth ≈ **4 pp** of 34.5%. Coherence labels are judgement calls by one reader |
| Criterion 5 as a validation bar | A **plausible proxy, not a validated predictor** — AUC 0.547 on SciQ; on OBQA the test is unavailable through range restriction (97% of items at ceiling) |
| The ex6 OBQA discount | Confounded with substitution tier; drops 39% of a 145-item split. Direction and rough magnitude only |
| Target-anchored rewriting | Designed, not built. Its central claim — validity by construction — is **unvalidated** |

### 10.3 Not settled — and I stopped rather than guess

**OBQA `exact`'s 34.5% pair-level `unstable_other` has no identified cause.** Four candidates ruled
out (passage length, substitution fidelity, cause (C), and — as a sufficient cause — target
incoherence). About **30 pp remains unexplained on clean, coherent, determining items**, and the
same absence explains why SciQ sits at 19.5% and OBQA at 34.5% for reasons this audit cannot name.

This matters for scope: a construction fix is being designed against a defect whose main driver is
unknown. That does not invalidate the fix — coverage, validity and leakage are worth fixing on their
own terms — but it does mean **no construction method should be promised to improve OBQA's
Setting-C stability.**

### 10.4 Open scope decisions — for your mentor, before any build

1. **Criterion 6's bar.** My recommendation: keep it **waived**, with the honest reason (driver
   unknown), not the original wrong one (cause C). The alternative is to make identifying the
   remaining ~30 pp its own diagnostic task before building anything.
2. **Filter vs. rewriter priority.** The distractor-plausibility filter is far cheaper, applies to
   **both** datasets, and is testable on the existing 42 items with no generator — but buys ~4 pp.
   The target-anchored rewriter is the only thing that moves eligibility (29% → up to ~82%) and is
   the load-bearing proposal. My recommendation: **rewriter first**, filter as a component of it,
   since the filter alone does not unblock anything.
3. **The Q1b re-scope.** Reverts to what you originally pinned. Worth an explicit confirmation given
   it moved twice.
4. **Whether OBQA is worth it at all.** ~25% of the split is structurally unreachable, the residual
   instability is unexplained, and §6's caveat means the result would not be comparable to SciQ.
   The alternative — fix SciQ's 36 `partial` items and drop OBQA — is a much smaller job. I lean
   toward proceeding, because the +43 pp leakage finding means the current OBQA numbers are
   *known-defective* and leaving them uncorrected is its own cost. But this is the decision with
   the largest effort consequence and it is genuinely yours.
5. **Cross-dataset comparability (§6).** Already accepted; restated here because it is a permanent
   cost of the recommended path and your mentor should weigh it alongside decision 4.

---

## 11. Close-out: OpenBookQA is retained as a stress test, not pursued

**Decision, 2026-09-16.** The counterfactual intervention does not transfer to OpenBookQA. After
the root-cause audit above, it is closed as a documented limitation rather than engineered around.
No target-anchored rewriter, no distractor-plausibility filter, no further construction work.

**The mechanism fails on contact.** Answer-span substitution assumes the reference material
*states* the answer. OBQA's `fact1` is a one-clause deductive rule of which the gold option is an
instance, so **71.0% of the split has no span to substitute** — 145/500 eligible at the `partial`
tier, 42/500 (8.4%) at `exact`. Two thirds of what survives is substituted badly: of 97 `partial`
items, 5 reach full coverage of the gold answer and 79 splice in a target differing from the matched
span by more than one token, producing text like *"wind carries **sand is always moving** from one
place to another place."*

**The rewrite manufactures a surface cue.** A model-free lexical-overlap probe picks the gold from
the Setting-B passage for 55.5% of `partial` items but picks the counterfactual target from Setting C
for **98.5%** — a **+43.0 pp** asymmetry, against −3.4 pp on SciQ's `exact` tier. Those items skew
toward `unstable_other` (21.4% vs 12.4%), so they *depress* the measured signal rather than inflating
it: excluding them raises OBQA's Tier-1 AUC for `D` from 0.655 to 0.741.

**Better construction would not rescue it.** The clean 42-item `exact` subset — one-word swaps, no
fidelity defects — still shows **34.5% pair-level `unstable_other`** against SciQ `exact`'s 19.5%.
Four candidate causes were tested and ruled out: passage length (a within-SciQ control tops out at
23.4%), substitution fidelity (these swaps are clean), fact-insufficiency (16/16 items read have a
determining fact), and counterfactual-target incoherence (φ = 0.253; fixing every incoherent target
buys ≈ 4 pp of the 34.5%). **Roughly 30 pp remains unexplained.**

**The obvious repairs were costed and declined.** Antonym/scalar rule inversion raises eligibility to
47–56% but only **4.7%** of reachable items have an option that becomes correct under the inverted
rule, so there is no counterfactual target to detect. Target-anchored rewriting — choosing the
distractor first and rewriting the fact to entail it — would satisfy that condition by construction
and could lift eligibility toward a **~82% ceiling**, but it requires LLM generation with a human
validation gate, it leaves the unexplained 30 pp untouched, and it would put OBQA's Setting C on a
different scale from SciQ's, forfeiting cross-dataset comparability (§6). Given a fixed deadline and
a sound SciQ instrument, that is not a good trade.

**What OBQA is now.** A stress test that establishes the boundary of the method: **the counterfactual
intervention requires extractive reference material that states the answer.** SciQ satisfies this
(92.0% `exact`); OBQA's rule-to-instance facts do not. That is a real and reportable finding about
scope — not a failed experiment — and it is why every OBQA Setting-C figure in this repository
carries a caveat rather than a conclusion. The ex7 audit tooling is committed and re-runnable, so
the finding is auditable by anyone who wants to revisit it.

## Related

- [`README.md`](../../README.md) §2.4, §3.2, §6.6 — Setting C, ClashEval lineage, known caveats
- [`docs/ex2_counterfactual/counterfactual_obqa_analysis.md`](../ex2_counterfactual/counterfactual_obqa_analysis.md) — the OBQA fidelity numbers this audit extends
- [`docs/ex6_psfd_score/psfd_formulation.md`](../ex6_psfd_score/psfd_formulation.md) — Phase 4 eligibility table; §14 circularity (out of scope here)
- [`docs/ex5_failure_audit/annotation_final_conclusion.md`](../ex5_failure_audit/annotation_final_conclusion.md) — the Task A 13:1 precedent behind §7
