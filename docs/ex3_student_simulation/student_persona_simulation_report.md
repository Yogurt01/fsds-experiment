# Persona-Conditioned Student Simulation under Zero Context

**Experiment**: ex3 (`code/ex3_student_simulation/`)
**Model**: `Qwen3-4B-Instruct-2507`, 4-bit NF4 (bitsandbytes), fp16 compute, RTX 3050 Laptop (4 GB)
**Setting**: Zero-Context / Setting A — no reference passage is shown at any point
**Data**: SciQ test (884 items), OpenBookQA test (500 items)

## 1. Motivation

The ex1 reproduction established that this model is **parametrically saturated** on these
benchmarks when scored with no context at all:

| Dataset | $Acc_{wof}$ (no persona, no fact) | $KDA_{disc}$ |
|---|---|---|
| SciQ | 0.954 | 0.951 |
| OBQA | 0.826 | 0.586 |

At 95.4% zero-context accuracy on SciQ the simulated student already knows nearly every
target fact, so KDA's discriminative term is computed over a vanishing denominator and the
metric loses its power to rank questions. The question this experiment asks is narrow and
mechanical: **can persona conditioning alone manufacture separable ability tiers from a
single saturated model, without introducing external context?**

## 2. Design

### 2.1 Personas

Three ability tiers — Beginner, Intermediate, Advanced — written separately for each
dataset (`code/ex3_student_simulation/personas.py`), because a "beginner" on SciQ's
college-textbook prose is a different reader from a "beginner" on OBQA's elementary
science. Each profile names its *failure mechanism* explicitly (unknown technical
vocabulary, word-matching against the stem, longest-option preference, named
misconceptions, inability to chain two reasoning steps) rather than vaguely asking the
model to be worse.

### 2.2 Paradigm 1 — Joint (contrastive)

All three profiles appear in **one** prompt with the question. The assistant turn is
teacher-forced through a fixed scaffold, and each tier's committed letter enters the
context before the next tier is scored:

```
Beginner: <scored here, argmax committed>
Intermediate: <scored here, argmax committed>
Advanced: <scored here>
```

This is what makes the prediction genuinely joint rather than three independent draws.

### 2.3 Paradigm 2 — Isolated (roleplay)

Three fully independent prompts, one per tier, each asking the model to *be* that student.
No tier sees any other.

### 2.4 Scoring

Both paradigms use the **letter-logit protocol** carried over from ex1, so the two are
directly comparable and no answer is ever lost to free-text parsing: logits at the scored
position are restricted to A/B/C/D, all single-token spellings of a letter are combined
with `logsumexp`, and a softmax over those four values gives a proper distribution over the
options. The joint scaffold reuses a KV cache across its three read points; this was
verified to be numerically identical to recomputing each position from scratch
(max |Δp| ≈ 5e-3, no prediction changes).

### 2.5 Order control

The joint scaffold necessarily emits the tiers in *some* order. Because that turned out to
matter more than anything else in this study, the whole joint condition was re-run with the
order reversed (`--joint-order descending`, Advanced -> Intermediate -> Beginner) as a
control. Everything else — personas, question format, scoring — is identical.

## 3. Results

### 3.1 Accuracy stratification

Full test splits, 95% Wilson intervals. The persona-free reference is the same model on the
same zero-context prompt from ex1.

**SciQ (n = 884), persona-free $Acc_{wof}$ = 0.954**

| Paradigm | Beginner | Intermediate | Advanced | Gap (A−B) | Monotone |
|---|---|---|---|---|---|
| Joint (Beg→Int→Adv) | 0.528 [0.495, 0.561] | 0.887 [0.864, 0.906] | 0.870 [0.846, 0.890] | +0.342 | **no** |
| Isolated | 0.920 [0.900, 0.936] | 0.955 [0.939, 0.967] | 0.956 [0.940, 0.968] | +0.036 | yes |

**OpenBookQA (n = 500), persona-free $Acc_{wof}$ = 0.826**

| Paradigm | Beginner | Intermediate | Advanced | Gap (A−B) | Monotone |
|---|---|---|---|---|---|
| Joint (Beg→Int→Adv) | 0.664 [0.621, 0.704] | 0.732 [0.692, 0.769] | 0.436 [0.393, 0.480] | **−0.228** | **no** |
| Isolated | 0.716 [0.675, 0.754] | 0.814 [0.778, 0.846] | 0.824 [0.788, 0.855] | +0.108 | yes |

Two observations, both of which survive the analysis below:

**The isolated advanced persona is a no-op.** SciQ 0.956 vs a persona-free 0.954; OBQA
0.824 vs 0.826. Asking the model to role-play an expert changes nothing, which is expected
— but the beginner persona barely moves it either (SciQ −3.4 points). Under isolated
roleplay the parametric prior leaks straight through the persona, exactly as the paradigm-2
hypothesis feared.

**Neither paradigm is monotone.** Isolated is technically monotone but its total spread
(0.036 on SciQ) is smaller than the width of the confidence intervals it separates. Joint
violates monotonicity on both datasets — mildly on SciQ (Int 0.887 > Adv 0.870), severely
on OBQA, where the advanced tier lands 22.8 points *below* the beginner.

### 3.2 Paradigm comparison

Exact McNemar tests on paired per-question predictions (b = first-only-correct,
c = second-only-correct):

| Dataset | Comparison | b/c | p |
|---|---|---|---|
| SciQ | joint: advanced vs beginner | 377/75 | <0.001 |
| SciQ | isolated: advanced vs beginner | 41/9 | <0.001 |
| SciQ | beginner: joint vs isolated | 13/359 | <0.001 |
| SciQ | advanced: joint vs isolated | 10/86 | <0.001 |
| OBQA | joint: advanced vs beginner | 107/221 | <0.001 |
| OBQA | isolated: advanced vs beginner | 78/24 | <0.001 |
| OBQA | advanced: joint vs isolated | 25/219 | <0.001 |

| Dataset | Gap (joint) | Gap (isolated) | Joint − isolated |
|---|---|---|---|
| SciQ | +0.342 | +0.036 | **+0.305** |
| OBQA | −0.228 | +0.108 | **−0.336** |

Taken at face value this answers the paradigm-comparison question in opposite directions on
the two datasets: joint contrastive prompting produces a far wider gap on SciQ and a far
worse one on OBQA. Section 3.3 shows both are the same effect.

### 3.3 The order ablation — the joint gap is an emission-order artifact

The joint scaffold commits each tier's letter into the context before the next tier is
scored. That is the mechanism intended to make the prediction contrastive. What it actually
does is pressure whichever tier is written **last** into disagreeing with the tiers already
written.

| Dataset | Order | Last tier | Breaks a consensus | Those breaks that are wrong | Breaks away from a *correct* consensus |
|---|---|---|---|---|---|
| SciQ | Beg→Int→Adv | advanced | 23.7% (n=498) | 63.6% | 60.2% |
| SciQ | Adv→Int→Beg | beginner | 71.4% (n=744) | **97.7%** | 96.2% |
| OBQA | Beg→Int→Adv | advanced | 72.1% (n=341) | 85.4% | 78.0% |
| OBQA | Adv→Int→Beg | beginner | 66.7% (n=363) | **92.1%** | 85.1% |

All three tiers give distinct answers only 5–15% of the time (chance for three independent
uniform picks: 37.5%), so the tiers are *not* independent draws — they are one answer plus a
positional perturbation.

Reversing the emission order flips the entire result:

| Dataset | Order | Beginner | Intermediate | Advanced | Gap | Monotone |
|---|---|---|---|---|---|---|
| SciQ | Beg→Int→Adv | 0.528 | 0.887 | **0.870** | +0.342 | no |
| SciQ | Adv→Int→Beg | **0.291** | 0.853 | 0.930 | **+0.639** | yes |
| OBQA | Beg→Int→Adv | 0.664 | 0.732 | **0.436** | −0.228 | no |
| OBQA | Adv→Int→Beg | **0.348** | 0.720 | 0.756 | **+0.408** | yes |

Putting the beginner last produces a textbook ability ladder on both datasets: monotone,
gaps of +0.639 and +0.408, and a beginner tier pushed down to near the 0.25 chance floor on
SciQ. It looks like a complete success. It is not: the same knob that produces it produced
the OBQA inversion, and the tier being suppressed is simply the one that had to disagree
with two letters already on the page. **The joint paradigm does not simulate ability; it
simulates disagreement, and the ability labels only determine who is unlucky enough to go
last.**

### 3.4 Error-pattern analysis

**Do beginner errors cluster on the designed traps?** No. Trap-hit rate among errors,
restricted to questions where the trap option exists uniquely and is a distractor; a student
erring uniformly over the three distractors scores 1/3. Two-sided exact binomial vs 1/3:

| Dataset | Paradigm | Tier | Longest option | Lookalike to gold |
|---|---|---|---|---|
| SciQ | joint | beginner | 0.367 (n=248, p=0.28) | 0.304 (n=329, p=0.27) |
| SciQ | isolated | beginner | 0.452 (n=42, p=0.10) | 0.393 (n=56, p=0.39) |
| OBQA | joint | beginner | 0.337 (n=89, p=1.00) | 0.349 (n=126, p=0.71) |
| OBQA | isolated | beginner | 0.328 (n=67, p=1.00) | 0.224 (n=107, p=0.02 — *below* chance) |

Not one beginner trap rate significantly exceeds chance. The profiles name the longest-option
heuristic and surface-association explicitly, and the model does not reproduce either at a
measurable rate.

**What the errors actually are: positional collapse.** Comparing each tier's answer-letter
histogram against the dataset's own gold-letter spread (Pearson χ², df=3):

| Dataset | Paradigm | Tier | Answers A/B/C/D | χ² vs gold | Errors A/B/C/D |
|---|---|---|---|---|---|
| SciQ | joint (Beg last→no) | beginner | **0.54**/0.19/0.15/0.12 | 346.1, p<0.001 | **0.63**/0.18/0.10/0.09 |
| SciQ | joint | advanced | 0.26/0.24/0.25/0.26 | 0.6, p=0.90 | 0.20/0.34/0.25/0.21 |
| SciQ | isolated | beginner | 0.26/0.23/0.27/0.24 | 2.2, p=0.53 | 0.17/0.23/0.39/0.21 |
| OBQA | joint desc | beginner | 0.08/0.14/0.25/**0.53** | 339.6, p<0.001 | 0.04/0.12/0.24/**0.61** |
| — | *gold spread* | SciQ / OBQA | 0.27/0.23/0.25/0.25 — 0.28/0.25/0.26/0.21 | — | — |

The degraded beginner tier does not make a beginner's mistakes; it piles onto a letter. In
the ascending SciQ run 63% of its errors are "A"; in the descending OBQA run 61% are "D".
The one case that avoids position collapse — SciQ descending beginner, χ²=30.3 with a nearly
flat error spread (0.33/0.24/0.21/0.22) — instead sits at 0.291 accuracy, i.e. **random
noise** (chance = 0.25). So beginner errors degrade either into a letter fixation or into
guessing, and never into the misconception structure the persona describes.

Individual errors nonetheless *look* plausible, which is why this needs measuring rather
than eyeballing. Genuine near-miss confusions from the isolated beginner:

| Question | Gold | Beginner picked |
|---|---|---|
| What are the simplest organic compounds? | hydrocarbons | carbohydrates |
| What are biochemical catalysts that speed up biochemical reactions? | enzymes | polymers |
| How does a neon light produce visible light? | electroluminescence | luminescence |
| …punch a shark where it pulls in air from | its gills | its nose |
| A person has a chance to experience an equinox | biannually | annually |

These are exactly the right kind of error. There are simply far too few of them (71 errors
on SciQ, nearly 6× fewer than the joint tier), and in aggregate they are not concentrated on any
measurable trap.

**Does the ability ladder nest?** A coherent ladder is nested: what an advanced student
misses, a beginner misses too.

| Dataset | Paradigm | Jaccard B∩I | Jaccard B∩A | Jaccard I∩A | Advanced errors also missed by beginner |
|---|---|---|---|---|---|
| SciQ | joint | 0.180 | 0.081 | 0.108 | 34.8% |
| SciQ | isolated | 0.423 | 0.375 | 0.717 | **76.9%** |
| OBQA | joint | 0.379 | 0.157 | 0.159 | 21.6% |
| OBQA | isolated | 0.451 | 0.386 | 0.676 | **72.7%** |

This is the one place isolated roleplay clearly wins. Its tiers form a proper containment
hierarchy; the joint tiers do not (only ~a fifth to a third of advanced errors are shared
with the beginner), which is again what you expect if the joint tiers differ by positional
perturbation rather than by knowledge.

## 4. Conclusions

1. **Neither paradigm yields calibrated ability curves under zero context.** Isolated
   roleplay is monotone and correctly nested but leaves saturation intact — the advanced
   persona reproduces the persona-free baseline to within 0.2 points, and the total spread
   (0.036 on SciQ) is narrower than its own confidence intervals. Joint prompting produces
   large spreads but they are not ability.
2. **The joint paradigm's ability gap is an emission-order artifact.** Reversing the tier
   order moves SciQ's gap from +0.342 to +0.639 and OBQA's from −0.228 to +0.408, and makes
   both monotone — because the suppressed tier is always the last one written, not the least
   able one. The paradigm-1 hypothesis (that side-by-side profiles force contrastive
   reasoning) is **not supported**: what the context forces is disagreement.
3. **Persona conditioning does not reproduce the specified failure modes.** No beginner trap
   rate beats a random-distractor baseline. The degradation mechanism is letter fixation
   (up to 63% of errors on one option, χ² up to 346) or a collapse to chance accuracy.
4. **Implication for KDA.** Persona-conditioned zero-context simulation is not a usable
   substitute for the missing ability spread on a saturated model. A difficulty ranking
   built on the joint tiers would be ranking option position; one built on the isolated
   tiers would have almost no dynamic range. Restoring KDA's discriminative power on
   `Qwen3-4B-Instruct-2507` will need a mechanism that actually removes knowledge — context
   ablation, counterfactual passages (ex2), or a genuinely weaker student model — rather
   than one that asks the model to pretend.

## 5. Limitations

- One model, one quantisation (4-bit NF4). Positional pressure in the joint scaffold may be
  weaker in a larger or unquantised model.
- Greedy argmax commits each tier's letter in the joint scaffold. Sampling would spread the
  committed letters, but would also add variance to the tier being measured.
- The two trap probes are lexical proxies (character-trigram similarity, content-word
  overlap). They do not capture conceptual misconceptions, so "no measurable trap
  concentration" is a statement about these operationalisations, not about all possible
  error structure. The positional-collapse and chance-accuracy findings do not depend on
  them.
- Only two emission orders were tested, not all six permutations.

## 6. Reproduction

```bash
# smoke test (50 items/dataset, ~3 min)
uv run --active python code/ex3_student_simulation/run_student_simulation.py --limit 50 --tag smoke50

# main run: both datasets, both paradigms (~40 min on a 4 GB RTX 3050)
uv run --active python code/ex3_student_simulation/run_student_simulation.py

# emission-order control (~18 min)
uv run --active python code/ex3_student_simulation/run_student_simulation.py \
    --paradigms joint --joint-order descending --tag order_desc \
    --log-file results/ex3_student_simulation/order_desc.log

# regenerate every table in this report
uv run --active python code/ex3_student_simulation/summarize_simulation.py
```

| Artifact | Path |
|---|---|
| Runner | `code/ex3_student_simulation/run_student_simulation.py` |
| Personas | `code/ex3_student_simulation/personas.py` |
| Table generator | `code/ex3_student_simulation/summarize_simulation.py` |
| SciQ results | `results/ex3_student_simulation/results_persona_simulation_sciq.json` |
| OBQA results | `results/ex3_student_simulation/results_persona_simulation_obqa.json` |
| Order-control results | `results/ex3_student_simulation/results_persona_simulation_{sciq,obqa}_order_desc.json` |
| Logs | `results/ex3_student_simulation/{student_simulation,order_desc,smoke50}.log` |

Runtime: SciQ 26.4 min (1.79 s/question), OBQA 13.8 min (1.65 s/question), order control
17.7 min. Peak VRAM 3.41 GB of 3.68 GB.
