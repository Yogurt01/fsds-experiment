# The five confirmed counterfactual-construction failures (SciQ)

Items **0, 25, 222, 716, 805** — the five of 55 Q2 items where the counterfactual passage still
reads as stating the **gold** answer. Each was read directly against the question, options, gold
answer and original passage, with no annotator label consulted.

The question asked of each passage: **does it clearly state the counterfactual target?** All five:
no.

Bold marks the surviving gold; the substituted target is in `code`.

---

## Item 0 — `oxidants` → `Oxygen`

*Q: "Compounds that are capable of accepting electrons, such as o 2 or f2, are called what?"*

**Substitution:** `exact` tier, matched `oxidants`, **1 site**. The matcher's first hit was the
section heading — *"Oxidants and Reductants"* → *"`Oxygen` and Reductants"* — and that was the only
occurrence it rewrote.

**What survives:** the answering clause, verbatim —

> Compounds that are capable of accepting electrons, such as O 2 or F2, are called**oxidants** (or
> oxidizing agents) because they can oxidize other compounds.

**Why:** the passage writes *"calledoxidants"* as one token, a real SciQ typesetting artifact. The
whole-word matcher requires a non-letter to the left, so the fused occurrence was invisible to it
while the heading was not. The one occurrence that answers the question is the one it could not see.

---

## Item 25 — `immune system` → `respiratory system`

*Q: "…80 percent of those 60 years and older have one or more chronic disease associated with
deficiencies of this?"*

**Substitution:** `exact` tier, matched `immune system`, **1 site** — again the title.
*"Immune system"* → *"`Respiratory system`"*.

**What survives:**

> The CDC estimates that 80 percent of those 60 years and older have one or more chronic disease
> associated with deficiencies of the **immune systems**.

**Why:** the answering clause uses the **plural**. The substitution ran only on the singular surface
form it had matched, so the plural was never a candidate. The passage's later *"loss of **immune**
function"* and *"age-related **immune** deficiencies"* also stand, reinforcing the gold reading.

---

## Item 222 — `chemical state of solute` → `similar state of solute`

*Q: "What are the majority of solution properties dependent upon?"*

**Substitution:** `partial` tier. The gold phrase never appears in the passage at all, so the matcher
fell back to the fragment `of solute` — and found it in the **wrong sentence**, a colligative-
properties aside. Substituting the full target for that fragment produced a non-sentence:

> …that depend only upon the total concentration `similar state of solute` species, regardless of
> their identities.

**What survives:** the actual answering clause, untouched —

> Many solution properties are dependent upon the **chemical identity of the solute**.

**Why:** two distinct defects compound. The passage states the gold as a **paraphrase** (*chemical
identity of*, not *chemical state of*), which no surface matcher can locate; and the fragment
fallback then anchored on a sentence that does not answer the question. The result asserts nothing
readable while leaving the gold statement intact.

---

## Item 716 — `antioxidants` → `neurotransmitters`

*Q: "What do you call health-promoting molecules that inhibit the oxidation of other molecules?"*

**Substitution:** `exact` tier, matched `antioxidants`, **2 sites** — both plural, both in framing
sentences:

> `Neurotransmitters` are important for the health of a cell. … `Neurotransmitters` prevent these
> chain reactions from even initiating.

**What survives:** the definitional sentence, which is the one that answers the question —

> An **antioxidant** is a molecule that inhibits the oxidation of other molecules.

**Why:** the definition is **singular**. Substituting both plural occurrences left the only sentence
that matches the question's wording untouched — and produced a passage that now names
`neurotransmitters` as cell-protective while still defining an *antioxidant* as the oxidation
inhibitor the question asks for.

---

## Item 805 — `acid` → `base`

*Q: "Certain air pollutants form which liquid when dissolved in water droplets in the air?"*

**Substitution:** `exact` tier, matched `acid`, **4 sites** — *"`base` fog"*, *"`base` rain"*,
*"`Base` rain"*, etc.

**What survives:** the answering clause, which is a near-verbatim restatement of the question —

> Certain air pollutants form **acids** when dissolved in water droplets in the air.

**Why:** **plural** again. Four substitutions, none of them the one that mattered. The result is also
internally incoherent — *"`base` fog and `base` rain, which may have a pH of 4 or even lower"* — and
the passage retains *"**Acidity** is an important factor"* and *"too **acidic** for fish"*, so the
gold reading is supported from three directions.

---

## The shared mechanism

**One defect produces all five: the pipeline matches a single surface form of the gold answer,
rewrites every occurrence of *that form*, and stops — while the passage states the same fact in
other forms that were never candidates.**

Four things follow from the case reads, and each is independently checkable:

1. **The surviving form is almost always an inflection.** Plural-against-singular in 25, 716 and
   805; a fused word boundary in 0; a paraphrase in 222. Nothing exotic — the commonest morphology
   in English prose.

2. **Site selection is adversarial to the goal.** In 0 and 25 the matcher's only hit was a
   **heading**, the least semantically load-bearing occurrence in the passage. The clause a reader
   actually answers from was left alone precisely because it was written differently.

3. **Substitution count is not a safety signal.** Item 805 rewrote four sites and still failed;
   item 0 rewrote one and failed. What matters is whether the *answering clause* was among them, and
   nothing in the pipeline checks that.

4. **Failure is not confined to the degraded tier.** Four of the five are `exact`-tier
   substitutions — the tier treated as clean. Only 222 is `partial`, and it fails for an additional
   reason: the gold is a paraphrase, so there was no correct site to find.

**Consequence.** In four of the five, the original answering clause survives **verbatim**. A solver
answering gold on these items is reading the passage correctly, not overriding it — so the
`prior_dependent` label those responses receive is an artifact of the construction, not a behavioural
finding.

**What a fix has to do.** Post-substitution, verify that no morphological, fused or fragmentary form
of the gold remains in a sentence that answers the question — the check that the current whole-word
residual counter cannot perform. Item 222's class is not repairable this way and should be excluded
rather than rewritten: when the passage only paraphrases the gold, there is no site a lexical
substituter can correctly target.
