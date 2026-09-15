# Annotation Guide — Task A (judge validation) and Task B (corpus defect)

**Audience:** the person doing the labelling, with this file open beside the sheet.
**Written:** 2026-09-08. **Prerequisite:** none — this file is self-contained.
**Status of every label in it:** *instructional only.* No real item is labelled anywhere below.

> ### The one-line version
>
> | | **Task A** | **Task B** |
> |---|---|---|
> | Sheet | `results/ex4_free_response/validation_sheet_pilot_bidir.csv` | `results/ex5_failure_audit/corpus_defect_sheet_sciq33.csv` |
> | Rows | **74** | **33** (15 of them also scored against annotator 1) |
> | You answer | *Does the model's answer mean the same as the gold answer?* | *Is this passage about this question?* |
> | Column | `human_verdict` | `verdict` |
> | Values | `CORRECT` / `INCORRECT` | `MISMATCH` / `ON_TOPIC` |
> | You look at | question, gold answer, model answer | question, passage |
> | You never look at | `judge_forward`, `judge_reverse`, `judge_agree` | `corpus_defect_audit.md` §3.1–3.2 and `passage_overlap_handcheck.json` (see §2.6.3) |
> | Time | ~60–90 min | ~45 min + ~20 min |
>
> The two tasks are unrelated. Do them in separate sittings. Task A is the one that blocks
> downstream work.

---

## 0. Before you start

**Ten minutes of setup that prevents the two ways this goes wrong.**

1. **Task A — hide three columns.** Open the CSV in a spreadsheet and hide (or delete from a
   working copy) `judge_forward`, `judge_reverse`, `judge_agree`. They are the thing you are
   measuring against. Reading them first turns κ into a measure of your compliance, not your
   agreement. The sheet asks for this itself, at
   [`validation_sheet_pilot_bidir.csv:6-7`](../../results/ex4_free_response/validation_sheet_pilot_bidir.csv).

2. **Task B — do not read `docs/ex5_failure_audit/corpus_defect_audit.md` §3.1 or §3.2, or
   `results/ex5_failure_audit/passage_overlap_handcheck.json`, until your sheet is locked.** Those
   carry **annotator 1's verdicts** for 21 of your 33 items, and by the decision recorded in §2.6
   they are the other half of the κ you are about to compute. Reading them first does not just
   weaken the measurement — it voids it. This is easy to trip over because §6 of that same
   document is the protocol you are following: read §6, then stop.

3. **Label in one pass per task, in sheet order.** Rows in the Task A sheet are already in
   randomised order (`plan_option_free_response_experiment.md:579`). Do not sort, do not skip
   ahead, do not go back and "even out" the ratio of CORRECT to INCORRECT.

4. **Write nothing you are unsure of into the verdict column mid-thought.** Both scorers treat a
   blank as "not labelled" and simply drop the row, so a blank is safe; a wrong keystroke is not.

5. **Smoke-test the Task A scorer now, on the empty sheet:**

```bash
.venv/bin/python code/ex4_free_response/compute_kappa.py --sheet results/ex4_free_response/validation_sheet_pilot_bidir.csv
```

It should print `rows in sheet : 74`, `human-labelled : 0`, and
`Nothing to score yet -- fill the human_verdict column first.` If it prints anything else, stop
and fix the plumbing before labelling.

---

# TASK A — the 74-row judge-validation gate

> ### ✅ COMPLETE — 2026-09-08
> All 74 rows labelled; κ = **0.754** (both-directions) / 0.614 (forward-only); gate passed.
> Report: `results/ex4_free_response/kappa_report.json`. Findings and caveats:
> `docs/NEXT_PHASE_HANDOFF.md` §4.0. **The section below is retained as the rubric of record**
> — it documents the criteria those labels were produced under, and would be the starting
> point for the re-run at n=150 after a full-scale free-response run.

## 1.1 The exact judgment, in plain language

You are marking a short-answer science quiz.

For each row you get three things: a **question**, the **gold answer** (the answer key), and a
**model answer** (what a student wrote). Your job is one question:

> **Would you give this student the mark?** That is: does the model answer mean the same thing as
> the gold answer, for this question?

Write `CORRECT` if yes, `INCORRECT` if no. There is no half mark.

**What this is *not* asking.** Five things people naturally try to judge here, and shouldn't:

| Not your job | Why |
|---|---|
| Whether the gold answer is itself right | The gold answer *is* the key by definition. If it looks wrong to you, mark against it anyway and note it separately (§1.5). |
| Whether the model answer is well written, complete, or nicely phrased | The criterion is meaning, not quality. |
| Whether a real teacher would accept it | The criterion is fixed below; apply it, not your own generosity. |
| Whether the model "knew" it or guessed | You cannot see that and it does not matter. |
| What the `cell` (`A_prime` / `B_prime`) or `dataset` column says | They are stratification bookkeeping. `B_prime` means the model was shown a supporting fact; that changes nothing about whether its answer means the same as the gold. Apply the identical criterion to every row. |

## 1.2 The decision criteria — sourced, not invented

There are exactly **three** rules, and they are the ones the LLM judge was given, so agreement is
being measured against a like-for-like standard.

**Source 1 — the sheet's own instruction block,**
[`results/ex4_free_response/validation_sheet_pilot_bidir.csv:3-5`](../../results/ex4_free_response/validation_sheet_pilot_bidir.csv):

> ```
> # CORRECT means: the model_answer means the same thing as the gold_answer, in the context of
> # the question. Ignore spelling, capitalisation and phrasing. A more specific or more general
> # answer counts as CORRECT only if it identifies the same thing.
> ```

**Source 2 — the judge's own system prompt,**
[`docs/ex4_free_response/plan_option_free_response_experiment.md:203-207`](../ex4_free_response/plan_option_free_response_experiment.md)
(same three rules, near-verbatim):

> "Decide whether the student's answer means the same thing as the reference answer, in the
> context of the question. Ignore spelling, capitalisation, and phrasing. A more specific or more
> general answer counts as correct only if it identifies the same thing."

So, as a checklist:

| # | Rule | Verbatim from |
|---|---|---|
| **A1** | Same meaning as the gold, **in the context of the question** | CSV:3-4 · plan:204-205 |
| **A2** | **Ignore** spelling, capitalisation, phrasing | CSV:4 · plan:205 |
| **A3** | More specific / more general → CORRECT **only if it identifies the same thing** | CSV:4-5 · plan:205-206 |

**Source 3 — the label vocabulary,**
[`code/ex4_free_response/compute_kappa.py:39`](../../code/ex4_free_response/compute_kappa.py):
`LABELS = ("CORRECT", "INCORRECT")`. Binary. Nothing else exists; there is no `PARTIAL`, no
`UNSURE`, no third bucket to escape into.

Everything in §1.3 below is **derived** from A1–A3 for cases they do not settle on their face. It
is marked as derived because it does not carry the same authority — where a case is genuinely
50/50 after §1.3, A1 is the tiebreak, not §1.3.

## 1.3 Decision procedure for the ambiguous cases

Run these in order. Stop at the first one that fires.

```
   ┌─ Is the model answer blank, a refusal, a restatement of the question,
   │  or a list of several mutually exclusive candidates?         ── yes ──▶ INCORRECT
   │
   ├─ Strip it down: lower-case, drop punctuation, drop a leading
   │  "a"/"an"/"the", singular↔plural. Do the two strings now match?
   │                                                              ── yes ──▶ CORRECT   [A2]
   │
   ├─ Different words, but do they name the SAME THING for this question?
   │  (synonym, common name vs technical name, formula vs name,
   │   unit conversion of the same quantity)                      ── yes ──▶ CORRECT   [A1]
   │
   ├─ One is broader or narrower than the other.
   │  Ask: in the context of THIS question, does the model's answer
   │  still pick out the gold, and only the gold?                 ── yes ──▶ CORRECT   [A3]
   │                                                              ── no  ──▶ INCORRECT [A3]
   │
   └─ Otherwise                                                            ▶ INCORRECT
```

### The four cases that actually bite

**(a) Paraphrase vs. genuinely different answer.** The test is *referent*, not *wording*. Ask
yourself: *if both answers were correct, would they be pointing at the same object, process, or
value?* If yes → CORRECT no matter how differently worded. If they point at two different things
that merely live in the same topic → INCORRECT, however close the two sound. Two answers being
neighbours in a textbook is not evidence they mean the same thing; near-miss distractors are
designed to be close.

**(b) Partial credit — there is none.** [A "no half mark" rule, derived from `LABELS` at
`compute_kappa.py:39`.] If the gold names a compound thing (two items, a thing plus a
qualifier) and the model supplies only one part, the model has *not* identified the same thing →
INCORRECT. The exception is when the missing part is redundant in context — if the question
itself has already fixed the qualifier, then dropping it still identifies the same thing
(rule A3) → CORRECT.

**(c) More general / more specific.** [Derived from A3.] The operative question is uniqueness in
context, not set membership:

- Model gives a **broader** term. CORRECT only if, given the question, nothing else in that
  broader category could have been the answer. If the broader term would also be satisfied by
  something that is *not* the gold, the student has not identified the gold → INCORRECT.
- Model gives a **narrower** term. CORRECT only if the narrower thing *is* the gold, i.e. the
  question is asking at a level where naming an instance answers it. If the question asks for the
  category and the model names one member of it, that is a different answer → INCORRECT.

**(d) Units, numbers, rounding, format.** [Derived from A1 + A2.] Format is phrasing; value is
meaning.

- Same quantity in different units, or different notation for the same value → **CORRECT**
  (`0.5` / `50%` / `one half`; `2 kg` / `2000 g`; `14th` / `day 14`).
- A *different* value → **INCORRECT**, even if close.
- Rounding: CORRECT if the rounded form still picks out the same answer at the precision the
  question asks for; INCORRECT if the rounding lands on what would be a different answer.
- A number where the gold is a name, or vice versa, is fine if they denote the same thing.

### Two more habits worth having

- **Judge the answer, not the model.** Some rows will have an answer that is factually true and
  still INCORRECT, because it is not what the gold says. That is the intended behaviour of this
  gate: you are measuring agreement on *matching*, not on science.
- **Do not look anything up.** No dataset files, no web, no checking what the "real" answer is.
  Everything you need is in the three columns.

## 1.4 PRACTICE EXAMPLES — not from the real dataset

> ⚠️ **Every example below is invented for this guide.** None of these questions, answers, or
> `item_id`s appears in `validation_sheet_pilot_bidir.csv`. They exist so you can calibrate before
> touching the real 74. Do not copy any verdict from here into the sheet.

Format matches the real sheet's readable columns
(`row_id, dataset, cell, item_id, question, gold_answer, model_answer`). The judge columns are
omitted here for the same reason you should hide them there.

### Clear CORRECT

**P1** — `P1, sciq, A_prime, 9001`
| field | value |
|---|---|
| question | What is the process by which plants convert light energy into chemical energy called? |
| gold_answer | photosynthesis |
| model_answer | Photosynthesis. |

→ **CORRECT.** Rule A2: capitalisation and punctuation only. This is the shape of roughly half the
sheet; it should take you two seconds.

**P2** — `P2, sciq, B_prime, 9002`
| field | value |
|---|---|
| question | At what temperature does pure water boil at sea level? |
| gold_answer | 100 degrees Celsius |
| model_answer | 212 °F |

→ **CORRECT.** §1.3(d): identical quantity, different unit. Format is phrasing (A2); the value is
the meaning (A1). Note `cell = B_prime` changes nothing.

### Clear INCORRECT

**P3** — `P3, obqa, A_prime, 9003`
| field | value |
|---|---|
| question | Rock is worn away over centuries at the base of a waterfall by |
| gold_answer | moving water |
| model_answer | tectonic activity |

→ **INCORRECT.** Two different mechanisms. Same topic (geology, rock change), different referent —
§1.3(a). Topical proximity is not meaning.

**P4** — `P4, obqa, B_prime, 9004`
| field | value |
|---|---|
| question | An animal that hunts at night most likely has |
| gold_answer | large eyes for gathering light |
| model_answer | It depends on the species — could be echolocation, night vision, or a strong sense of smell. |

→ **INCORRECT.** Two independent reasons, either sufficient: it never commits to one answer
(first branch of the flowchart), and the candidates it does list are mutually exclusive
alternatives, not the gold.

### Boundary cases — read the reasoning, not just the verdict

**P5 — broader term that still uniquely identifies the gold** — `P5, sciq, A_prime, 9005`
| field | value |
|---|---|
| question | Blood leaving the left ventricle enters which major artery? |
| gold_answer | the aorta |
| model_answer | the body's largest artery |

→ **CORRECT.** A3 + §1.3(c). The model's phrase is a description rather than a name, and it is
strictly broader in form — but in the context of this question there is exactly one artery it can
denote, and that artery is the gold. Nothing else satisfies the description. *The reasoning that
would flip this:* if the question had been "name two arteries in the systemic circuit", the same
phrase would no longer pin down a unique referent, and it would be INCORRECT.

**P6 — narrower term that is an instance, where the question asked for the category** —
`P6, obqa, A_prime, 9006`
| field | value |
|---|---|
| question | Sound travels fastest through which state of matter? |
| gold_answer | a solid |
| model_answer | steel |

→ **INCORRECT.** §1.3(c), narrowing branch. `steel` is a solid, so the answer is not *false* — but
the question asks which **state of matter**, and naming one particular solid does not answer that
question. The student has demonstrated something adjacent to the gold, not identified it. *The
reasoning that would flip this:* had the question been "sound travels fastest through which of
these materials", `steel` would be the level the question operates at.

**P7 — compound gold, one part supplied** — `P7, sciq, B_prime, 9007`
| field | value |
|---|---|
| question | Electrolysis of water produces which two gases? |
| gold_answer | hydrogen and oxygen |
| model_answer | hydrogen |

→ **INCORRECT.** §1.3(b). The question explicitly asks for **two**; half of a two-part answer does
not identify the same thing, and there is no partial credit. *Contrast, to show where the
exception lives:* if the question had been "electrolysis of water produces oxygen and which other
gas?", the question itself supplies the second half, and `hydrogen` alone would then be CORRECT.

## 1.5 Exact entry format — Task A

| | |
|---|---|
| **File** | `results/ex4_free_response/validation_sheet_pilot_bidir.csv` |
| **Column** | `human_verdict` — the 8th column, already present and empty on all 74 rows |
| **Rows to fill** | all **74** (`row_id` 1 … 74) |
| **Everything else** | leave byte-identical. Do not add, remove, reorder, or rename columns; do not delete the 8 leading `#` lines. |

**Accepted values.** Anything in the left column below is parsed;
[`compute_kappa.py:76-82`](../../code/ex4_free_response/compute_kappa.py) upper-cases and strips
whitespace first, so case and stray spaces do not matter.

| You may write | Parsed as |
|---|---|
| `CORRECT` · `C` · `1` · `Y` · `YES` · `TRUE` | **CORRECT** |
| `INCORRECT` · `I` · `0` · `N` · `NO` · `FALSE` | **INCORRECT** |
| anything else, including blank | **not a label** — the row is silently skipped and excluded from κ |

> ⚠️ That last line is the trap. `y`, `n`, `correct` are fine. `CORECT`, `ok`, `?`, `x`, `maybe`,
> `TRUE ` with a trailing tab — all fine too *except* the misspellings and the non-aliases, which
> vanish without an error message. **The scorer prints `still blank : N`. After labelling, that
> number must be 0.** If it is not, something you typed did not parse.

**Recommended:** write the full words `CORRECT` / `INCORRECT`. They are unambiguous to a later
reader of the committed sheet.

**If you want to flag a suspected mis-keyed gold answer** — do not encode it in `human_verdict`.
Mark the row against the gold as instructed, and keep a separate plain-text note of the `row_id`
and why. Adding a column would not break the scorer (it reads by field name), but it would make
the committed artifact diverge from the one `build_validation_sample.py` produces.

**Spreadsheet warning.** If you open the CSV in Excel or LibreOffice, save back as CSV (UTF-8),
not as `.xlsx`, and check that nothing has been reformatted — some questions contain commas and
quotes, and some `gold_answer` values (e.g. `14th`) can be mangled into dates by autocorrect.
Editing in a plain text editor avoids all of this.

## 1.6 Scoring, and what the number means

```bash
.venv/bin/python code/ex4_free_response/compute_kappa.py \
    --sheet results/ex4_free_response/validation_sheet_pilot_bidir.csv \
    --out results/ex4_free_response/kappa_report.json
```

It computes Cohen's κ twice — against the judge's forward verdict, and against its
both-directions-agree verdict ([`compute_kappa.py:108-119`](../../code/ex4_free_response/compute_kappa.py))
— plus κ within each of the 7 strata, then applies the pre-registered gate
`KAPPA_GATE = 0.70` ([`compute_kappa.py:38`](../../code/ex4_free_response/compute_kappa.py),
marked *"pre-registered; do not change after seeing results"*).

| Outcome | Consequence, applied automatically by the script |
|---|---|
| κ ≥ 0.70 | Judge is reliable enough. `acc_band_low` / `acc_band_high` become the headline, κ quoted alongside. |
| κ < 0.70 | Judge is **not** reliable enough. `acc_normalised` (the deterministic floor) is the headline; the band is diagnostic only; `acc_judged` is not quoted. |

**Neither outcome is a failure of your labelling, and you must not re-label to move the number.**
A low κ is a finding about the judge. Once you have run the scorer, the labels are fixed — going
back to revise rows after seeing κ would void the pre-registration, which is the whole reason the
threshold was written into code before the sheet was filled.

---

# TASK B — the 33-item SciQ corpus-defect screen

> ### ✅ COMPLETE — 2026-09-09
> All 33 rows labelled blind and locked; κ = **0.857** on the pre-registered 15 (agreement
> 0.933, one disagreement, item #584 adjudicated `ON_TOPIC`). Confirmed mismatch rate:
> **12/884 = 1.36%**. Write-up: `docs/ex5_failure_audit/corpus_defect_audit.md` §8. **The section below is
> retained as the rubric of record** — it documents the criteria those labels were produced
> under.

## 2.1 The exact judgment, in plain language

Each item is a **question** from a science quiz, paired with a **passage** that is supposed to be
the supporting text it was written from. Sometimes the pairing is simply wrong — the passage is
about a completely different subject, because of how the corpus was scraped together.

Your job is one question:

> **Is this passage about the same subject as this question?**

Write `ON_TOPIC` if yes, `MISMATCH` if no.

A useful way to feel the judgment: *if a student were handed this passage and asked this question,
would they think a page had been swapped in by mistake?* If yes → `MISMATCH`.

**What this is *not* asking.** These are the five things that will tempt you, and all five are out
of scope:

| Not your job | Why |
|---|---|
| Whether the passage contains the gold answer word-for-word | 24 of your 33 items are on the sheet *precisely because* it doesn't. That is the screen, not the finding. |
| Whether the passage is enough to *answer* the question | A passage can be on topic and still not state the answer. Still `ON_TOPIC`. |
| Whether the gold answer is correct | Mis-key estimation was **explicitly dropped** from this plan — `corpus_defect_audit.md:264` ("Recommendation: drop the mis-key rate estimation"). |
| Whether the passage is well written | Several are scrape artifacts with image credits in them. Judge subject, not quality. |
| Whether the question is a good question | Out of scope entirely. |

## 2.2 The decision criteria — sourced, not invented

**Source 1 — the binary definition,**
[`docs/ex5_failure_audit/corpus_defect_audit.md:276-277`](../ex5_failure_audit/corpus_defect_audit.md):

> "'is this passage about this question?' is a far more objective judgement than the five-way
> failure taxonomy, so κ should be high, and if it is not, the rubric rather than the sample is at
> fault."

**Source 2 — the operative definition of the positive class,**
[`docs/ex5_failure_audit/corpus_defect_audit.md:139`](../ex5_failure_audit/corpus_defect_audit.md), heading the `topic_mismatch` table:

> "The passage is about an unrelated subject and cannot answer the question"

**Source 3 — the full category definitions**, from the previous round's own artifact,
`results/ex5_failure_audit/passage_overlap_handcheck.json` → `categories`:

| Category | Definition, verbatim |
|---|---|
| `topic_mismatch` | "the passage is about an unrelated subject; it cannot answer the question" |
| `legitimate_low_overlap` | "passage is on topic and correct; low overlap is a property of the phrasing (paraphrase) or of the benchmark's design (deductive rule)" |
| `orthographic_defect` | "passage is on topic but a spelling/typo prevents lexical matching" |
| `other` | "anything else" |

**Collapsing four categories to two.** §6 asks for a binary judgement, and the previous round found
`orthographic_defect` = 0 and `other` = 0 across 41 SciQ items examined
([`corpus_defect_audit.md:130-135`](../ex5_failure_audit/corpus_defect_audit.md)). So:

| Sheet value | Covers |
|---|---|
| **`MISMATCH`** | `topic_mismatch` |
| **`ON_TOPIC`** | `legitimate_low_overlap`, `orthographic_defect`, `other` |

If you hit something that is genuinely neither — on topic but garbled beyond reading, or an empty
passage — label it `ON_TOPIC` (it is not a *topic* mismatch) and describe it in the `note` column.
Adjudication (step 3) is where that gets resolved; do not invent a third value, because the κ
computation is binary.

**Source 4 — the known blind spot,**
[`docs/ex5_failure_audit/corpus_defect_audit.md:299-301`](../ex5_failure_audit/corpus_defect_audit.md):

> "Lexical only. The detector cannot see a passage that is on topic but does not support the keyed
> answer — a semantic mismatch with high lexical overlap would pass undetected. Nothing in this
> audit addresses that class."

This is why §2.3's rule below sends "on topic but doesn't support the answer" to `ON_TOPIC`: that
class is *declared out of scope for this audit*. Labelling it `MISMATCH` would silently widen the
audit's definition and make the published rate incomparable with the rate already reported
(1.47% single-annotator at the time this rubric was written; 1.36% after adjudication, `corpus_defect_audit.md` §8).

## 2.3 Decision procedure for the ambiguous cases

```
   ┌─ Read the question. Name its subject in three or four words,
   │  as specifically as you can ("the heart's chambers", not "biology").
   │
   ├─ Read the passage. Name its subject the same way.
   │
   ├─ Are those two subjects the same thing?              ── yes ──▶ ON_TOPIC
   │
   ├─ Different, but is one the direct subject matter of
   │  the other — the passage explains the mechanism,
   │  defines the term, or gives the rule the question
   │  is an instance of?                                  ── yes ──▶ ON_TOPIC
   │
   ├─ Same broad field (both anatomy, both chemistry) but
   │  about DIFFERENT specific structures/processes, and
   │  the passage never touches the question's subject?    ── yes ──▶ MISMATCH
   │
   └─ Otherwise: unrelated subject                                 ▶ MISMATCH
```

### "Topically adjacent" — where the line actually sits

This is the case you asked about and the one that will cost you the most time. The resolution is
at the level of the **specific subject**, not the field.

- **Same field is not enough.** Two passages can both be about human anatomy, or both about
  thermodynamics, and still be a mismatch — if the passage discusses a *different* structure or
  process and never touches the one the question is about, it is `MISMATCH`. A passage does not
  become on-topic by being in the right chapter.
- **Incidental shared vocabulary is not enough either.** A passage can share several content words
  with the question by coincidence — generic words like "water", "parts", "system", "energy" — and
  still be about something else entirely. Ignore word overlap; you are judging subject. (The
  `overlap` column on the sheet is provenance, not evidence — see §2.5.)
- **Different vocabulary is not a mismatch.** The converse trap. A passage that shares almost no
  words with the question can be squarely on topic, when the question is a definitional rewording
  of what the passage says plainly. `corpus_defect_audit.md:168-171` records exactly this shape as
  `legitimate_low_overlap`.
- **"On topic but doesn't state the answer" → `ON_TOPIC`.** By Source 4 above. The passage
  discusses the right subject; whether it happens to contain the keyed answer is a different
  question, and one this audit explicitly does not ask.
- **The tiebreak.** If after all of that you still cannot decide, ask: *could this passage
  plausibly have been retrieved as support for this question by a system trying to do the right
  thing?* If yes → `ON_TOPIC`; if the only explanation is a filing error → `MISMATCH`. Then write
  a one-line `note`. Genuinely contested items are meant to survive to step 3, not to be forced.

## 2.4 PRACTICE EXAMPLES — not from the real dataset

> ⚠️ **Every example below is invented for this guide.** None of these questions or passages is one
> of the 33 real items, and none of these `item_id`s is in the list at §2.5. Passages are
> abbreviated to a couple of sentences; real ones run 50–150 words and often trail off mid-clause.
> Do not copy any verdict from here into the sheet.

### Clear ON_TOPIC

**Q1** — `item_id 9101`, gold `evaporation`, question:
*"What is the process by which liquid water becomes water vapour called?"*
Passage: *"When water is heated, molecules at the surface gain enough energy to escape into the
air. This process is called evaporation, and it is the first stage of the water cycle."*

→ **ON_TOPIC.** Same subject, stated plainly. The easy majority of the sheet looks like this.

**Q2** — `item_id 9102`, gold `a lever`, question:
*"A rigid bar that pivots about a fixed point in order to multiply an applied force is known as?"*
Passage: *"Levers make work easier. Pushing down on the long end of a lever lifts a heavy load on
the short end, because the effort is applied further from the pivot."*

→ **ON_TOPIC.** The passage is unmistakably about levers; the question is a definitional rewording
that shares almost no vocabulary with it. Low word overlap, right subject — the
`legitimate_low_overlap` shape described at `corpus_defect_audit.md:168-171`.

### Clear MISMATCH

**Q3** — `item_id 9103`, gold `the mantle`, question:
*"Which layer of the Earth lies directly beneath the crust?"*
Passage: *"Enzymes are proteins that speed up chemical reactions in cells. Each enzyme has an
active site whose shape fits a specific substrate, and enzyme activity falls sharply outside a
narrow range of pH."*

→ **MISMATCH.** Geology question, biochemistry passage. Nothing connects them. Most true
mismatches are this blatant.

**Q4** — `item_id 9104`, gold `nitrogen`, question:
*"Which gas makes up roughly 78% of Earth's atmosphere?"*
Passage: *"Figure 4.2: Cumulus clouds over the Atlantic. Left: M. Alvarez; Right: NOAA archive,
CC-BY-NC-SA 3.0. Reproduced with permission. See the chapter opener for full credits."*

→ **MISMATCH.** This is not textbook prose at all — it is a caption and credit block that the
scrape pulled in. `corpus_defect_audit.md:162-164` records that some real mismatches take exactly
this form. It has no subject, so it cannot have this question's subject.

### Boundary cases — read the reasoning, not just the verdict

**Q5 — same field, different component** — `item_id 9105`, gold `a capacitor`, question:
*"Which circuit component stores electrical charge for later release?"*
Passage: *"A resistor limits the current flowing in a circuit. Ohm's law relates the current
through a resistor to the voltage across it and its resistance, so doubling the voltage across a
fixed resistor doubles the current."*

→ **MISMATCH.** Both texts are about components in an electrical circuit, and "circuit" and
"current" are shared vocabulary, so this will not feel as blatant as Q3. But resistors and
capacitors are different components doing different jobs, and the passage never touches charge
storage. Same field, different component, no contact with the question's subject → the third
branch of §2.3's flowchart. *What would make this ON_TOPIC:* a passage describing how charge
builds up on two parallel plates separated by an insulator, never once using the word "capacitor",
would still be about the question's subject — arguable, and exactly the kind of item to send to
step 3 with a note.

**Q6 — on topic but does not support the answer** — `item_id 9106`, gold `carbon dioxide`,
question: *"Which gas do plants take in through their stomata during photosynthesis?"*
Passage: *"Stomata are tiny pores on the underside of a leaf, bounded by two guard cells. When the
guard cells swell, the pore opens; when they lose water, it closes, which limits the plant's water
loss on hot days."*

→ **ON_TOPIC.** The passage is squarely about stomata, which is the question's subject. It happens
to discuss water regulation rather than gas intake, so it does **not** support the keyed answer —
but that is Source 4's declared blind spot (`corpus_defect_audit.md:299-301`), explicitly out of
scope. Labelling this `MISMATCH` would change what the published rate measures. Add a `note` if you
want it visible at adjudication.

**Q7 — incidental shared vocabulary** — `item_id 9107`, gold `the sun`, question:
*"What is the primary source of energy that drives Earth's weather systems?"*
Passage: *"A household energy audit begins by identifying where heat is lost. Draught-proofing
doors and adding loft insulation are the cheapest measures; the payback period for double glazing
is considerably longer."*

→ **MISMATCH.** "Energy" and "heat" appear in both, so this item scores non-zero overlap and might
*look* related at a glance. But the passage is about domestic energy efficiency and the question is
about atmospheric physics. Coincidental word sharing is not subject sharing — §2.3, second bullet.
This shape is why the audit's operating point is 0.40 rather than 0.10: real mismatches can score
above zero on incidental words.

## 2.5 Exact entry format — Task B

**The sheet is built and waiting.** It was generated by
[`code/ex5_failure_audit/build_corpus_defect_sheet.py`](../../code/ex5_failure_audit/build_corpus_defect_sheet.py)
(Appendix A), which is deterministic, loads no model, and takes about a second. You do not need to
run anything before you start — just open the file.

| | |
|---|---|
| **File** | `results/ex5_failure_audit/corpus_defect_sheet_sciq33.csv` |
| **Rows** | **33** (`row_id` 1 … 33), SciQ test split only |
| **Column to fill** | `verdict` |
| **Free-text column** | `note` — optional, use it for anything contested |
| **Other columns** | `row_id, item_id, question, gold_answer, passage, overlap, screen` |

Re-running the builder is safe: it **refuses to overwrite a sheet that already carries any
verdict** unless you pass `--force`. Its companion output,
`results/ex5_failure_audit/corpus_defect_comparison_subset.json`, pre-registers the 15-item
comparison subset and the annotator configuration decided in §2.6.

`item_id` is the index into `datasets/sciq/sciq_test_full.json`. The 33 are the union of the two
screens defined at [`corpus_defect_audit.md:243`](../ex5_failure_audit/corpus_defect_audit.md):

```
23, 33, 54, 127, 153, 183, 201, 203, 237, 251, 306, 321, 381, 386, 412, 525, 566,
584, 594, 598, 602, 610, 619, 639, 655, 739, 746, 754, 760, 762, 828, 843, 844
```

(12 in both screens · 9 overlap-screen only · 12 ineligibility-screen only. Verified against
committed data; the builder re-derives them rather than hard-coding them.)

**Accepted values** — from `ALIASES` in
[`code/ex5_failure_audit/kappa_corpus_defect.py:64-68`](../../code/ex5_failure_audit/kappa_corpus_defect.py),
which upper-cases and strips whitespace first, so case and stray spaces do not matter:

| You may write | Parsed as |
|---|---|
| `MISMATCH` · `M` · `1` · `Y` · `YES` | **MISMATCH** |
| `ON_TOPIC` · `ON-TOPIC` · `ONTOPIC` · `O` · `0` · `N` · `NO` | **ON_TOPIC** |
| anything else, including blank | **not a label** — row excluded from κ |

**Recommended:** write the full words `MISMATCH` / `ON_TOPIC`.

> These aliases live in the Task B scorer, which is a **new script**, not an existing one — unlike
> Task A's, which come from `compute_kappa.py:76-82`. `compute_kappa.py` could not be reused: its
> `LABELS` constant is hardcoded to `("CORRECT", "INCORRECT")` at line 39, it reads Task-A-only
> columns (`judge_forward`, `judge_reverse`, `stratum`), and it applies the κ ≥ 0.70 **gate**,
> which belongs to Task A alone and has no authority here. `kappa_corpus_defect.py` reuses its κ
> *formula* (`compute_kappa.py:42-66`) verbatim and nothing else.

**Ignore the `overlap` and `screen` columns while deciding.** They record *why* each item was
screened in, and they are there so the finished sheet is self-describing for a later reader. They
are not evidence about the answer — an item can be on the sheet for a purely lexical reason and be
perfectly on topic. Hide them if your editor allows, same as Task A's judge columns.

## 2.6 The double-annotation protocol — DECIDED 2026-09-08

Step 2 of the plan, verbatim from
[`docs/ex5_failure_audit/corpus_defect_audit.md:272`](../ex5_failure_audit/corpus_defect_audit.md):

> | 2. Double-annotate 15 of the 33, report Cohen's κ | 15 items | ~20 min |

That single table row is the **entire** specification in the source document. It leaves three
things open: *which* 15, *who* the second annotator is, and whether the passes are *blind*. All
three were closed by decision on **2026-09-08**. They are recorded here, in
`results/ex5_failure_audit/corpus_defect_comparison_subset.json` → `_provenance`, and in
`docs/NEXT_PHASE_HANDOFF.md` §2.4–2.5, so a later session does not re-decide them.

**The configuration, in one table:**

| | Decision |
|---|---|
| **Annotator 1** | the existing hand-check, `results/ex5_failure_audit/passage_overlap_handcheck.json` — produced in an earlier session by an LLM agent, single pass, before this guide existed |
| **Annotator 2** | the human filling `corpus_defect_sheet_sciq33.csv` |
| **The 15** | `23, 153, 201, 203, 251, 306, 321, 412, 584, 598, 602, 619, 655, 746, 828` — a seeded draw from the 21 items annotator 1 covered |
| **Blind** | yes, one-directional: annotator 1's labels were fixed long before; annotator 2 must not read them until their sheet is locked |
| **Gate** | none |

### 2.6.1 Which 15 of the 33 — **DECIDED**

`corpus_defect_audit.md` §6 gives a count and no selection rule, so one was chosen and fixed
before any labelling:

```
random.Random(20260904).sample(sorted(prior_labelled_ids), 15)
```

where `prior_labelled_ids` are the **21** SciQ items annotator 1 assigned a category to. The seed
is the project's standing one, the default in both
[`passage_overlap_audit.py:193`](../../code/ex5_failure_audit/passage_overlap_audit.py) and
[`build_validation_sample.py:149`](../../code/ex4_free_response/build_validation_sample.py).

**The 15:**

```
23, 153, 201, 203, 251, 306, 321, 412, 584, 598, 602, 619, 655, 746, 828
```

**Why drawn from the 21, not randomly across all 33.** Only those 21 have a first verdict to
compare against; the other 12 entered the sheet through the Setting-C-ineligibility screen alone
and annotator 1 never saw them. A random 15 across all 33 would have landed roughly 6 items on
which no comparison is possible, leaving κ on ~9 pairs instead of 15.

**This leaks nothing.** The 21 items annotator 1 covered are *exactly* the overlap < 0.40 screen —
the builder asserts this on every run and warns if it ever stops holding. So knowing an item is in
the comparison subset tells you only which screen caught it, which the sheet's own `screen` column
already says. It tells you nothing about how annotator 1 labelled it.

**The remaining 18** (6 unselected from the 21, plus the 12 with no prior verdict) are labelled
single-pass, no comparison. They still count toward the census and the published mismatch rate;
they just do not contribute to κ. They are listed in the manifest as `single_pass_item_ids`.

**The sheet does not mark which 15 are which** — deliberately. An annotator who knows which rows
are scored against someone else tends to work those rows harder, which biases κ upward relative to
the care the other 18 received. Label all 33 uniformly, in `row_id` order.

### 2.6.2 Who the second annotator is — **DECIDED: the existing hand-check is annotator 1**

`corpus_defect_audit.md` §6 names neither a person nor a role, and
`NEXT_PHASE_HANDOFF.md` §2.5 flagged this as needing confirmation before starting. The
configuration chosen is the alternative sketched at §2.6.4 of the first draft of this guide:

**Annotator 1 = the committed hand-check.** `passage_overlap_handcheck.json` already contains a
complete, timestamped, single-pass categorisation of the 21 low-overlap items, produced in an
earlier session, by an LLM agent, *before this rubric was written*. That last point is what makes
it usable: those labels cannot have been shaped by the criteria in §2.2–§2.3, so the comparison is
a genuine test of whether two independent readers converge.

**Annotator 2 = you, the human.**

This satisfies the binding constraint from `NEXT_PHASE_HANDOFF.md` §5 — the double annotation
requires **"≥ 1 real human"**, and *"one agent labeling twice measures nothing."* Here one rater is
a human and the two passes are genuinely separate processes, months and one rubric apart.

**What must be declared when this is published:** one of the two raters is an LLM agent. The κ is
therefore human-vs-LLM agreement on an objective question, not human-vs-human. That is a real
limitation and belongs in the write-up next to the number — it is not a reason to avoid the
comparison, since the project's alternative was no κ at all. The scorer writes this into its
`_provenance` block automatically.

**Its four categories collapse to the binary judgement** exactly as §2.2 sets out:
`topic_mismatch` → `MISMATCH`; `legitimate_low_overlap` / `orthographic_defect` / `other` →
`ON_TOPIC`. The mapping lives in `CATEGORY_TO_LABEL`,
[`kappa_corpus_defect.py:71-76`](../../code/ex5_failure_audit/kappa_corpus_defect.py).

**No file containing annotator 1's verdicts is ever created.** The scorer reads them straight out
of the committed artifact at scoring time. That is a deliberate choice in service of §2.6.3: the
fewer copies of those labels exist, the harder they are to stumble into.

### 2.6.3 Blind — **DECIDED: yes**

§6 says nothing about blinding, but κ is only interpretable if the two annotations are
independent, and §6's own stated purpose (*"if κ is not high, the rubric rather than the sample is
at fault"*, `corpus_defect_audit.md:277-278`) only works if each annotator applied the rubric
alone.

Blinding here is **one-directional and already half-satisfied**: annotator 1's labels were fixed in
an earlier session and cannot be influenced by anything you do. The whole burden is on annotator 2.

**The protocol:**

1. Label all 33 rows of `corpus_defect_sheet_sciq33.csv` from this guide alone.
2. **Lock the file** — commit it, or copy it somewhere read-only. This is the moment the labels
   become final.
3. *Only then* run the scorer, and read `corpus_defect_audit.md` §3.1–3.2 if you want to.
4. Adjudicate disagreements (step 3), knowing both sides.

**Until step 2 is done, these are off limits:** `docs/ex5_failure_audit/corpus_defect_audit.md` §3.1 and §3.2,
`results/ex5_failure_audit/passage_overlap_handcheck.json`, and any summary of either. §6 of that
document is fine — it is the protocol. Read §6 and stop.

The sheet repeats this warning in its own `#` header block, so it travels with the file.

### 2.6.4 What is still a genuine limitation

Recorded so it is not rediscovered as a surprise:

- **One rater is an LLM.** See §2.6.2. Declare it.
- **n = 15 is small.** κ moves several hundredths per disagreement. Always quote n beside it.
- **Annotator 1 labelled a set defined by the overlap screen**, so the comparison covers the
  low-overlap items only. The 12 ineligibility-only items get no second read. The published
  mismatch rate rests on all 33; κ speaks only to the 15.
- **Anything you already remember** from a previous read of §3.1–3.2 is unavoidable and
  uncorrectable. If you believe you remember a specific item's verdict, note that item in the
  `note` column so adjudication can discount it.

## 2.7 How κ gets computed for Task B

Cohen's κ over the double-annotated subset only, treating the two annotators symmetrically:

```
   κ = (Pₒ − Pₑ) / (1 − Pₑ)

   Pₒ = fraction of the 15 items where the two annotators wrote the same value
   Pₑ = Σ over {MISMATCH, ON_TOPIC} of  P(annotator1 = L) × P(annotator2 = L)
```

That is exactly `cohens_kappa` at
[`code/ex4_free_response/compute_kappa.py:42-66`](../../code/ex4_free_response/compute_kappa.py),
with the label set swapped. It is reused verbatim in
[`code/ex5_failure_audit/kappa_corpus_defect.py`](../../code/ex5_failure_audit/kappa_corpus_defect.py)
(Appendix B). **Run it only after your sheet is locked** (§2.6.3 step 2):

```bash
.venv/bin/python code/ex5_failure_audit/kappa_corpus_defect.py \
    --out results/ex5_failure_audit/corpus_defect_kappa.json
```

Every path defaults correctly for the decided configuration, so no arguments are needed: annotator
1 is read from `passage_overlap_handcheck.json`, annotator 2 from `corpus_defect_sheet_sciq33.csv`,
and the 15 comparison items from `corpus_defect_comparison_subset.json`. It joins on `item_id`,
scores only items both annotators labelled, and prints n, observed agreement, expected agreement,
κ, the 2×2 confusion table, and the list of disagreeing `item_id`s — your worklist for step 3.

Useful flags: `--subset all` scores every item the two share rather than the pre-registered 15;
`--annotator-a <sheet.csv>` compares two filled sheets instead of using the hand-check.

**It is safe to run before you finish.** On a blank or partial sheet it prints
`NOT YET LABELLED BY B: [...]` with the outstanding `item_id`s and exits without scoring, so you
can use it as a progress check without seeing any of annotator 1's verdicts.

**There is no gate here.** `KAPPA_GATE = 0.70` belongs to Task A and does not apply. §6 says only
that κ *should* be high and that a low κ indicts the rubric:
*"if κ is not high, the rubric rather than the sample is at fault"* (`corpus_defect_audit.md:278`).
So a low κ is a finding about §2.2–§2.3 of this guide, and the response is to sharpen the rubric and
say so — **not** to re-label until the number improves.

**Two degenerate cases to expect at n=15.** If both annotators label every item the same way (all
`ON_TOPIC`, say), Pₑ = 1 and κ is undefined — the script prints `n/a`, and you report observed
agreement plus the fact that κ is undefined on a degenerate margin. And κ is unstable at n=15
regardless: one disagreement moves it several hundredths. Report n alongside it, always.

**Then step 3** (`corpus_defect_audit.md:273`): adjudicate the disagreements, and publish a corpus
mismatch rate for SciQ as `<confirmed MISMATCH count> / 884`. The previously published figure is
13/884 = 1.47% (`corpus_defect_audit.md:24`); your census either confirms it or moves it, and either
way the number you publish should quote n, κ, and the annotator configuration from §2.6.2.
**Outcome (2026-09-09): it moved — 12/884 = 1.36%, item #584 adjudicated `ON_TOPIC`. See
`corpus_defect_audit.md` §8.**

---

# 3. Self-check quiz — PRACTICE ONLY

> ⚠️ **All five items are invented for this guide.** None is from the real 74 or the real 33.
> Answer all five, then check §3.1. If you miss any, re-read the section the answer key points at
> **before** touching the real sheets.

**Cover the answer key.**

---

**Quiz 1 — Task A**
| field | value |
|---|---|
| question | The waxy layer that reduces water loss from a leaf's surface is called the |
| gold_answer | cuticle |
| model_answer | the cuticle layer |

Verdict?

---

**Quiz 2 — Task A**
| field | value |
|---|---|
| question | Which planet in our solar system has the shortest year? |
| gold_answer | Mercury |
| model_answer | the innermost planet |

Verdict?

---

**Quiz 3 — Task A**
| field | value |
|---|---|
| question | An object's mass is 5 kg. What is its weight on Earth, to the nearest newton? |
| gold_answer | 49 N |
| model_answer | about 50 newtons |

Verdict?

---

**Quiz 4 — Task B**
`item_id 9201`, gold `a conductor`, question: *"A material that allows electric charge to flow
through it easily is called what?"*
Passage: *"Copper wiring is used throughout houses because charge moves through it with little
resistance. Rubber is used for the sheath, since charge does not move through it at all."*

Verdict?

---

**Quiz 5 — Task B**
`item_id 9202`, gold `the epidermis`, question: *"What is the outermost layer of human skin
called?"*
Passage: *"The outer bark of a tree protects the living tissue beneath it. Beneath the bark lies
the phloem, which transports sugars produced in the leaves down towards the roots."*

Verdict?

---

## 3.1 Answer key

**Quiz 1 → `CORRECT`.** Rule A2 plus the leading-article/normalisation branch of §1.3. `the
cuticle layer` and `cuticle` denote the same structure; "the" and "layer" are phrasing. *If you
said INCORRECT,* you are being stricter than the rubric — re-read §1.2 A2.

**Quiz 2 → `CORRECT`.** §1.3(c), the broader-term branch, matching worked example **P5**. `the
innermost planet` is a description, not a name, but in this solar system exactly one planet
satisfies it and that planet is Mercury. Unique identification in context is what A3 asks for.
*If you said INCORRECT,* you applied "it didn't say the word" rather than "it didn't identify the
thing" — re-read §1.3(c).

**Quiz 3 → `INCORRECT`.** §1.3(d), the rounding branch. The question asks *to the nearest newton*,
which fixes the precision; `about 50 newtons` is a different value at that precision. This is the
sharp edge of the units rule: `49 N` / `49 newtons` / `49` would all be CORRECT, because those are
format differences. `50` is not a format difference. *If you said CORRECT,* you treated a numeric
difference as phrasing — re-read §1.3(d).

**Quiz 4 → `ON_TOPIC`.** §2.3. The passage never uses the word "conductor", and its lexical overlap
with the question is low — but it is unmistakably about materials that let charge flow, which is the
question's subject, and it even contrasts them with insulators. Same subject, different vocabulary.
Compare worked example **Q2**. *If you said MISMATCH,* you were judging word overlap rather than
subject — re-read §2.3, third bullet, and remember `overlap` is provenance, not evidence.

**Quiz 5 → `MISMATCH`.** §2.3, third branch, matching worked example **Q5**. Both texts are about
"the outermost protective layer of a living thing", so they are genuinely topically adjacent and
share the structural idea — but one is human skin and the other is tree anatomy. Different specific
subject, no contact with the question's. *If you said ON_TOPIC,* you resolved at the level of the
analogy rather than the subject — re-read §2.3, first bullet ("Same field is not enough").

**Score yourself:** 5/5, start labelling. 4/5, re-read the section named and start. 3 or fewer, read
§1.2–§1.3 and §2.2–§2.3 again in full first — the criteria are short, and a systematic
misunderstanding will show up as a low κ that looks like a finding but isn't.

---

# 4. What you need to hand back

When you are finished, a later session must be able to verify completeness without asking you
anything. Produce exactly these:

**Task A** — `results/ex4_free_response/validation_sheet_pilot_bidir.csv`, with all **74** data rows
(`row_id` 1–74) carrying a parsed value in the **`human_verdict`** column and every other column
byte-identical to the file you started from; plus
`results/ex4_free_response/kappa_report.json`, written by
`compute_kappa.py --sheet <that file> --out <that path>`. Completeness check: the scorer must print
`rows in sheet : 74`, `human-labelled : 74`, `still blank : 0`, and the JSON must contain
`n_labelled: 74`, non-null `forward_only.kappa` and `both_directions.kappa`, all 7 strata under
`by_stratum`, and a `gate_passed` boolean. **Task B** — one sheet only:
`results/ex5_failure_audit/corpus_defect_sheet_sciq33.csv`, with all **33** rows (`row_id` 1–33;
`item_id` matching the list in §2.5 exactly) carrying a value in the **`verdict`** column, locked
before scoring; annotator 1 needs no file, since the scorer reads
`passage_overlap_handcheck.json` directly (§2.6.2). Plus
`results/ex5_failure_audit/corpus_defect_kappa.json`, written by `kappa_corpus_defect.py --out`,
which must record `n_scored = 15`, `unlabelled_by_B: []`, observed agreement, κ (or `null` with the
degenerate-margin reason), the 2×2 confusion table, the disagreeing `item_id`s, and a `_provenance`
block naming both annotators, the subset source, and the blind protocol — the script fills all of
that in on its own, so the check is simply that the fields are populated. The pre-registered
`results/ex5_failure_audit/corpus_defect_comparison_subset.json` is already written and should be
committed unchanged alongside them. Finally, a short note — a paragraph is enough, in
`docs/ex5_failure_audit/corpus_defect_audit.md` as a new §8 or a fresh file — recording the adjudicated SciQ corpus
mismatch rate as `<count>/884` alongside n, κ, and the annotator configuration **including the fact
that one rater is an LLM agent** (§2.6.4), so the rate published at `corpus_defect_audit.md:24`
is either confirmed or superseded on the record. ✅ **Done 2026-09-09** — see `corpus_defect_audit.md`
§8: the figure moved from 13/884 = 1.47% to **12/884 = 1.36%**.

---

# Appendix A — `build_corpus_defect_sheet.py`

[`code/ex5_failure_audit/build_corpus_defect_sheet.py`](../../code/ex5_failure_audit/build_corpus_defect_sheet.py)
— **already run; the sheet exists.** Re-run it only to rebuild from scratch.

```bash
.venv/bin/python code/ex5_failure_audit/build_corpus_defect_sheet.py
```

Deterministic, loads no model, ~1 second. It re-derives the 33 IDs from committed data rather than
hard-coding them: the overlap tail from `passage_overlap_audit.json` → `all_overlaps` (**not**
`tail_items`, which was exported at a 0.10 threshold and holds only 12), and Setting-C
ineligibility from `results_counterfactual_sciq_test_full.json`. It then draws the 15-item
comparison subset from the ids annotator 1 covered, per §2.6.1.

**Writes two files:**

| Path | Contents |
|---|---|
| `results/ex5_failure_audit/corpus_defect_sheet_sciq33.csv` | the 33-row annotation sheet, `verdict` column empty |
| `results/ex5_failure_audit/corpus_defect_comparison_subset.json` | the pre-registered 15, the seed, the selection rule, and the full annotator configuration in `_provenance` |

**Expected output:**

```
  overlap < 0.4        : 21
  Setting-C-ineligible  : 24
  union (sheet rows)    : 33
  prior-labelled (A1)   : 21  [ids only; no verdict was read]
  comparison subset     : 15 (seed 20260904)
  single-pass remainder : 18
```

If the union is not 33, stop — `datasets/` is gitignored and regenerable, and neither preparer pins
an upstream HuggingFace revision, so a different count means the underlying rows have moved.

**Two safety properties worth knowing:**

- It **refuses to overwrite** a sheet that already carries any verdict, unless you pass `--force`.
  So an accidental re-run cannot destroy your labelling.
- It reads `passage_overlap_handcheck.json` for **ids only, never categories**, and asserts that
  those 21 ids are exactly the overlap screen — warning loudly if that ever stops holding, since
  the subset draw would then start leaking which items carry a prior verdict (§2.6.1).

Flags: `--out`, `--manifest`, `--threshold` (default 0.40), `--comparison-n` (15), `--seed`
(20260904), `--force`.

---

# Appendix B — `kappa_corpus_defect.py`

[`code/ex5_failure_audit/kappa_corpus_defect.py`](../../code/ex5_failure_audit/kappa_corpus_defect.py)
— run this **after your sheet is locked** (§2.6.3).

```bash
.venv/bin/python code/ex5_failure_audit/kappa_corpus_defect.py \
    --out results/ex5_failure_audit/corpus_defect_kappa.json
```

The κ estimator is `compute_kappa.py:42-66` reused verbatim with `LABELS` swapped to
`("MISMATCH", "ON_TOPIC")`. **No gate is applied** — `KAPPA_GATE = 0.70` is pre-registered for
Task A only.

**Defaults match the decided configuration**, so no arguments are required:

| Flag | Default |
|---|---|
| `--annotator-a` | `handcheck` — reads annotator 1 from the committed artifact, so no second file of verdicts is ever created |
| `--handcheck` | `results/ex5_failure_audit/passage_overlap_handcheck.json` |
| `--annotator-b` | `results/ex5_failure_audit/corpus_defect_sheet_sciq33.csv` |
| `--subset` | `results/ex5_failure_audit/corpus_defect_comparison_subset.json`; pass `all` to score every shared item |
| `--out` | none — pass a path to write the JSON report |

Category collapse (`CATEGORY_TO_LABEL`, lines 71–76), matching §2.2: `topic_mismatch` →
`MISMATCH`; `legitimate_low_overlap` / `orthographic_defect` / `other` → `ON_TOPIC`.

Prints n, observed and expected agreement, κ, the 2×2 confusion table, and the disagreeing
`item_id`s; writes the same plus a `_provenance` block recording both annotators, the subset
source, the blind protocol, the absence of a gate, and the small-n / one-LLM-rater caveats.

**Before you are done it is a progress check, not a spoiler:** on a blank or partial sheet it
prints the outstanding `item_id`s and exits 1 without scoring anything, so it reveals none of
annotator 1's verdicts.

---

## Sources, in one place

| Rule / fact | Source |
|---|---|
| Task A criterion (3 rules) | `results/ex4_free_response/validation_sheet_pilot_bidir.csv:3-5` · `docs/ex4_free_response/plan_option_free_response_experiment.md:203-207` |
| Task A label set | `code/ex4_free_response/compute_kappa.py:39` |
| Task A accepted aliases | `code/ex4_free_response/compute_kappa.py:76-82` |
| Task A gate = 0.70, pre-registered | `code/ex4_free_response/compute_kappa.py:38` |
| Task A two κ mechanisms | `code/ex4_free_response/compute_kappa.py:108-119` |
| Task A canonical output path | `docs/ex4_free_response/plan_option_free_response_experiment.md:584-587` |
| Task A: 74 rows, 7 strata, agent must not label | `docs/NEXT_PHASE_HANDOFF.md` §4.1–4.2 |
| Task B binary question | `docs/ex5_failure_audit/corpus_defect_audit.md:276-277` |
| Task B positive-class definition | `docs/ex5_failure_audit/corpus_defect_audit.md:139` |
| Task B four source categories | `results/ex5_failure_audit/passage_overlap_handcheck.json` → `categories` |
| Task B out-of-scope class (on topic, unsupported answer) | `docs/ex5_failure_audit/corpus_defect_audit.md:299-301` |
| Task B mis-key estimation dropped | `docs/ex5_failure_audit/corpus_defect_audit.md:260-265` |
| Task B 4-step plan, "double-annotate 15 of the 33" | `docs/ex5_failure_audit/corpus_defect_audit.md:267-278` |
| Task B annotator config / the 15 / blind — **decided 2026-09-08** | this guide §2.6 · `results/ex5_failure_audit/corpus_defect_comparison_subset.json` → `_provenance` · `docs/NEXT_PHASE_HANDOFF.md` §2.4–2.5 |
| Task B accepted aliases | `code/ex5_failure_audit/kappa_corpus_defect.py:64-68` |
| Task B category collapse (4 → 2) | `code/ex5_failure_audit/kappa_corpus_defect.py:71-76` |
| Task B union screen = 33 items | `docs/ex5_failure_audit/corpus_defect_audit.md:243` · `docs/NEXT_PHASE_HANDOFF.md` §2.2–2.3 |
| Task B second annotator must include a human | `docs/NEXT_PHASE_HANDOFF.md` §2.5, §5 |
| κ formula reused | `code/ex4_free_response/compute_kappa.py:42-66` |
| Seed convention 20260904 | `code/ex5_failure_audit/passage_overlap_audit.py:193` · `code/ex4_free_response/build_validation_sample.py:149` |
