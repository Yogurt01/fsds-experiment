# Annotation Guide — RQ2 independent label set

**Everything you need in one place.** Two sheets, two separate passes. Read §1, then start.

| | |
|---|---|
| Sheet 1 | [`results/ex8_independent_labels/q1_answerability_sheet.csv`](../../results/ex8_independent_labels/q1_answerability_sheet.csv) — 100 rows, fill `rating` |
| Sheet 2 | [`results/ex8_independent_labels/q2_context_following_sheet.csv`](../../results/ex8_independent_labels/q2_context_following_sheet.csv) — 55 rows, fill `stated_answer` |
| Scoring | `kappa_independent_labels.py` (agreement), then `compare_metrics.py` (the comparison) |

**Why this exists.** Every metric comparison so far has been judged against labels derived from the
solvers' own Setting-C behaviour, and those labels turn out to be near-tautologically determined by
the thing being measured. These sheets are the independent instrument: judgements made by a person
reading only the question, options and passage, with no model output, no score, and no earlier label
anywhere in view. Nothing in either sheet reveals any metric.

---

## 1. Order of work, and what to drop if time runs short

**Work top to bottom. Do not reorder rows, and do not sort the file.** Priority is already encoded
in the row order, so simply working downward spends your time in the right place.

| Sheet 1 rows | What they are | Budget @ 2.5 min/row | If you stop here |
|---|---|---|---|
| **1–30** | Double-annotation block | ~75 min | **Do not skip.** These are the agreement check |
| **31–69** | Highest-value comparison items | ~98 min | Primary test still works, slightly reduced power |
| **70–86** | Supporting comparison items | ~43 min | Primary test loses power |
| **87–100** | Calibration only | ~35 min | **Safe to drop.** Costs only a secondary estimate |

**Sheet 2 (55 rows, ~1.5 min/row ≈ 82 min)** covers the same high-value items as rows 31–69 plus
their share of the double block. Its first 16 rows are the double-annotation block.

**Suggested sequence:** Sheet 1 rows 1–30 → Sheet 1 rows 31–86 → **Sheet 2 (separate day)** →
Sheet 1 rows 87–100 if time remains.

**Full budget ≈ 7.9 h.** Dropping rows 87–100 brings it to ≈ 7.0 h; also stopping Sheet 1 at row 69
brings it to ≈ 6.3 h. You do not have to decide now — just stop when you need to.

> **Do rows 1–30 of Sheet 1 first and in one sitting.** If agreement on that block turns out poor,
> the rubric is at fault and the rest of the time would be wasted. A second rater can score that
> block early and you will know within ~2.5 h rather than at the end.

---

## 2. Sheet 1 — answerability without the passage

You see: question, four options, the gold answer, and the passage. Fill `rating`.

> **Imagine a typical secondary-school student who has studied general science but has NOT studied
> this particular topic, and who has NOT read the passage.** They see only the question and the four
> options. **How likely are they to pick the correct answer?**
>
> | | |
> |---|---|
> | **1** | **Very unlikely** — they would have to guess. Nothing in the question or options points to the answer; the passage is doing all the work |
> | **2** | **Unlikely** — they might narrow it to two options, but could not reliably choose |
> | **3** | **Likely** — general science knowledge or a clue in the wording gets them there most of the time |
> | **4** | **Very likely** — obvious from the question and options alone: common knowledge, a giveaway word, or the other three are plainly wrong |

**There is no middle option on purpose.** Commit to a side. The hard cases are the ones that matter
most, and a neutral box would absorb exactly those.

> **AI tools, added 2026-09-24.** **Never** use ChatGPT, Claude, Copilot or any other AI tool to
> suggest, draft, rank or check a rating or a `stated_answer`. Every judgement must be your own.
> **Translation is the one permitted use, and it must be declared**: if you need the guide or an
> item rendered into another language to read it, that is fine, but tell whoever is running the
> study which tool you used, and **read the original English alongside the translation**. Also say
> so if you draft your `notes` in another language and translate them.
>
> *Why this was added: on the first run of this protocol both annotators used AI translation of the
> guide and the item text. Neither broke a stated rule — the guide only said "no model output … in
> view" and "do not look anything up", which did not clearly cover translation. Their judgements
> were their own, so the harm was limited, but two problems remained. Machine translation can repair
> text that was corrupted on purpose, which is exactly what Sheet 2 asks you to read literally; and
> it can add or remove the "giveaway word" that Sheet 1's scale asks you to judge. If both
> annotators use the same tool, those distortions are shared rather than independent, which inflates
> the agreement figure. Reading the English alongside is what keeps this from mattering.*

**Three things not to judge.** Whether the keyed answer is factually correct. Whether the passage is
well written. Whether **you** know the answer — you are a bad model of the imagined student, because
you have read a lot of these. Ask what *they* would do.

**`unusable`.** Put `X` and leave `rating` blank when the row cannot be judged: the passage is about
an unrelated topic, the question is incoherent, or two options are identical. Do not rate it anyway.
Expect a small number — roughly 1% of SciQ passages are known to be off-topic.

### Practice examples — fabricated, not from the sheet

| Question | Options | Rating | Why |
|---|---|---|---|
| *What gas do plants absorb during photosynthesis?* | oxygen · **carbon dioxide** · helium · argon | **4** | School-curriculum fact; the distractors are not plausible plant gases |
| *Digestive enzymes are secreted by the organs of which body system?* | nervous · endocrine · urinary · **digestive** | **4** | The stem contains "digestive"; the option repeats it. A giveaway, whether or not the student knows any physiology |
| *What are the hormones that cause a plant to grow?* | **gibberellins** · pistills · pores · sporozoans | **2** | Specialist term from an undergraduate syllabus. The student might eliminate two, but could not reliably land on it |
| *Compounds capable of accepting electrons are called what?* | oxygen · antioxidants · residues · **oxidants** | **1** | "Oxidants" vs "antioxidants" is a coin flip without the passage; the surface offers no help |

### Boundary rule that actually bites

When you are torn between **2** and **3**, ask: *would they get it right more often than not?* Yes →
3. No → 2. That boundary is the one the analysis collapses on, so spend your uncertainty there rather
than at the 1/2 or 3/4 edges.

---

## 3. Sheet 2 — what does this passage say?

You see: question, four options, a passage. **No gold answer** — by design. Fill `stated_answer`.

> **Which of the four options does THIS PASSAGE state or clearly imply is the answer?**
> `a` / `b` / `c` / `d`, or `none` if the passage does not clearly point to any of them.

**Answer only from the passage in front of you.** Do not use outside knowledge. Do not judge whether
the passage is factually true — **several of these passages have been edited on purpose, and some
now say things that are false.** That is expected and is exactly what is being tested. A passage
saying *"hydrogen fuel cells provide gravity for manned space vehicles"* should be answered
`gravity`, not `electricity`.

### Why the separate pass matters

Sheet 2 shows edited versions of passages you may have seen in Sheet 1. If you recognise an item and
answer *"they changed X to Y"* rather than *"this passage says Y"*, the measurement is spoiled. The
sheets already use different row orders and Sheet 2 withholds the gold answer, but that only reduces
recognition — it does not remove it. So:

- **Do Sheet 2 on a different day from Sheet 1.** Not the same sitting.
- Do not open Sheet 1 while working on Sheet 2, and do not look anything up.
- If you clearly remember an item, answer from the text in front of you anyway and note `recalled`
  in the `notes` column so it can be checked later.

> **If a second annotator becomes available, the clean fix is to give Sheet 2 to them instead** —
> splitting by person rather than by day removes carryover entirely. Worth doing if you can; the
> design does not require it.

---

## 4. Filling the sheets

- Open in a spreadsheet or a text editor; **keep it as CSV**, do not convert to `.xlsx`.
- The `#` comment lines at the top are part of the file. Leave them. Both scripts skip them.
- Do not add, delete, reorder or re-sort rows. Do not edit any column except `rating`,
  `unusable`, `stated_answer` and `notes`.
- Save each annotator's copy under its own filename, e.g. `q1_answerability_sheet.annotator1.csv`.
- `notes` is free text and is never scored — use it for anything you want revisited.

**When both sheets are filled:**

```bash
python code/ex8_independent_labels/kappa_independent_labels.py --q1-a <a.csv> --q1-b <b.csv>
```

That reports agreement on the double block against the pre-registered gate of **κ ≥ 0.70**. Run it
before the comparison. The comparison itself is one command and is already written and tested:

```bash
python code/ex8_independent_labels/compare_metrics.py --q1 <a.csv> --q1-b <b.csv> --q2 <q2.csv> --out results/ex8_independent_labels/metric_comparison.json
```

---

## 5. What is deliberately hidden, and why

Neither sheet carries a score, a Setting-C class, a contingency bucket, a substitution tier or a
stratum name. Stratum is withheld specifically because it is a deterministic function of the two
metrics being compared, so printing it would leak both rankings — which is why priority appears as
row order instead of a column. The gold answer is shown on Sheet 1 (the rating is unanswerable
without knowing which option is correct) and withheld on Sheet 2 (where it would give away the
edit). Row orders differ between sheets so the two cannot be paired by eye.

Every one of those choices is recorded in `annotation_manifest.json`, and
`annotation_sheets.LOCK.json` carries the SHA-256 of both sheets as issued, so a filled sheet can be
proved to descend from this issue.
