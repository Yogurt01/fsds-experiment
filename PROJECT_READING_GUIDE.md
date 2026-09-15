# Project Reading Guide

**Purpose:** a single navigation document. Read the numbered pipeline in §2 top to bottom and you
will understand what this project asks, what it has established, and what is still open — without
needing to have been present for any of it.

**Status:** written 2026-09-08, immediately after the filesystem reorganization recorded in §5.
Paths in this guide are the *current* ones.

> This guide does not replace [`README.md`](README.md). README states the research argument;
> this document states the *reading order* and the *provenance* of every artifact, and records
> where the live repository has diverged from README's description (§4).

---

## 1. What this project is

Automatic question generation needs an automatic evaluator that can tell whether a generated
multiple-choice question genuinely *requires* the learning material it was generated from. The KDA
metric (Moon et al., EMNLP 2022) claims to measure exactly that, by comparing how well a simulated
student answers with and without the target fact. This project tests whether that claim survives
contact with modern solvers — and finds that it largely does not: a large share of the "the fact
made this answerable" signal is produced by the solver's parametric prior rather than by its
reading of the material. The work then builds and evaluates an intervention — a **counterfactual
context perturbation** that rewrites the passage to assert a distractor, so that only a solver
actually reading the context can earn credit — plus a series of follow-up experiments probing
whether the underlying measurement problem can be fixed by other means.

**Three research questions** (README §1.3), and **five build-order experiments**. These are two
independent numbering systems — `exN` counts the order in which experiments were built, not which
RQ they serve. One RQ can span several experiments; one experiment can inform several RQs.

| | |
|---|---|
| **RQ1** — *Do modern LLM-based evaluators reliably measure knowledge dependency?* | the finding |
| **RQ2** — *How do we formulate a metric that separates context reliance from prior?* | the method |
| **RQ3** — *Does the metric improve the quizzes that reach students?* | downstream validation (not yet built) |

---

## 2. The reading pipeline

Open these in order. Each step says why it comes where it does.

> **Where the documents live.** `docs/` is grouped by role: `papers/` (the three source-paper
> walkthroughs), `ex1_reproduce_KDA/` … `ex5_failure_audit/` (mirroring `code/`'s names exactly),
> `guides/` (operator runbooks), `synthesis/` (cross-cutting summaries), and
> `NEXT_PHASE_HANDOFF.md` alone at the top level. Full map and rationale: §5.7.

### Stage A — Theory: what is being measured, and by whom

**1.** [`README.md`](README.md) §1–§2 — the research story and the KDA formulations
($KDA_{disc}$, $KDA_{cont}$, and the theoretical gap in §2.4 that motivates everything after).
*Start here: every later document assumes the §2.4 vocabulary.*

**2.** [`paper_references/2022.emnlp-main.718.pdf`](paper_references/2022.emnlp-main.718.pdf) →
[`docs/papers/KDA_Paper_Documentation.md`](docs/papers/KDA_Paper_Documentation.md) — the metric being tested.
*Read the documentation if short on time; it is a full walkthrough.* Note §6.2–§6.3 of the original
paper, where the authors flag "too easy questions" and the difficulty of measuring PLM ignorance —
this project's RQ1 is that caveat becoming the dominant term.

**3.** [`docs/papers/ClashEval_Paper_Documentation.md`](docs/papers/ClashEval_Paper_Documentation.md)
(paper: [`paper_references/NeurIPS-2024-clasheval-…pdf`](paper_references/NeurIPS-2024-clasheval-quantifying-the-tug-of-war-between-an-llms-internal-prior-and-external-evidence-Paper-Datasets_and_Benchmarks_Track.pdf))
— why prior-vs-context arbitration is a measurement problem rather than a data-cleaning step.
*This is the direct intellectual source of Experiment 2.*

**4.** [`docs/papers/QG-SMS_Paper_Documentation.md`](docs/papers/QG-SMS_Paper_Documentation.md)
(paper: [`paper_references/2025.acl-long.1268.pdf`](paper_references/2025.acl-long.1268.pdf))
— student modeling and item analysis. *Source of Experiment 3's persona design and of the planned
RQ3 validation.*

**5.** [`README.md`](README.md) §3–§4 — how the three papers position this work, and the current
directory map. *Read after the papers so the positioning argument lands.*

### Stage B — Infrastructure: how data and models get here

**6.** [`code/utils/paths.py`](code/utils/paths.py) — 62 lines. `PROJECT_ROOT` anchoring and
`resolve()`. *Read first among code: 15 of the other 20 scripts import it, and it is why every `--data` /
`--out` argument in this repo is working-directory-independent.*

**7.** [`code/pre_data/prepare_sciq.py`](code/pre_data/prepare_sciq.py) and
[`code/pre_data/prepare_openbookqa.py`](code/pre_data/prepare_openbookqa.py) → outputs in
[`datasets/`](datasets), logs
[`prep_sciq.log`](results/ex1_reproduce_KDA_pipeline/prep_sciq.log) /
[`prep_openbookqa.log`](results/ex1_reproduce_KDA_pipeline/prep_openbookqa.log).
*The KDA input format is defined here; every experiment consumes it.* Note the OBQA `fact1` join
across the `main` and `additional` configs.

**8.** [`code/pre_data/download_models.py`](code/pre_data/download_models.py) →
[`model_download_report.json`](results/ex1_reproduce_KDA_w_modernLLM/model_download_report.json).
*Only needed before any Qwen-based stage (steps 14, 20, 24).*

### Stage C — Experiment 1: the baseline reproduces, and shows the symptom

**9.** [`code/ex1_reproduce_KDA/kda_tiny.py`](code/ex1_reproduce_KDA/kda_tiny.py) — the
$KDA_{cont}$ ensemble implementation. *The mathematical core. Read the module docstring's
explanation of why $|M| \ge 2$ is mandatory before anything else in this stage.*

**10.** [`code/ex1_reproduce_KDA/run_experiment.py`](code/ex1_reproduce_KDA/run_experiment.py) —
the Setting A/B runner.

**11.** [`REPRODUCE.md`](REPRODUCE.md) — the full step-by-step reproduction, and the *only*
document that analyses the two early exploratory runs (§776 the 2-model 884-question run, §877 the
50-question run). *Skim §5–§9; read §636 "Findings" closely.*

**12.** [`docs/ex1_reproduce_KDA/kda_reproduction_summary.md`](docs/ex1_reproduce_KDA/kda_reproduction_summary.md) — the headline
baseline comparison, SciQ vs OpenBookQA, on the official 4-model `KDA_small` suite.
Backing data:
[`sciq/results_kda_small_sciq_test_full.json`](results/ex1_reproduce_KDA_pipeline/sciq/results_kda_small_sciq_test_full.json)
and [`openbookqa/results_kda_small_obqa_test_full.json`](results/ex1_reproduce_KDA_pipeline/openbookqa/results_kda_small_obqa_test_full.json).

**13.** [`code/ex1_reproduce_KDA/categorize_kda_results.py`](code/ex1_reproduce_KDA/categorize_kda_results.py)
→ [`results/ex1_category_questions/basic_category/`](results/ex1_category_questions/basic_category).
*The four-bucket contingency table (`both_correct` / `wrong_to_correct` / `both_wrong` /
`correct_to_wrong`) is the vocabulary every later experiment uses. `both_correct` is the bucket
Experiment 2 targets.*

**14.** [`code/ex1_reproduce_KDA/kda_qwen_eval.py`](code/ex1_reproduce_KDA/kda_qwen_eval.py) →
[`docs/ex1_reproduce_KDA/kda_qwen3_4b_evaluation_report.md`](docs/ex1_reproduce_KDA/kda_qwen3_4b_evaluation_report.md).
*This is where RQ1 becomes undeniable: Qwen3-4B answers 95.4% of SciQ with no passage at all,
leaving $KDA_{disc}$ computed from 41 of 884 questions.* Read this before Experiment 2 — it is
what makes the intervention necessary.

### Stage D — Experiment 2: the proposed method (RQ2)

**15.** [`code/ex2_counterfactual/counterfactual_passage.py`](code/ex2_counterfactual/counterfactual_passage.py)
— 257 lines, no ML. The minimal lexical answer→distractor rewrite and its four matching tiers
(`exact` / `glued` / `morphological` / `partial`). *Read the docstring's worked example first.*

**16.** [`code/ex2_counterfactual/run_counterfactual_experiment.py`](code/ex2_counterfactual/run_counterfactual_experiment.py)
— the Setting A/B/C sweep and the three adjusted estimators.

**17.** [`docs/ex2_counterfactual/counterfactual_experiment_methodology.md`](docs/ex2_counterfactual/counterfactual_experiment_methodology.md)
— the reviewer-facing walkthrough: how passages were generated, scored, and classified into
`context_dependent` / `prior_dependent` / `unstable_other`. *Read before the results.*

**18.** [`docs/ex2_counterfactual/counterfactual_obqa_analysis.md`](docs/ex2_counterfactual/counterfactual_obqa_analysis.md) — the
results, SciQ vs OpenBookQA head to head. Backing data:
[`results_counterfactual_sciq_test_full.json`](results/ex2_counterfactual/results_counterfactual_sciq_test_full.json),
[`results_counterfactual_obqa_test_full.json`](results/ex2_counterfactual/results_counterfactual_obqa_test_full.json),
[`results_counterfactual_obqa_test_exact_tier.json`](results/ex2_counterfactual/results_counterfactual_obqa_test_exact_tier.json).
*The key negative result is here: OBQA's one-clause deductive facts do not support the lexical
rewrite at scale (145 of 500 eligible at `partial`, 42 at `exact`).*

### Stage E — Experiment 3: can prompting fix the saturation instead?

**19.** [`code/ex3_student_simulation/personas.py`](code/ex3_student_simulation/personas.py) —
three ability tiers, written separately per dataset, with named failure mechanisms.
*Read the docstring's argument for why "be worse" is not a usable instruction.*

**20.** [`code/ex3_student_simulation/run_student_simulation.py`](code/ex3_student_simulation/run_student_simulation.py)
and [`summarize_simulation.py`](code/ex3_student_simulation/summarize_simulation.py) — the JOINT
(contrastive) vs ISOLATED (roleplay) paradigms.

**21.** [`docs/ex3_student_simulation/student_persona_simulation_report.md`](docs/ex3_student_simulation/student_persona_simulation_report.md) —
*the answer is essentially "no": persona conditioning does not manufacture a usable ability spread.
Read this as a negative result that closes a branch, not as a component of the main method.*

### Stage F — Experiment 4: is the closed option set the real problem?

**22.** [`docs/ex4_free_response/plan_option_free_response_experiment.md`](docs/ex4_free_response/plan_option_free_response_experiment.md)
§1–§7 — the plan, written before the code. *Read the plan before the harness; §4.2 argues the
three-stage matching cascade against its alternatives.*

**23.** [`code/ex4_free_response/matching.py`](code/ex4_free_response/matching.py) — normalised
exact → morphological → LLM judge, with the deciding stage recorded per item.

**24.** [`code/ex4_free_response/run_free_response.py`](code/ex4_free_response/run_free_response.py)
— cells A′ (no options, no fact) and B′ (no options, fact present). *The first stage in this repo
that asks the model to **recall** rather than **recognise**; everything before it has a 0.25 chance
floor.*

**25.** [`docs/ex4_free_response/plan_option_free_response_experiment.md`](docs/ex4_free_response/plan_option_free_response_experiment.md)
§8–§9 — the two pilots and what they revealed: the judge decides 76–100% of items and flips 24.3%
of verdicts under reference/student swap. *This is why the experiment is blocked.*

**26.** [`code/ex4_free_response/build_validation_sample.py`](code/ex4_free_response/build_validation_sample.py)
→ [`validation_sheet_pilot_bidir.csv`](results/ex4_free_response/validation_sheet_pilot_bidir.csv),
then [`compute_kappa.py`](code/ex4_free_response/compute_kappa.py). *The κ ≥ 0.70 gate is
pre-registered in code, before the human labels exist. The sheet is built and waiting.*

### Stage G — Experiment 5: auditing what Experiments 1–2 left on the table

**27.** [`docs/ex5_failure_audit/rq1_test_split_failure_analysis.md`](docs/ex5_failure_audit/rq1_test_split_failure_analysis.md) — the
manual failure taxonomy over 1,384 test items, and the artifact it produced,
[`rq1_flagged_questions.json`](results/ex5_failure_analysis/rq1_flagged_questions.json) (408 records).

**28.** [`docs/ex5_failure_audit/provenance_rq1_flagged_questions.md`](docs/ex5_failure_audit/provenance_rq1_flagged_questions.md) —
**read immediately after §27.** That artifact's generator never existed; this document establishes
it by eight independent searches, and records the reimplementation that now re-derives it exactly.

**29.** [`code/ex5_failure_audit/rebuild_flagged_pool.py`](code/ex5_failure_audit/rebuild_flagged_pool.py)
— the restored generator. Run `--verify` to reproduce the EXACT MATCH claim yourself:

```bash
.venv/bin/python code/ex5_failure_audit/rebuild_flagged_pool.py --dataset sciq --verify
```

**30.** [`docs/ex5_failure_audit/failure_taxonomy_methodology.md`](docs/ex5_failure_audit/failure_taxonomy_methodology.md) — the
reconstructed labelling rubric, and §6.1's honest treatment of the annotator effect introduced by
[`obqa_new_admission_labels.json`](results/ex5_failure_analysis/obqa_new_admission_labels.json).

**31.** [`code/ex5_failure_audit/bucket_diagnostics.py`](code/ex5_failure_audit/bucket_diagnostics.py)
→ [`docs/ex5_failure_audit/unexploited_buckets_analysis.md`](docs/ex5_failure_audit/unexploited_buckets_analysis.md) — what is in
`both_wrong` and `correct_to_wrong`, the two buckets no report had analysed.

**32.** [`code/ex5_failure_audit/passage_overlap_audit.py`](code/ex5_failure_audit/passage_overlap_audit.py)
→ [`docs/ex5_failure_audit/corpus_defect_audit.md`](docs/ex5_failure_audit/corpus_defect_audit.md) — a benchmark-wide screen for
questions whose passage does not match them. *A corpus defect no metric can correct; it began as a
five-item observation inside step 31.*

### Stage H — Synthesis and open work

**33.** [`docs/synthesis/rq1_research_synthesis.md`](docs/synthesis/rq1_research_synthesis.md) — **the capstone.**
Cross-checks five reports field-for-field against their source JSON. *If you read only one
analysis document, read this — but read it last, because it assumes all of them.*

**34.** [`RUN_QWEN2.5.md`](RUN_QWEN2.5.md) — the status board *and* the execution guide for
everything blocked on cloud GPU, merged into one file on 2026-09-11 (§5.8). §0 is a four-entry
checklist; entry 3 is the only item on the critical path. §1 is the shared setup every entry needs
— including the **4-bit NF4 precision policy** that keeps the cloud numbers comparable with the
local run. Appendix A carries Entry 1's script inline for zero-repository-access use;
[`code/ex1_reproduce_KDA/run_qwen7b_eval.py`](code/ex1_reproduce_KDA/run_qwen7b_eval.py) is its
canonical form.

**35.** [`README.md`](README.md) §6.5–§6.6 — the planned RQ3 downstream validation, and the known
caveats that constrain every number above.

**36.** §6 of this guide — Open Threads.

---

## 3. Provenance map

One row per artifact. **Status** values: `done` (produced and analysed) · `partial` (produced, not
fully analysed, or superseded) · `planned` (referenced, not produced) · `undocumented in README`
(exists and is sound, but README does not mention it).

### Papers

| # | File | Type | Reads | Produces / Analyses | RQ | Status |
|---|---|---|---|---|---|---|
| 2 | `paper_references/2022.emnlp-main.718.pdf` | paper | — | the metric under test | RQ1, RQ2 | done |
| 3 | `paper_references/NeurIPS-2024-clasheval-…pdf` | paper | — | motivates Setting C | RQ2 | done |
| 4 | `paper_references/2025.acl-long.1268.pdf` | paper | — | motivates ex3 + RQ3 design | RQ3 | done |
| 2 | `docs/papers/KDA_Paper_Documentation.md` | docs | KDA paper | documents it | RQ1, RQ2 | done |
| 3 | `docs/papers/ClashEval_Paper_Documentation.md` | docs | ClashEval paper | documents it | RQ2 | done |
| 4 | `docs/papers/QG-SMS_Paper_Documentation.md` | docs | QG-SMS paper | documents it | RQ3 | done |

### Shared infrastructure

| # | File | Type | Reads | Produces / Analyses | RQ | Status |
|---|---|---|---|---|---|---|
| 6 | `code/utils/paths.py` | code | — | library: `PROJECT_ROOT`, `resolve()`, `ensure_parent()` | all | done |
| 7 | `code/pre_data/prepare_sciq.py` | code | HF `allenai/sciq` | `datasets/sciq/*.json` + `prep_sciq.log` | all | done |
| 7 | `code/pre_data/prepare_openbookqa.py` | code | HF `allenai/openbookqa` (main + additional) | `datasets/openbookqa/*.json` + `prep_openbookqa.log` | all | done |
| 8 | `code/pre_data/download_models.py` | code | HF Hub | `models/`, `model_download_report.json`, `model_download.log` | — | done |
| 7 | `datasets/sciq/sciq_test_full.json` (884 q) | data | prepare_sciq | primary SciQ corpus | all | done |
| 7 | `datasets/sciq/sciq_50.json` | data | prepare_sciq | smoke subset, step 11 | RQ1 | done |
| 7 | `datasets/sciq/{sciq_train_full,sciq_val_full,sciq_all_combined}.json` | data | prepare_sciq | exported, never consumed by any experiment | — | undocumented in README |
| 7 | `datasets/openbookqa/obqa_test_full.json` (500 q) | data | prepare_openbookqa | primary OBQA corpus | all | done |
| 7 | `datasets/openbookqa/{obqa_train_full,obqa_val_full,obqa_all_combined,obqa_50}.json` | data | prepare_openbookqa | exported, never consumed | — | undocumented in README |
| 8 | `results/ex1_reproduce_KDA_w_modernLLM/model_download_report.json` | results | download_models | SHA-256 / git-blob verification of every shard | — | done |
| 8 | `results/ex1_reproduce_KDA_w_modernLLM/model_download.log` | results | download_models `--verify-only` | verification-pass log (no transfer) | — | partial — no doc cites it |

### Experiment 1 — KDA baseline (+ 1b, modern LLM)

| # | File | Type | Reads | Produces / Analyses | RQ | Status |
|---|---|---|---|---|---|---|
| 9 | `code/ex1_reproduce_KDA/kda_tiny.py` | code | — | library: `KDATiny`, `make_student`, `KDA_SMALL` | RQ1 | done |
| 10 | `code/ex1_reproduce_KDA/run_experiment.py` | code | any prepared dataset | Setting A/B results + log | RQ1 | done |
| 11 | `REPRODUCE.md` | docs | — | full reproduction; sole analysis of the two early runs | RQ1 | done |
| 12 | `docs/ex1_reproduce_KDA/kda_reproduction_summary.md` | docs | the 4-model runs | SciQ vs OBQA baseline comparison | RQ1 | done |
| 11 | `…/sciq/results_kda_tiny2_sciq_test_sample50.json` + `.log` | results | run_experiment | SciQ test, n=50, **2-model: distilbert + bert-base-uncased**, 176.4M | RQ1 | partial — REPRODUCE.md §877 only |
| 11 | `…/sciq/results_kda_tiny2_sciq_test_full.json` + `.log` | results | run_experiment | SciQ test, n=884, **2-model: distilbert + scibert**, 176.9M — *a different pair from the row above* | RQ1 | partial — REPRODUCE.md §776 only |
| 12 | `…/sciq/results_kda_small_sciq_test_full.json` + `.log` | results | run_experiment | SciQ test, n=884, 4-model `KDA_small`, 355.1M — **the primary SciQ baseline** | RQ1 | done |
| 12 | `…/openbookqa/results_kda_small_obqa_test_full.json` + `.log` | results | run_experiment | OBQA test, n=500, 4-model `KDA_small` | RQ1 | done |
| 12 | `…/openbookqa/console_kda_small_obqa_test_full.txt` | results | `tee` of the above | console capture; embeds pre-reorg paths (historical) | RQ1 | partial |
| 7 | `…/prep_sciq.log`, `…/prep_openbookqa.log` | results | prepare_* | dataset preparation logs | all | done |
| 13 | `code/ex1_reproduce_KDA/categorize_kda_results.py` | code | auto-detects the 3 SciQ result files | the four-bucket split | RQ1 | done |
| 13 | `…/basic_category/categorized_summary.json` | results | categorize | cross-model manifest + `output_files` map | RQ1 | done |
| 13 | `…/basic_category/categorized_{t5_small_ssm_nq, kda_albert_xlarge_v2_race, kda_mpnet_base_race, kda_scibert_uncased_race, ensemble_pooled}.json` | results | categorize | per-model bucket breakdown, n=884 | RQ1 | partial — only `categorized_summary` is cited by a doc |
| 13 | `…/basic_category/categorized_mpnet_base.json` | results | categorize (line 419) | **intentional duplicate** of the primary model's file under a fixed name; MD5-identical, deliberately absent from `output_files` | RQ1 | done |
| 13 | `…/basic_category/categorize_kda_results.log` | results | categorize | execution log | RQ1 | done |
| 13 | `…/complicated_category/README.md` | docs | — | self-declared placeholder; directory holds no data | RQ1 | planned |
| 14 | `code/ex1_reproduce_KDA/kda_qwen_eval.py` | code | both test splits | Qwen3-4B KDA results + log | RQ1 | done |
| 14 | `…/ex1_reproduce_KDA_w_modernLLM/kda_qwen3_4b_results.json` + `kda_qwen3_4b_eval.log` | results | kda_qwen_eval | RQ1 at 4B scale — the saturation result | RQ1 | done |
| 14 | `docs/ex1_reproduce_KDA/kda_qwen3_4b_evaluation_report.md` | docs | the above | analyses it | RQ1 | done |
| 34 | `code/ex1_reproduce_KDA/run_qwen7b_eval.py` | code | builds own data | **self-contained**, zero repo imports; for Colab/Kaggle | RQ1 | planned — not yet run |
| 34 | `RUN_QWEN2.5.md` §2 + Appendix A | docs | — | Colab/Kaggle walkthrough for the above, plus the script inline | RQ1 | done (guide); target run planned |

### Experiment 2 — counterfactual perturbation

| # | File | Type | Reads | Produces / Analyses | RQ | Status |
|---|---|---|---|---|---|---|
| 15 | `code/ex2_counterfactual/counterfactual_passage.py` | code | — | library: `build_counterfactual`, `TIERS`, `morphological_variants` | RQ2 | done |
| 16 | `code/ex2_counterfactual/run_counterfactual_experiment.py` | code | a dataset + `ex1.kda_tiny` | Settings A/B/C + adjusted KDA | RQ2 | done |
| 18 | `…/results_counterfactual_sciq_test_full.json` + `counterfactual_sciq_test_full.log` | results | the above | SciQ 884, tier `partial`, 860 eligible — **the primary RQ2 result** | RQ1, RQ2 | done |
| 18 | `…/results_counterfactual_obqa_test_full.json` + `counterfactual_obqa_test_full.log` | results | the above | OBQA 500, tier `partial`, 145 eligible | RQ2 | done |
| 18 | `…/results_counterfactual_obqa_test_exact_tier.json` + `counterfactual_obqa_test_exact_tier.log` | results | the above | OBQA 500, tier `exact`, 42 eligible, 0 glued residuals | RQ2 | done |
| 17 | `…/counterfactual_passages_preview_sciq.json` | results | `--dry-run` | 884 samples, 860 eligible; *filename predates the current derivation rule* | RQ2 | done |
| 18 | `…/counterfactual_passages_preview_obqa.json` + `counterfactual_obqa_dryrun.log` | results | `--dry-run` | 500 samples, 145 eligible | RQ2 | done |
| 17 | `docs/ex2_counterfactual/counterfactual_experiment_methodology.md` | docs | the runs above | reviewer-facing methodology walkthrough | RQ2 | done |
| 18 | `docs/ex2_counterfactual/counterfactual_obqa_analysis.md` | docs | the runs above | SciQ vs OBQA results | RQ1, RQ2 | done |
| 18a | `code/ex2_counterfactual/run_counterfactual_qwen.py` | code | both test splits + Qwen3-4B + the ex2 generator | **E1**: Setting C with a saturated LLM solver; 2 correctness gates against ex1b and the committed preview | RQ2 | done |
| 18a | `…/ex2_counterfactual/results_counterfactual_qwen3_4b.json` + `.log` | results | the above | SciQ 884, target set 819 | RQ1, RQ2 | done |
| 18a | `…/ex2_counterfactual/counterfactual_passages_for_cloud.json` | results | the above | standalone passages so the self-contained Kaggle script can run Setting C | RQ2 | done |
| 18a | `docs/ex2_counterfactual/e1_counterfactual_llm_scale.md` | docs | the above | **E1 write-up**: the intervention survives saturation; κ = 0.046 across solvers | RQ2, RQ3 | done |
| 18b | `code/ex2_counterfactual/run_prior_vs_rejection.py` | code | the ex2 generator + E1's records + Qwen3-4B | **E2**: instruction ablation (Setting B'/C') + intrinsic plausibility gradient; gated against E1 | RQ2 | done |
| 18b | `…/ex2_counterfactual/results_prior_vs_rejection_qwen3_4b.json` + `.log` | results | the above | brackets genuine prior-dependence at 46.0%–62.5% | RQ2 | done |
| 18b | `docs/ex2_counterfactual/e2_prior_vs_rejection.md` | docs | the above | **E2 write-up**: the rejection hypothesis does not hold up | RQ2 | done |
| 16a | `code/ex2_counterfactual/recompute_adjusted_kda.py` | code | the 3 committed runs, no model | re-derives the adjusted-KDA block with **both readings** of $KDA_{adj}^{excl}$ + a 9-way $\mathcal{Q}_{ctx}$ sweep; verifies every pre-existing figure before writing | RQ2 | done |
| 16a | `…/ex2_counterfactual/adjusted_kda_corrected.json` | results | the above | the corrected exclusion figures; inputs unmodified | RQ2 | done |
| 16b | `code/ex2_counterfactual/unstable_other_conventions.py` | code | the 3 committed runs, no model | all four estimators under strict / lenient / abstain | RQ2 | done |
| 16b | `…/ex2_counterfactual/unstable_other_conventions.json` | results | the above | the convention comparison behind the decision | RQ2 | done |
| 16b | `docs/ex2_counterfactual/unstable_other_convention.md` | docs | the above | **the decision**: strict, with the two rejected alternatives and why | RQ2 | done |

### Experiment 3 — persona-conditioned student simulation

| # | File | Type | Reads | Produces / Analyses | RQ | Status |
|---|---|---|---|---|---|---|
| 19 | `code/ex3_student_simulation/personas.py` | code | — | library: `TIERS`, `TIER_LABELS`, per-dataset profiles | RQ1 | undocumented in README |
| 20 | `code/ex3_student_simulation/run_student_simulation.py` | code | both test splits + Qwen3-4B | JOINT / ISOLATED simulation results | RQ1 | undocumented in README |
| 20 | `code/ex3_student_simulation/summarize_simulation.py` | code | the results below | Wilson CIs, McNemar tests, report tables | RQ1 | undocumented in README |
| 21 | `…/results_persona_simulation_{sciq,obqa}.json` + `student_simulation.log` | results | run_student_simulation | the full runs, 884 + 500 | RQ1 | undocumented in README |
| 21 | `…/results_persona_simulation_{sciq,obqa}_order_desc.json` + `order_desc.log` | results | `--joint-order descending` | order control for the joint scaffold | RQ1 | undocumented in README |
| 21 | `…/results_persona_simulation_{sciq,obqa}_smoke50.json` + `smoke50.log` | results | `--limit 50 --tag smoke50` | smoke run | RQ1 | partial — smoke only |
| 21 | `docs/ex3_student_simulation/student_persona_simulation_report.md` | docs | the above | **negative result**: persona conditioning does not stratify ability | RQ1 | undocumented in README |

### Experiment 4 — option-free free response

| # | File | Type | Reads | Produces / Analyses | RQ | Status |
|---|---|---|---|---|---|---|
| 23 | `code/ex4_free_response/matching.py` | code | `ex2.counterfactual_passage.morphological_variants` | 3-stage matching cascade | RQ1 | undocumented in README |
| 24 | `code/ex4_free_response/run_free_response.py` | code | both test splits + Qwen3-4B | cells A′ / B′ | RQ1 | undocumented in README |
| 26 | `code/ex4_free_response/build_validation_sample.py` | code | a tagged run | stratified human-annotation sheet | RQ1 | undocumented in README |
| 26 | `code/ex4_free_response/compute_kappa.py` | code | a filled sheet | **pre-registered κ ≥ 0.70 gate** | RQ1 | undocumented in README |
| 25 | `…/results_free_response_{sciq,obqa}_pilot.json` + `free_response_pilot.log` | results | `--limit 25 --tag pilot` | pilot 1, forward-only | RQ1 | partial — pilot |
| 25 | `…/results_free_response_{sciq,obqa}_pilot_bidir.json` + `free_response_pilot_bidir.log` | results | `--tag pilot_bidir` | pilot 2, bidirectional; exposed the 24.3% judge flip rate | RQ1 | partial — pilot |
| 26 | `…/validation_sheet_pilot_bidir.csv` | results | build_validation_sample | **awaiting human labels** | RQ1 | planned |
| 22, 25 | `docs/ex4_free_response/plan_option_free_response_experiment.md` | docs | the pilots | plan (§1–7), pilot results + revision (§8–9) | RQ1 | undocumented in README |

### Experiment 5 — failure audit

| # | File | Type | Reads | Produces / Analyses | RQ | Status |
|---|---|---|---|---|---|---|
| 29 | `code/ex5_failure_audit/rebuild_flagged_pool.py` | code | datasets + ex1 + ex2 results + the legacy artifact | restored generator; `--verify` proves EXACT MATCH | RQ1 | undocumented in README |
| 31 | `code/ex5_failure_audit/bucket_diagnostics.py` | code | 5 committed result files, no model | confidence gate + Setting-C ineligibility join | RQ1 | undocumented in README |
| 32 | `code/ex5_failure_audit/passage_overlap_audit.py` | code | both test splits (imports `rebuild_flagged_pool.content_tokens`) | benchmark-wide question/passage overlap screen | RQ1 | undocumented in README |
| 27 | `…/ex5_failure_analysis/rq1_flagged_questions.json` | results | **no generator — hand-built** | 408 records (SciQ 328 / OBQA 80); re-derivable except `failure_category` | RQ1 | partial — see step 28 |
| 29 | `…/ex5_failure_analysis/rq1_flagged_questions_rebuilt_{sciq,obqa}.json` | results | rebuild_flagged_pool `--out` | the re-derived pools | RQ1 | done |
| 30 | `…/ex5_failure_analysis/obqa_new_admission_labels.json` | results | **hand-annotated**, 20 items | C3-admitted OBQA labels; `_provenance` declares a known annotator effect | RQ1 | partial — single annotator, no κ |
| 31 | `…/ex5_failure_audit/bucket_diagnostics.json` | results | bucket_diagnostics | reproduces exactly on re-run | RQ1 | done |
| 31 | `…/ex5_failure_audit/both_wrong_followup.json` | results | **hand triage** | first-pass scoping; `_provenance` says explicitly *not* a substitute for full re-annotation | RQ1 | partial |
| 32 | `…/ex5_failure_audit/passage_overlap_audit.json` | results | passage_overlap_audit | reproduces exactly on re-run | RQ1 | done |
| 32 | `…/ex5_failure_audit/passage_overlap_handcheck.json` | results | **hand-annotated** | low-overlap tail; single pass, no second rater, no κ | RQ1 | partial |
| 27 | `docs/ex5_failure_audit/rq1_test_split_failure_analysis.md` | docs | ex1 + ex2 results | the manual failure taxonomy | RQ1 | undocumented in README |
| 28 | `docs/ex5_failure_audit/provenance_rq1_flagged_questions.md` | docs | the legacy artifact | establishes its provenance and the reimplementation | RQ1 | undocumented in README |
| 30 | `docs/ex5_failure_audit/failure_taxonomy_methodology.md` | docs | the pool | reconstructed rubric; §6.1 on the annotator effect | RQ1 | undocumented in README |
| 31 | `docs/ex5_failure_audit/unexploited_buckets_analysis.md` | docs | bucket_diagnostics | `both_wrong` / `correct_to_wrong` | RQ1 | undocumented in README |
| 32 | `docs/ex5_failure_audit/corpus_defect_audit.md` | docs | passage_overlap_audit | benchmark-wide corpus defect rate | RQ1 | undocumented in README |

### Synthesis and coordination

| # | File | Type | Reads | Produces / Analyses | RQ | Status |
|---|---|---|---|---|---|---|
| 33 | `docs/synthesis/rq1_research_synthesis.md` | docs | 5 reports + their JSON | **capstone**; every figure re-derived from source | RQ1 | undocumented in README |
| 34 | `RUN_QWEN2.5.md` | docs | — | live status board **and** cloud execution guide for all cloud-blocked work (merged 2026-09-11, §5.8) | RQ1 | undocumented in README |
| — | `docs/NEXT_PHASE_HANDOFF.md` | docs | the whole tree | living session-to-session handoff; spans every experiment, so it stays at `docs/` top level (§5.7) | all | undocumented in README |
| — | `docs/guides/ANNOTATION_GUIDE.md` | docs | — | operator rubric for **both** human tasks — Task A (ex4) and Task B (ex5) | RQ1 | undocumented in README |
| 1, 5, 35 | `README.md` | docs | — | research story, formulations, positioning, caveats | all | done — but see §4 |
| 11 | `REPRODUCE.md` | docs | — | full reproduction guide (ex1 + pre_data only) | RQ1 | done |
| — | `question-score/` | code | — | vendored reference KDA implementation, untouched | RQ1 | done |

---

## 4. Drift from README

Every item below was observed directly, by diffing the live tree against README's claims.

### 4.1 Exists on disk, absent from README

| What | Detail |
|---|---|
| `code/ex3_student_simulation/` | 3 scripts, ~1,540 LOC. Not in README's §4 tree. |
| `code/ex4_free_response/` | 4 scripts, ~864 LOC. Not in README's §4 tree. |
| `code/ex5_failure_audit/` | 3 scripts, ~1,017 LOC. Not in README's §4 tree. |
| `results/ex3_student_simulation/` | 9 files. Not in README's §4 tree. |
| `results/ex4_free_response/` | 7 files. Not in README's §4 tree. |
| `results/ex5_failure_audit/` | 4 files. Not in README's §4 tree. |
| `results/ex5_failure_analysis/` | 4 files. Not in README's §4 tree. |
| 12 of 20 `docs/` files | `ANNOTATION_GUIDE` · `NEXT_PHASE_HANDOFF` · `corpus_defect_audit` · `counterfactual_experiment_methodology` · `counterfactual_obqa_analysis` · `failure_taxonomy_methodology` · `plan_option_free_response_experiment` · `provenance_rq1_flagged_questions` · `rq1_research_synthesis` · `rq1_test_split_failure_analysis` · `student_persona_simulation_report` · `unexploited_buckets_analysis` — none appear in README's TOC or References. (README does now cite the other nine, including `e1_counterfactual_llm_scale`, `e2_prior_vs_rejection` and `unstable_other_convention`.) |
| `RUN_QWEN2.5.md` | Live status board **and** cloud execution guide at the repo root (1,147 lines after the 2026-09-11 merge, §5.8). README links to it from §6.5; `docs/ex4_free_response/plan_option_free_response_experiment.md` also links to it. |
| 7 of 10 dataset files | Only `sciq_50`, `sciq_test_full`, `obqa_test_full` are named. The train / val / combined exports and `obqa_50.json` are not. |
| `counterfactual_passage.py` | Appears in README's §4 tree but is never linked from README prose. |

**README's §4 tree was updated during this cleanup** to carry the new `results/` folder names, so
its paths resolve. It was **not** extended to cover the packages above — that gap is left visible
here rather than silently closed.

### 4.2 README claims that did not resolve (all now fixed)

| Claim | What was wrong | Action |
|---|---|---|
| `results/results_counterfactual_experiment.json` (README + 3 docs) | pre-reorganization path | repointed |
| `results/results_kda_small_test_full.json`, `results/openbookqa/…`, `results/categorized_results/categorized_summary.json` (all in `docs/ex5_failure_audit/rq1_test_split_failure_analysis.md`) | that document was written before the earlier `results/` restructure and never updated — **broken before this cleanup began** | repointed |
| README §4 tree, `results/` block | listed 4 stage folders; 8 exist | names corrected (see §4.1 for the remaining gap) |

### 4.3 Checked and found accurate

- **All 14 CLI flags in README §5.1** verified against live `argparse`. No flag has been renamed,
  removed, or changed scope.
- **No script referenced anywhere in README is missing.**
- Paths such as `results/…/smoke.json`, `cf_smoke.json`, `experiment_train_full.log`,
  `results_3models.json`, `dataset_prep_seed7.log`, `sciq_train_clean.json` do not exist on disk
  **by design** — they are illustrative CLI examples in README/REPRODUCE whose outputs are not
  committed. Not drift.
- All `*_qwen25*`, `kda_qwen2.5_7b_*`, `kappa_report.json`, and untagged
  `results_free_response_{sciq,obqa}.json` paths are **planned outputs** of queued work, correctly
  described as such in `RUN_QWEN2.5.md` §0 and `docs/ex4_free_response/plan_option_free_response_experiment.md`.

### 4.4 Version-control state

The repository has a **single commit** (`edd0354 Initial commit`). All of `docs/`,
`paper_references/`, `README.md`, `RUN_QWEN2.5.md`, and the ex3 / ex4 / ex5 tranche are
**untracked**; the earlier `code/` and `results/` restructure is staged but uncommitted. Git
history therefore carries no ordering information — the build order in §2 and §5 was established
from file mtimes, using **first-output** timestamps (result files are written once; code files
were edited after running, so their mtimes are last-edit, not creation).

---

## 5. Renamed during cleanup

Performed 2026-09-08. Motivation: `code/` used an `exN_` build-order prefix on four of five
experiment packages, and `results/` carried no ordinals at all; separately, SciQ outputs were
loose and untagged while OpenBookQA outputs were subfoldered and fully tagged.

### 5.1 Packages and stage folders

| Old | New |
|---|---|
| `code/rq1_audit/` | `code/ex5_failure_audit/` |
| `results/reproduce_KDA_pipeline/` | `results/ex1_reproduce_KDA_pipeline/` |
| `results/reproduce_KDA_w_modernLLM/` | `results/ex1_reproduce_KDA_w_modernLLM/` |
| `results/category_questions/` | `results/ex1_category_questions/` |
| `results/counterfact_results/` | `results/ex2_counterfactual/` |
| `results/student_simulation/` | `results/ex3_student_simulation/` |
| `results/free_response/` | `results/ex4_free_response/` |
| `results/rq1_audit/` | `results/ex5_failure_audit/` |
| `results/rq1_failure_analysis/` | `results/ex5_failure_analysis/` |

`ex5` was assigned from first-output evidence: `results/rq1_audit/bucket_diagnostics.json` was
written 2026-09-04 10:29, three minutes after Experiment 4's first output (10:26). The Sep-1
`rq1_flagged_questions.json` predates Experiment 3, but it has no generator — it is an *input* that
ex5 reconstructs, not an earlier experiment, so it claims no earlier ordinal.

### 5.2 Experiment 1 pipeline — SciQ brought up to the OpenBookQA convention

| Old | New |
|---|---|
| `results.json` / `experiment.log` | `sciq/results_kda_tiny2_sciq_test_sample50.json` / `.log` |
| `results_test_full.json` / `experiment_test_full.log` | `sciq/results_kda_tiny2_sciq_test_full.json` / `.log` |
| `results_kda_small_test_full.json` / `experiment_kda_small_test_full.log` | `sciq/results_kda_small_sciq_test_full.json` / `.log` |
| `dataset_prep.log` | `prep_sciq.log` |
| `openbookqa_prep.log` | `prep_openbookqa.log` |

> **`kda_tiny2` covers two different ensembles.** The `sample50` run used
> `kda-distilbert-base-uncased-race` + `kda-bert-base-uncased-race`; the `test_full` run used
> `kda-distilbert-base-uncased-race` + `kda-scibert-uncased-race`. Both are 2-model / ~176M, and
> the filenames are unique via `sample50` vs `test_full`. The membership difference is recorded in
> §3 rather than in the filename, and in each file's own `summary.models`.

### 5.3 Experiment 2 — SciQ given the explicit tag OpenBookQA already had

| Old | New |
|---|---|
| `results_counterfactual_experiment.json` | `results_counterfactual_sciq_test_full.json` |
| `counterfactual_experiment.log` | `counterfactual_sciq_test_full.log` |
| `counterfactual_passages_preview.json` | `counterfactual_passages_preview_sciq.json` |
| `counterfactual_obqa_experiment.log` | `counterfactual_obqa_test_full.log` |
| `counterfactual_obqa_experiment_exact_tier.log` | `counterfactual_obqa_test_exact_tier.log` |

### 5.4 Experiment 3

| Old | New |
|---|---|
| `order_control.log` | `order_desc.log` (matches the `--tag order_desc` that produced the results) |

### 5.5 Deliberately not renamed

- **`results/ex4_free_response/`** — already fully consistent (every file dataset- and run-tagged).
- **`rq1_flagged_questions*.json`** — cited as a proper noun in five documents, one of which
  (`docs/ex5_failure_audit/provenance_rq1_flagged_questions.md`) is named after it. Renaming would break prose that
  refers to it by name, for cosmetic gain.
- **Log, console, and result-file *contents*.** `console_kda_small_obqa_test_full.txt` and the
  `.log` files embed the absolute paths that were current when they ran, and result JSONs record a
  `source_results_file` basename that is now stale. These are immutable execution records; editing
  them would falsify the run history and could break the byte-level `--verify` comparison in
  `rebuild_flagged_pool.py`.
- **Doc paths recorded inside `results/` artifacts.** 26 references across 11 result files —
  including `corpus_defect_sheet_sciq33_BLIND.LOCK.json`, whose `_provenance.protocol` field names
  the annotation rubric the labelling followed — cite `docs/` paths from before the 2026-09-10
  documentation reorganization (§5.7). They are left as written for the same reason as the log
  files above; **resolve any such path through the §5.7 map.**

### 5.6 Post-move verification

All checks run after the moves, against the moved files:

| Check | Result |
|---|---|
| `py_compile` on all 28 Python files | pass |
| All cross-package imports, incl. `ex5_failure_audit.passage_overlap_audit → rebuild_flagged_pool` | pass |
| `rebuild_flagged_pool.py --dataset sciq --verify` | **EXACT MATCH**, 328/328 on every field |
| `rebuild_flagged_pool.py --dataset obqa --no-c3 --verify` | **EXACT MATCH**, 80/80 on every field |
| `bucket_diagnostics.py` re-run vs committed artifact | byte-identical |
| `passage_overlap_audit.py` re-run vs committed artifact | byte-identical |
| `run_counterfactual_experiment.py --dry-run` vs committed SciQ preview | generation stats identical (884 samples, 860 eligible) |
| `summarize_simulation.py` | reads ex3 results, produces report tables |
| `categorize_kda_results.py` auto-detect candidates | all 3 resolve |

---

### 5.7 `docs/` reorganized — 2026-09-10

Performed after the code/ and results/ restructures in §5.1, and using the *same* `exN_` names as
`code/`. Motivation: `docs/` was 21 loose files at one level, with no signal about which experiment
a document belonged to.

| Old | New |
|---|---|
| `docs/KDA_Paper_Documentation.md` | `docs/papers/KDA_Paper_Documentation.md` |
| `docs/ClashEval_Paper_Documentation.md` | `docs/papers/ClashEval_Paper_Documentation.md` |
| `docs/QG-SMS_Paper_Documentation.md` | `docs/papers/QG-SMS_Paper_Documentation.md` |
| `docs/kda_reproduction_summary.md` | `docs/ex1_reproduce_KDA/kda_reproduction_summary.md` |
| `docs/kda_qwen3_4b_evaluation_report.md` | `docs/ex1_reproduce_KDA/kda_qwen3_4b_evaluation_report.md` |
| `docs/counterfactual_experiment_methodology.md` | `docs/ex2_counterfactual/counterfactual_experiment_methodology.md` |
| `docs/counterfactual_obqa_analysis.md` | `docs/ex2_counterfactual/counterfactual_obqa_analysis.md` |
| `docs/e1_counterfactual_llm_scale.md` | `docs/ex2_counterfactual/e1_counterfactual_llm_scale.md` |
| `docs/e2_prior_vs_rejection.md` | `docs/ex2_counterfactual/e2_prior_vs_rejection.md` |
| `docs/unstable_other_convention.md` | `docs/ex2_counterfactual/unstable_other_convention.md` |
| `docs/student_persona_simulation_report.md` | `docs/ex3_student_simulation/student_persona_simulation_report.md` |
| `docs/plan_option_free_response_experiment.md` | `docs/ex4_free_response/plan_option_free_response_experiment.md` |
| `docs/rq1_test_split_failure_analysis.md` | `docs/ex5_failure_audit/rq1_test_split_failure_analysis.md` |
| `docs/failure_taxonomy_methodology.md` | `docs/ex5_failure_audit/failure_taxonomy_methodology.md` |
| `docs/provenance_rq1_flagged_questions.md` | `docs/ex5_failure_audit/provenance_rq1_flagged_questions.md` |
| `docs/unexploited_buckets_analysis.md` | `docs/ex5_failure_audit/unexploited_buckets_analysis.md` |
| `docs/corpus_defect_audit.md` | `docs/ex5_failure_audit/corpus_defect_audit.md` |
| `docs/rq1_research_synthesis.md` | `docs/synthesis/rq1_research_synthesis.md` |
| `docs/ANNOTATION_GUIDE.md` | `docs/guides/ANNOTATION_GUIDE.md` |
| `docs/RUN_QWEN2.5_7B_CLOUD_GUIDE.md` | `docs/guides/RUN_QWEN2.5_7B_CLOUD_GUIDE.md` — *superseded 2026-09-11: merged into `RUN_QWEN2.5.md`, see §5.8* |
| `docs/NEXT_PHASE_HANDOFF.md` | **unchanged** — see below |

**The three placement judgements, and why.**

- **`NEXT_PHASE_HANDOFF.md` stays at `docs/` top level.** It spans every experiment, so no `exN_`
  folder is honest; and it is a living working document rather than a permanent entry point, so the
  repository root — which holds the stable set (`README`, `REPRODUCE`, this guide, `RUN_QWEN2.5`) —
  is the wrong tier. As the only file at `docs/` top it reads as "start here".
- **`guides/` holds the operator runbooks.** `ANNOTATION_GUIDE.md` covers Task A (ex4) *and* Task
  B (ex5), so neither experiment folder fits. It was placed here alongside the Qwen2.5 cloud guide;
  **since that guide was merged into `RUN_QWEN2.5.md` on 2026-09-11 (§5.8), `guides/` now holds
  `ANNOTATION_GUIDE.md` alone.** The folder is kept rather than collapsed because it is the natural
  home for the next operator runbook, and because collapsing it would move `ANNOTATION_GUIDE.md` a
  second time in two days.
- **`synthesis/` starts with one file.** `rq1_research_synthesis.md` draws on ex1, ex2, ex3 and ex5
  and belongs to none. The folder anticipates the RQ2 counterpart that E1/E2 now make plausible.

> **Superseded by §5.8.** This section previously warned that `RUN_QWEN2.5.md` and
> `docs/guides/RUN_QWEN2.5_7B_CLOUD_GUIDE.md` were easily-confused but distinct — a status board
> and a single-task tutorial. They were merged on 2026-09-11; see §5.8 for what was combined, what
> was cut, and the two defects the merge resolved.

### 5.8 The two Qwen2.5 cloud files merged — 2026-09-11

`RUN_QWEN2.5.md` (root, 373 lines) and `docs/guides/RUN_QWEN2.5_7B_CLOUD_GUIDE.md` (725 lines)
became one file: **`RUN_QWEN2.5.md` at the root, 1,147 lines.** The guide is deleted.

**They did have distinct roles** — a living multi-entry status board versus a self-contained
single-task tutorial, and the board said so itself ("this entry is the index card for it"). The
merge was not done because they were duplicates. It was done because underneath those roles both
files independently carried a **setup layer that had already forked**:

| Duplicated content | Copies before | After |
|---|---|---|
| Install cell | **3**, with three different package lists | 1 reconciled cell (§1.2) — union of the three, keeps the `transformers>=4.44` pin and the Kaggle `numpy<3` fix |
| VRAM / precision table | 2 | 1 (§1.1) |
| Entry 1 "see the other file" pointer | 1 | deleted — the two sections are now adjacent |
| Guide's own title, TOC, "Related documents" | 1 each | rebuilt as the merged header and Contents |

**Two defects the merge resolved, both live before it:**

1. **A precision contradiction.** The board mandated 4-bit NF4 *"so quantisation is not a confound
   when diffing against the local runs"*; the guide's paste-ready commands used `--precision 8bit`
   and `fp16`. Anyone following the guide step-by-step silently forfeited comparability with the
   local Qwen3-4B run — the entire purpose of Entry 1. §1.4 now states NF4 as the policy and labels
   `8bit`/`fp16` as forfeiting that comparison.
2. **A stale planned-output path.** Entry 1 named `docs/kda_qwen2.5_7b_evaluation_report.md`, which
   §5.7 had moved; corrected to `docs/ex1_reproduce_KDA/…`. The §5.7 sweep missed it because the
   file does not exist yet.

**Deliberately kept as separate sections, because they look redundant and are not:**
uploading the *repo* (entries 2 and 4) versus uploading *dataset JSONs* with `--local-data`
(entry 1) — different payloads, both in §1.5/§2.3; and the file-naming *policy* for returned
artifacts (§7) versus Entry 1's specific output *inventory* (§2.4).

**Structure, chosen so the copy-paste-into-a-notebook use case survives a 1,147-line file:**
shared setup first and once (§1), each entry contiguous (§2–§5) so you never jump mid-task, shared
troubleshooting at the end (§6), and Entry 1's ~390-line script pushed to **Appendix A** so it never
sits between the reader and an entry. The appendix records that
`code/ex1_reproduce_KDA/run_qwen7b_eval.py` is canonical and that the inline copy is a lean
subset — it omits a `chunked()` helper and three logging constants, though all 17 shared functions
have identical signatures.

## 6. Open Threads for Next Phase

Raw material, drawn only from what is in the repository. Not recommendations.

### 6.1 Blocked, with the blocker named

1. **Experiment 4's full run is blocked on two conditions**, per
   `docs/ex4_free_response/plan_option_free_response_experiment.md` §9.3 and `RUN_QWEN2.5.md` §0:
   (a) the human validation gate — 150 items hand-labelled to κ ≥ 0.70; the sheet
   (`validation_sheet_pilot_bidir.csv`) is built and waiting, the labelling is the user's;
   (b) the Qwen2.5-7B cross-judge comparison, entry 3 on the status board — the **only** item
   marked as on the critical path.
2. **Three further Qwen2.5-7B counterparts are queued and unrun** (`RUN_QWEN2.5.md` entries 1, 2,
   4): the KDA saturation evaluation, the persona simulation, and free-response generation. All
   three are marked "not blocking."

### 6.2 Produced but not integrated into the main narrative

3. **`docs/synthesis/rq1_research_synthesis.md` synthesises five reports.** Three later documents —
   `unexploited_buckets_analysis.md`, `corpus_defect_audit.md`, `provenance_rq1_flagged_questions.md`
   — postdate it and are not in its evidence base.
4. **Six of seven per-model `basic_category/*.json` files are cited by no document.** Only
   `categorized_summary.json` is referenced.
5. **The two early Experiment 1 runs are analysed only in REPRODUCE.md**, not in any `docs/` file.
6. **`model_download.log`** is cited by no document.
7. **The three `_all_combined` / train / val dataset exports are consumed by no experiment.**

### 6.3 Carrying a stated methodological caveat

8. **Three hand-annotated artifacts declare their own limits** in `_provenance`:
   `passage_overlap_handcheck.json` and `both_wrong_followup.json` are single-pass with no second
   rater and no κ; `obqa_new_admission_labels.json` states a known annotator effect from mixing 20
   newly-labelled OBQA items with the original 80. `both_wrong_followup.json` says explicitly it is
   "NOT a substitute for" the full re-annotation.
9. **`rq1_flagged_questions.json`'s `failure_category` field remains the one component that cannot
   be re-derived** — everything else now verifies exactly.
10. **Setting C does not extend to OpenBookQA at scale.** 145 of 500 items eligible at `partial`,
    42 at `exact`. README §6.6 records this; `counterfactual_obqa_analysis.md` measures it.
11. ~~**`unstable_other` is credited to neither class**~~ — **resolved 2026-09-09.** The four
    estimators had been *inconsistent* about it (hard/soft/verified-accuracy scored it as a
    failure; sample exclusion retained it at full `KDA_cont`). The **strict** convention now
    applies to all four; see `docs/ex2_counterfactual/unstable_other_convention.md`. Exposure is 19.7% of SciQ and
    33.8–34.5% of OBQA (model, question) pairs and is reported with every run.

### 6.4 Placeholders and unbuilt work

12. **`results/ex1_category_questions/complicated_category/`** holds only a placeholder README,
    which names its own intended contents (the Setting-C classes currently living in
    `ex2_counterfactual/`) and says to "replace or delete it once real artefacts land here."
13. **RQ3 / the downstream filtering study has no code.** README §6.5 specifies three arms
    (No Filtering / KDA Filtering / Our Disentangled Filtering) and QG-SMS-based validation. No
    package exists; on the build-order convention it would be **`ex7_`**. *(Updated 2026-09-14:
    this reservation was `ex6_` until the P/S/F/D prototype claimed that ordinal — `ex6_psfd_score`
    was built first, so on a build-order convention it takes `ex6`. See
    `docs/ex6_psfd_score/psfd_formulation.md`.)*
14. ~~**Experiment 2's Setting C has never been run with an LLM solver.**~~ **Closed 2026-09-09**
    by E1 (`docs/ex2_counterfactual/e1_counterfactual_llm_scale.md`): it does still separate them — 62.5%
    `prior_dependent` on an 819-item target set, prior inflation +59.9 pp. Two new open items
    replace it: the Setting-C label agrees across solvers at only κ = 0.046 (**open — now the
    largest RQ2/RQ3 risk**), and at LLM scale `prior_dependent` might not be separable from
    "correctly rejected a false context" (**tested and largely ruled out by E2**,
    `docs/ex2_counterfactual/e2_prior_vs_rejection.md`: 46.0%–62.5% bracket, plausibility AUC 0.496).

### 6.4b Closed since this guide was written

18. **The two readings of $KDA_{adj}^{excl}$** (2026-09-09). README §2.4 defines a mean over the
    retained set; the code reported a mean over the eligible set with dropped items zero-filled.
    Both are now reported. Under README's reading the estimator is **inert** — SciQ 0.4793 against
    a 0.4767 baseline — because `KDA_cont` barely discriminates items that pass the context check
    from those that fail it (AUC 0.553). This also closed the $\mathcal{Q}_{ctx}$ sensitivity
    question: nine definitions, retained mean within a few percent of baseline on every one.
19. **The `unstable_other` convention** (2026-09-09) — item 11 above.

### 6.5 Repository hygiene

15. **Nothing since the initial commit is committed.** All of `docs/`, `paper_references/`,
    `README.md`, `RUN_QWEN2.5.md`, and the ex3/ex4/ex5 tranche are untracked; the `code/` and
    `results/` restructure is staged. A single commit currently represents the whole project.
16. **README's §4 tree still omits ex3, ex4, and ex5** (§4.1). Its paths now resolve, but the
    package list is four packages and four results folders short.
17. **`REPRODUCE.md` covers only Experiment 1 and `pre_data`.** Experiments 2–5 have no
    reproduction guide outside their individual methodology documents.
