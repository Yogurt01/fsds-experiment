# RQ1 Final Report — Appendix A: Corrections to the Earlier RQ1 Reports

Companion to [`rq1_final_report.md`](rq1_final_report.md). This appendix lists every point where
[`rq1_final_report.md`](rq1_final_report.md) departs from the two earlier RQ1 reports, and why:
- "Concl." = [`rq1_conclusion.md`](rq1_conclusion.md);
- "Synth." = [`rq1_research_synthesis.md`](../synthesis/rq1_research_synthesis.md).

Both earlier reports are left unchanged. Each row was checked against the primary source cited. The
"Treatment" column refers to section numbers in `rq1_final_report.md`.

| # | Earlier report states | Primary source says | Treatment |
|---|---|---|---|
| 1 | Concl. §10: `correct_to_wrong` "4 items under Qwen3-4B, 9 under Qwen2.5-7B", in a SciQ context | 4 and 9 are **OBQA**; SciQ is 0 and 1 ([Q3] §4.3; [Q25] §4.3) | Reported per dataset (§3) |
| 2 | Concl. §5, Synth. §2.2: weights near-constant for "three of four" students on OBQA | [FA] Table 11 text says three; the results JSON gives sd 0.327 / **0.041** / 0.279 / **0.099**, i.e. two | Both stated; recomputed values used (§2.3) |
| 3 | Concl. §4 "four criteria" vs §11 "three criteria… OBQA two" | [FA] lists four flags grouped as "three operational criteria"; C3 was later applied to OBQA ([FTM] §6.1) | Four flags named explicitly; OBQA update in §4 |
| 4 | Concl. §4, Synth. §4.1: 115 / 860 = 13.4% `prior_dependent` | Published value; ex7 strict gives 101 / 830 = 12.17% ([RL] §5.1) | Both shown (§2.2) |
| 5 | Concl. §7: "90.4% retention" | Strict convention gives 89.6% ([UO]) | Both shown (§2.5) |
| 6 | Concl. §8: generated-question rows "reach 0.51–0.64" | "All" column: KDDG 0.42\*, DG 0.61\*\*, QDG 0.64\*\*, DGen models 0.51\*\* ([KP] Table 10) | Per-row values (§2.7) |
| 7 | Concl. §8: Table 11 correctness "with 2022-era solvers" | [KP] gives average dataset correctness and does not attribute it to solvers | Qualifier dropped (§2.7) |
| 8 | Concl. §9: 44.0% "against 4.4% when emitted first" | Both are the **last-emitted** tier: beginner last 44.0%, advanced last 4.4% ([PSR] §7.3) | Corrected wording (§2.8) |
| 9 | Concl. §9: judge caveat cites κ 0.545 only | [FR] §10.2–10.3 also: bands self-judged; `acc_normalised` 0.535 is headline-licensed; human-vs-Qwen3-4B gate passed at κ 0.754 | Added (§2.8) |
| 10 | Concl. §9: "census" gives 0 / 500 on OBQA | 25-item sample of 123 zero-overlap items ([CD] §0, §4) | Stated as sample (§2.8) |
| 11 | Concl. §7: $KDA_{cont}$ AUC 0.5494 / 0.3550 | Leak-free: 0.5507 / 0.4163 ([CCA] §3.4); OBQA Setting C closed as a stress test ([CCA] §11) | Both shown (§2.5) |
| 12 | Concl. §6: unweighted AUC 0.580 / 0.569 only | IPW AUC 0.455 / 0.387 ([AR] §B2, §D3) | Added (§2.4) |
| 13 | Concl. §9–10: "roughly 30% construction artifacts" | 17 / 57 is a 55-item-sample figure; corpus strict leak is 30 / 860 = 3.5% ([RL] §3) | Both, with scope stated (§2.8) |
| 14 | Synth. §3: SciQ #587 `kda_adjusted_hard = 0.208` | Item value is 0.839; 0.208 is the `prior_dependent` group mean ([FA] §3, Table 6) | Case studies omitted; 0.208 used only as group mean |
| 15 | Synth. §4.2: "up to 92.1% of those breaks are away from a correct consensus" | 92.1% is OBQA "breaks that are wrong"; away-from-correct is 96.2% SciQ / 85.1% OBQA ([PSR] §3.3) | Not used |
| 16 | Synth. §5: `wrong_to_correct` 39 / 884, 51 / 500 = "4.6% / 17.4%" | 4.41% / 10.2%; 4.6% / 17.4% are the $KDA_{disc}$ denominators ([Q3] §4.3, §5) | Corrected (§2.1) |
| 17 | Synth. §4.1: OBQA "consistently 2–3×" worse | "Roughly double" for the ensemble; 5.6× for primary-model `partial` ([CO] §0, §3.2) | "Roughly double", ensemble only (§2.6) |
| 18 | Synth. §3: RR 1.13–1.21 for "templated option families and longest-option bias" | 1.13 templated, 1.17 longest, 1.21 stem-echo ([FA] Table 4) | Not used |
| 19 | Synth. Appendix: persona covers "one model" | Qwen2.5-7B replication added ([PSR] §7) | Both models (§2.8) |
| 20 | Synth. §4.1: $Acc_{wf}^{verified}$ 0.9058 → 0.8581, unlabelled | Primary model (mpnet); ex7 strict gives 0.8651 ([CO] §3.3; [RL] §5.1) | Labelled; both shown (§2.6) |
| 21 | Synth. §4.1: SciQ `both_correct` `prior_dependent` 8.2% / 22.3% | ex7 strict gives 6.7% (27/401) / 21.5% (79/368) | Both shown (§2.6) |

**Note on row 2.** The OBQA weight standard deviations were recomputed from the per-model `weight`
field of `results/ex1_reproduce_KDA_pipeline/openbookqa/results_kda_small_obqa_test_full.json`.
The same recomputation on the SciQ file reproduces [FA] Table 11 exactly (0.397 / 0.044 / 0.256 /
0.091).

**Note on rows 4, 20 and 21.** The corrected values are from
`results/ex7_counterfactual_construction/prior_dependence_corrected_sciq.json` (`strict` exclusion
set, n = 830).

**Not cited.** `docs/NEXT_PHASE_HANDOFF.md` §3.3 still carries pre-strict-convention exclusion
values (0.4781, AUC 0.536). The current values are in [CM] §6 and [CO] §2.3.

[Q3]: ../ex1_reproduce_KDA/kda_qwen3_4b_evaluation_report.md
[Q25]: ../ex1_reproduce_KDA/kda_qwen2.5_7b_evaluation_report.md
[CM]: ../ex2_counterfactual/counterfactual_experiment_methodology.md
[CO]: ../ex2_counterfactual/counterfactual_obqa_analysis.md
[UO]: ../ex2_counterfactual/unstable_other_convention.md
[PSR]: ../ex3_student_simulation/student_persona_simulation_report.md
[FR]: ../ex4_free_response/plan_option_free_response_experiment.md
[FA]: ../ex5_failure_audit/rq1_test_split_failure_analysis.md
[FTM]: ../ex5_failure_audit/failure_taxonomy_methodology.md
[CD]: ../ex5_failure_audit/corpus_defect_audit.md
[CCA]: ../ex7_counterfactual_construction/counterfactual_construction_audit.md
[RL]: ../ex7_counterfactual_construction/residual_leakage_rescan.md
[AR]: ../ex8_independent_labels/rq2_annotation_results.md
[KP]: ../papers/KDA_Paper_Documentation.md
