# Complicated category — placeholder

Reserved for question categorisations that go beyond the four-bucket contingency table in
[`../basic_category/`](../basic_category).

`basic_category/` holds the output of
[`code/ex1_reproduce_KDA/categorize_kda_results.py`](../../../code/ex1_reproduce_KDA/categorize_kda_results.py),
which splits every question by the pair `(r^q, r^{q+f})`:

| Bucket | Without fact | With fact |
|---|---|---|
| `both_correct` | correct | correct |
| `wrong_to_correct` | wrong | correct |
| `both_wrong` | wrong | wrong |
| `correct_to_wrong` | correct | wrong |

That split cannot say *why* an answer was correct. Richer categorisations belong here — for
example the counterfactual classes (`context_dependent` / `prior_dependent` /
`unstable_other`) produced by
[`code/ex2_counterfactual/run_counterfactual_experiment.py`](../../../code/ex2_counterfactual/run_counterfactual_experiment.py),
whose raw output currently lives in [`../../ex2_counterfactual/`](../../ex2_counterfactual).

This file is a placeholder so the directory is preserved in git; replace or delete it once
real artefacts land here.
