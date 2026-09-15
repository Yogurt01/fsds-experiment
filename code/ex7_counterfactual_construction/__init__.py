"""Experiment 7 — auditing and improving the counterfactual construction itself.

Experiment 2's Setting C rewrites the gold answer string inside the reference passage as a
distractor. Everything downstream — the context_dependent / prior_dependent labels, and
every ex6 P/S/F/D result computed from them — inherits whatever that rewrite does or fails
to do. This package audits the rewrite as a measurement instrument and designs a
replacement for the datasets where it does not hold up.

Out of scope, deliberately: the ex6 section-14 Tier-1 circularity finding. A perfectly
constructed counterfactual would still yield labels near-tautologically determined by
MarginC. Construction quality and label informativeness are separate problems and nothing
here resolves the second.

Nothing in this package modifies `code/ex2_counterfactual/` or any committed ex2/ex6
artifact. Outputs go to `results/ex7_counterfactual_construction/`.
"""
