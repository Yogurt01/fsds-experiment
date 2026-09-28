"""Experiment 8 — an independent, human-annotated label set for RQ2.

Every comparison in ex6 was adjudicated against Setting-C-derived labels, and ex6 section 14
showed those labels are near-tautologically determined by MarginC: any quantity built from
Setting C scores well on them in proportion to how directly it re-encodes them. That makes
Tier 1 unable to rank F against D against KDA_cont, which is the question RQ2 has to answer.

This package builds the missing instrument: labels produced by a human reading only the
question, the options and the passage, with no access to any solver output, Setting-A/B/C
result, KDA_cont value, P/S/F/D score, or failure-taxonomy category. Nothing here computes
or modifies a metric; it builds sheets and scores annotator agreement.
"""
