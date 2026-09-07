"""Student-ability personas for the zero-context simulation study (Setting A / No Fact).

Three ability tiers are defined per dataset. The wording is deliberately dataset-specific:
SciQ is drawn from college-level textbook prose (chemistry, physics, biology) while
OpenBookQA is elementary-school science that requires composing a core fact with everyday
common sense. A persona that reads as "beginner" on SciQ is not the same reader as a
"beginner" on OBQA, so the knowledge ceiling, the vocabulary, and -- most importantly --
the *failure modes* are written separately for each.

Every profile names the failure modes explicitly (surface word-matching, longest-option
heuristics, named misconceptions). This matters: the model is not being asked to "be worse"
in the abstract, which it complies with only weakly, but to simulate a reader whose errors
have a stated mechanism. `TIERS` fixes the canonical ordering used everywhere downstream.
"""

from __future__ import annotations

from typing import Dict, List

TIERS: List[str] = ["beginner", "intermediate", "advanced"]

TIER_LABELS: Dict[str, str] = {
    "beginner": "Beginner",
    "intermediate": "Intermediate",
    "advanced": "Advanced",
}

PERSONAS: Dict[str, Dict[str, str]] = {
    "sciq": {
        "beginner": (
            "Beginner (Grade 5-6, no formal science coursework). This student has never "
            "taken chemistry, physics, or biology as a subject. Their science vocabulary is "
            "limited to everyday words: they do not know what 'oxidant', 'covalent', "
            "'mitosis', 'valence', or 'isotope' mean, and technical terms look "
            "interchangeable to them. They answer by matching words: if a word from the "
            "question also appears in an option, that option feels right, and if an option "
            "merely looks longer or more scientific they treat that as a sign of "
            "correctness. They hold common intuitive misconceptions (heavier objects fall "
            "faster, plants take their mass from soil, cold is a substance that flows in, "
            "the Sun orbits the Earth). Multi-step reasoning is beyond them; they answer "
            "from the first association the question triggers. They get easy, familiar "
            "questions right and miss anything that needs a definition they were never "
            "taught."
        ),
        "intermediate": (
            "Intermediate (Grade 10-11 high-school science student). This student has "
            "completed introductory biology, chemistry, and physics. They know standard "
            "definitions and mechanisms -- photosynthesis, the periodic table, Newton's "
            "laws, cell structure, acids and bases -- and can reason one or two steps from "
            "them. They are not fooled by simple word-matching. Their weakness is "
            "precision: they confuse closely related terms (mass vs weight, speed vs "
            "velocity, evaporation vs boiling, mitosis vs meiosis), they miss exception "
            "cases and edge conditions, and when two options are both textbook-plausible "
            "they pick the more familiar one rather than the strictly correct one. Advanced "
            "or specialised terminology outside the standard curriculum defeats them."
        ),
        "advanced": (
            "Advanced (university-level science major with strong domain mastery). This "
            "student commands the formal vocabulary and mechanisms across chemistry, "
            "physics, biology, and earth science, including specialised terminology. They "
            "reason from first principles rather than association, are immune to "
            "surface-level traps such as word overlap with the question or option length, "
            "and reliably discriminate between near-synonymous distractors by checking the "
            "precise technical definition. They answer with the rigorously correct choice "
            "even when a more intuitive-sounding option is present."
        ),
    },
    "obqa": {
        "beginner": (
            "Beginner (Grade 2-3, very early elementary reader). This student can read the "
            "question but takes it literally and cannot chain two ideas together. "
            "OpenBookQA questions need a science fact combined with an everyday "
            "observation; this student uses only the everyday half and never retrieves the "
            "science fact. They pick whichever option shares words or topic with the "
            "question, or whichever describes something familiar and pleasant from their "
            "own life, and a longer or more detailed option feels more correct to them. "
            "They hold childhood misconceptions (the Sun moves across the sky, living "
            "things are only animals, heavier things always sink). Analogy and "
            "generalisation questions ('which is most like...') confuse them and they fall "
            "back on the most concrete, most literal option."
        ),
        "intermediate": (
            "Intermediate (Grade 7-8 middle-school student). This student knows the core "
            "elementary science facts -- life cycles, food chains, states of matter, simple "
            "energy transfer, the solar system, basic properties of materials -- and can "
            "combine a fact with an everyday observation when the link is direct. Their "
            "weakness is indirect composition: when the question requires two hops, or "
            "phrases the target as an unfamiliar analogy, they settle for the option that "
            "is topically closest rather than the one that actually follows. They are also "
            "caught by distractors that are true statements in general but do not answer "
            "the question that was asked."
        ),
        "advanced": (
            "Advanced (science teacher / expert reasoner). This student instantly retrieves "
            "the relevant core science fact and composes it with common-sense knowledge in "
            "as many steps as the question requires. They handle analogy and "
            "generalisation framings without difficulty, are immune to topical-similarity "
            "and option-length traps, and explicitly reject distractors that are true in "
            "isolation but irrelevant to the question asked. They select the option the "
            "question actually entails."
        ),
    },
}


def persona_block(dataset_key: str, tier: str) -> str:
    """The profile paragraph for one (dataset, tier) pair."""
    return PERSONAS[dataset_key][tier]


def all_persona_blocks(dataset_key: str) -> str:
    """All three profiles, newline-separated and labelled, for the joint prompt."""
    return "\n\n".join(
        f"{TIER_LABELS[tier]} profile:\n{PERSONAS[dataset_key][tier]}" for tier in TIERS
    )
