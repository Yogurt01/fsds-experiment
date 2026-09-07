"""KDA_tiny: a lightweight multi-model reimplementation of the KDA_cont metric.

The reference implementation (``question_score.KDA``) evaluates ``kda_small`` with
four models (~3.5GB total). KDA_tiny keeps the exact same mathematics but runs an
ensemble of small multiple-choice checkpoints (~700MB total by default), each one
acting as a different *simulated student*.

For a question ``q`` with target fact ``f`` and an ensemble of models ``M``:

    KDA_cont(q) = sum_{m in M} (1 - P_m(R^q = 1)) * P_m(R^{q+f} = 1)
                  / sum_{m in M} (1 - P_m(R^q = 1))

``P_m(R^q = 1)`` is the probability that model ``m`` picks the correct option when the
target fact is withheld, and ``P_m(R^{q+f} = 1)`` is the same probability once the
passage is prepended. The weight ``(1 - P_m(R^q = 1))`` is the model's prior
*un*-answerability, so students who could already guess the answer without the fact
contribute little to the score.

Why |M| >= 2 matters: with a single model the weight appears in both the numerator and
the denominator and cancels exactly, collapsing the metric to plain
``P(R^{q+f} = 1)``. The weighted average only carries information once several
students disagree, which is why this module requires at least two models.

Prompt construction follows ``KDA.infer_bert_model`` in the original repository:
for every option the question is combined with that option (``_`` is replaced by the
option when present, otherwise the option is appended), the passage is prepended for
the with-fact pass, and the four resulting strings are scored jointly by an
``AutoModelForMultipleChoice`` head and turned into a distribution with softmax.
"""

from __future__ import annotations

import gc
import logging
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModelForMultipleChoice, AutoModelForSeq2SeqLM, AutoTokenizer

transformers.logging.set_verbosity_error()

BERT_MAX_TOKEN_LEN = 512
TOK_MARGIN = 30
MIN_ENSEMBLE_SIZE = 2
DENOMINATOR_EPSILON = 1e-12

# Two lightweight simulated students, both RACE-fine-tuned multiple-choice checkpoints
# published by the authors of the KDA paper, used as-is with no retraining.
# Student 2 is a genuine member of the official KDA_SMALL suite; student 1 comes from the
# wider KDA_FULL list and is the smallest checkpoint the authors released.
# NOTE on names that do NOT exist on the HuggingFace Hub (the API returns 401 for both):
#   - "Riiid/kda-scibert-scivocab-uncased-race" -> the real name is
#     "Riiid/kda-scibert-uncased-race" (the one used below).
#   - "Riiid/kda-albert-base-v2-race" -> only xlarge/xxlarge ALBERT variants were released.
DEFAULT_MODELS = [
    "Riiid/kda-distilbert-base-uncased-race",  # ~268MB,  67.0M parameters (KDA_FULL)
    "Riiid/kda-scibert-uncased-race",          # ~440MB, 109.9M parameters (KDA_SMALL)
]

# Model families whose forward pass does not accept token_type_ids.
NON_TOKEN_TYPE_KEYWORDS = ("distilbert", "roberta", "mpnet", "xlnet")

# The official KDA_small suite from the paper (question_score/kda.py:14-19).
# Combined weights are ~1.42GB, which does not fit alongside activations on a 4GB card,
# so this ensemble must be evaluated with strategy="sequential".
KDA_SMALL = [
    "google/t5-small-ssm-nq",           # ~308MB, seq2seq -- scored by T5Student
    "Riiid/kda-albert-xlarge-v2-race",  # ~235MB (parameter sharing; wide activations)
    "Riiid/kda-mpnet-base-race",        # ~438MB
    "Riiid/kda-scibert-uncased-race",   # ~440MB
]

# Named presets accepted on the command line (resolved in run_experiment.py).
MODEL_PRESETS = {"KDA_SMALL": KDA_SMALL, "DEFAULT": DEFAULT_MODELS}


def get_bert_postfix(question: str, option: str) -> str:
    """Merge a question with one answer option (identical to the original repository)."""
    if question.find("_") != -1:
        return question.replace("_", " " + option + " ")
    return question + " " + option


class EncoderStudent:
    """An encoder multiple-choice model (BERT / ALBERT / MPNet / SciBERT / DistilBERT).

    All four options are scored jointly by an AutoModelForMultipleChoice head and a
    softmax over the resulting four logits yields the answer distribution.
    """

    def __init__(
        self,
        model_name: str,
        device: str,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.logger = logger or logging.getLogger(__name__)

        self.logger.info("Loading simulated student '%s' on device '%s'", model_name, device)
        load_started = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMultipleChoice.from_pretrained(
            model_name, num_labels=4
        ).to(device)
        self.model.eval()
        load_seconds = time.time() - load_started

        name_lower = model_name.lower()
        self.uses_token_type = not any(k in name_lower for k in NON_TOKEN_TYPE_KEYWORDS)
        self.n_parameters = sum(p.numel() for p in self.model.parameters())

        self.logger.info(
            "Loaded '%s' in %.2fs | %.1fM parameters (~%.0fMB fp32) | token_type_ids=%s",
            model_name,
            load_seconds,
            self.n_parameters / 1e6,
            self.n_parameters * 4 / 1e6,
            self.uses_token_type,
        )

    def unload(self) -> None:
        """Release the weights so the next student can fit in VRAM."""
        self.model.to("cpu")
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def truncate_passage(self, passage: str, question: str, options: Sequence[str]) -> str:
        """Trim the passage so the full prompt stays within the 512-token limit.

        The original implementation splits an over-long passage into several chunks and
        keeps the highest KDA; SciQ 'support' fields are short enough that this branch is
        effectively never taken, so a tail truncation is sufficient here.
        """
        if not passage:
            return passage
        max_postfix = max(
            len(self.tokenizer.encode(get_bert_postfix(question, option)))
            for option in options
        )
        budget = BERT_MAX_TOKEN_LEN - TOK_MARGIN - max_postfix
        token_ids = self.tokenizer.encode(passage, add_special_tokens=False)
        if len(token_ids) <= budget or budget <= 0:
            return passage
        self.logger.debug(
            "Truncating passage for '%s': %d -> %d tokens",
            self.model_name,
            len(token_ids),
            budget,
        )
        return self.tokenizer.decode(token_ids[:budget])

    @torch.no_grad()
    def choice_probabilities(
        self, prompt: str, question: str, options: Sequence[str]
    ) -> List[float]:
        """Return this student's probability distribution over the four options."""
        texts = [prompt + get_bert_postfix(question, option) for option in options]
        encoded = self.tokenizer(
            texts,
            add_special_tokens=True,
            truncation=True,
            max_length=BERT_MAX_TOKEN_LEN,
            padding=True,
            return_tensors="pt",
            return_token_type_ids=self.uses_token_type,
        )
        # AutoModelForMultipleChoice expects (batch_size, num_choices, sequence_length).
        batch = {key: value.unsqueeze(0).to(self.device) for key, value in encoded.items()}
        logits = self.model(**batch).logits[0]
        return F.softmax(logits.float(), dim=-1).cpu().tolist()


class T5Student:
    """A seq2seq student (google/t5-small-ssm-nq).

    T5 has no multiple-choice head, so an option is scored the way the reference
    implementation does it in `KDA.infer_t5_model`: the source is the prompt plus the
    question (the option is NOT appended), each option is used as the decoder target, and
    the option's score is the negative length-normalised NLL of generating it. A softmax
    over the four scores turns them into the same kind of answer distribution the encoder
    students produce.
    """

    def __init__(
        self,
        model_name: str,
        device: str,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        self.uses_token_type = False

        self.logger.info("Loading simulated student '%s' on device '%s'", model_name, device)
        load_started = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.model.eval()
        load_seconds = time.time() - load_started

        # Same loss configuration as the reference implementation.
        self.loss_fct = nn.NLLLoss(
            reduction="none", ignore_index=self.model.config.pad_token_id
        )
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.max_length = BERT_MAX_TOKEN_LEN

        self.n_parameters = sum(p.numel() for p in self.model.parameters())
        self.logger.info(
            "Loaded '%s' in %.2fs | %.1fM parameters (~%.0fMB fp32) | seq2seq scoring",
            model_name,
            load_seconds,
            self.n_parameters / 1e6,
            self.n_parameters * 4 / 1e6,
        )

    def unload(self) -> None:
        self.model.to("cpu")
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def truncate_passage(self, passage: str, question: str, options: Sequence[str]) -> str:
        """No explicit trimming: the tokenizer truncates the source to max_length."""
        return passage

    @torch.no_grad()
    def choice_probabilities(
        self, prompt: str, question: str, options: Sequence[str]
    ) -> List[float]:
        options = list(options)
        source = [prompt + question for _ in options]

        encoded_src = self.tokenizer(
            source, max_length=self.max_length, truncation=True, padding=True,
            return_tensors="pt",
        )
        encoded_tgt = self.tokenizer(
            options, max_length=self.max_length, truncation=True, padding=True,
            return_tensors="pt",
        )
        src_tokens = encoded_src["input_ids"].to(self.device)
        src_mask = encoded_src["attention_mask"].to(self.device)
        tgt_tokens = encoded_tgt["input_ids"].to(self.device)
        tgt_len = encoded_tgt["attention_mask"].sum(dim=1).to(self.device)

        output = self.model(
            input_ids=src_tokens, attention_mask=src_mask, labels=tgt_tokens
        )
        logits = output.logits.view(-1, self.model.config.vocab_size)
        loss = self.loss_fct(self.log_softmax(logits.float()), tgt_tokens.view(-1))
        loss = loss.view(tgt_tokens.shape[0], -1).sum(dim=1) / tgt_len
        # Higher score = more likely option; softmax matches KDA.get_correct_prob.
        return F.softmax(-loss, dim=-1).cpu().tolist()


# Backwards-compatible alias: earlier revisions exposed the encoder class by this name.
SimulatedStudent = EncoderStudent


def make_student(model_name: str, device: str, logger: logging.Logger):
    """Pick the right scoring pipeline for a checkpoint (seq2seq vs multiple choice)."""
    if "t5" in model_name.lower():
        return T5Student(model_name, device, logger)
    return EncoderStudent(model_name, device, logger)


class KDATiny:
    """Ensemble KDA_cont scorer over |M| >= 2 simulated students."""

    def __init__(
        self,
        model_names: Optional[Sequence[str]] = None,
        device: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        allow_single_model: bool = False,
        strategy: str = "auto",
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        model_names = list(model_names) if model_names else list(DEFAULT_MODELS)

        if len(model_names) < MIN_ENSEMBLE_SIZE and not allow_single_model:
            raise ValueError(
                f"KDA_cont requires at least {MIN_ENSEMBLE_SIZE} models: with |M| = 1 the "
                "weight (1 - P(R^q = 1)) cancels between numerator and denominator and the "
                "metric degenerates to P(R^{q+f} = 1). Pass allow_single_model=True only to "
                "reproduce that degenerate baseline on purpose."
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_names = model_names

        # "resident" keeps every student in VRAM at once (fast, needs the memory);
        # "sequential" evaluates the whole dataset one student at a time, unloading
        # between models. Three or more checkpoints do not fit alongside activations on
        # a 4GB card, so that is the automatic cut-off.
        if strategy == "auto":
            strategy = "sequential" if len(model_names) >= 3 else "resident"
        if strategy not in ("resident", "sequential"):
            raise ValueError(f"strategy must be 'auto', 'resident' or 'sequential', got {strategy!r}")
        self.strategy = strategy

        self.logger.info(
            "Initialising KDA_tiny ensemble with %d model(s) on device '%s' (strategy=%s)",
            len(model_names),
            device,
            strategy,
        )

        self.parameter_counts: Dict[str, int] = {}
        if strategy == "resident":
            self.students = [make_student(name, device, self.logger) for name in model_names]
            self.parameter_counts = {s.model_name: s.n_parameters for s in self.students}
            self.logger.info(
                "Ensemble ready: %d models, %.1fM parameters total (~%.0fMB fp32)",
                len(self.students),
                self.total_parameters / 1e6,
                self.total_parameters * 4 / 1e6,
            )
        else:
            self.students = []
            self.logger.info(
                "Sequential strategy: models are loaded one at a time and unloaded "
                "between passes, so peak VRAM is set by the largest single checkpoint."
            )

    @property
    def total_parameters(self) -> int:
        return sum(self.parameter_counts.values())

    def _log_vram(self, stage: str) -> None:
        if not torch.cuda.is_available() or self.device == "cpu":
            return
        free, total = torch.cuda.mem_get_info()
        self.logger.info(
            "  VRAM %-18s allocated=%.0fMB reserved=%.0fMB free=%.0fMB / %.0fMB",
            stage,
            torch.cuda.memory_allocated() / 1e6,
            torch.cuda.memory_reserved() / 1e6,
            free / 1e6,
            total / 1e6,
        )

    def _release(self, student) -> None:
        """Unload one student and reclaim its VRAM."""
        name = student.model_name
        student.unload()
        del student
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("Unloaded '%s' and cleared the CUDA cache", name)
        self._log_vram("after unload")

    @staticmethod
    def _student_probabilities(student, passage, question, options) -> Tuple[List[float], List[float]]:
        """Run one student's without-fact and with-fact passes for a single question."""
        truncated = student.truncate_passage(passage, question, options)
        prefix = (truncated.strip() + " ") if truncated.strip() else ""
        probs_without_fact = student.choice_probabilities("", question, options)
        probs_with_fact = student.choice_probabilities(prefix, question, options)
        return probs_without_fact, probs_with_fact

    def _aggregate(
        self,
        model_probabilities: Dict[str, Tuple[List[float], List[float]]],
        answer_idx: int,
        question: str = "",
    ) -> Dict:
        """Combine per-model probability vectors into the ensemble KDA_cont score.

        Shared by both strategies so the resident and sequential paths are guaranteed to
        produce identical numbers.
        """
        per_model: Dict[str, Dict] = {}
        numerator = 0.0
        denominator = 0.0

        for model_name, (probs_without_fact, probs_with_fact) in model_probabilities.items():
            p_correct_without_fact = probs_without_fact[answer_idx]  # P_m(R^q = 1)
            p_correct_with_fact = probs_with_fact[answer_idx]        # P_m(R^{q+f} = 1)
            weight = 1.0 - p_correct_without_fact                    # P_m(R^q = 0)

            numerator += weight * p_correct_with_fact
            denominator += weight

            predicted_without_fact = max(
                range(len(probs_without_fact)), key=probs_without_fact.__getitem__
            )
            predicted_with_fact = max(
                range(len(probs_with_fact)), key=probs_with_fact.__getitem__
            )

            per_model[model_name] = {
                "p_correct_without_fact": p_correct_without_fact,
                "p_correct_with_fact": p_correct_with_fact,
                "weight": weight,
                "weighted_contribution": weight * p_correct_with_fact,
                "probability_gain": p_correct_with_fact - p_correct_without_fact,
                "predicted_idx_without_fact": predicted_without_fact,
                "predicted_idx_with_fact": predicted_with_fact,
                "is_correct_without_fact": predicted_without_fact == answer_idx,
                "is_correct_with_fact": predicted_with_fact == answer_idx,
                "probabilities_without_fact": probs_without_fact,
                "probabilities_with_fact": probs_with_fact,
            }

        # Safety check: every model already answers correctly with full confidence.
        if denominator <= DENOMINATOR_EPSILON:
            self.logger.warning(
                "Denominator sum(1 - P_m(R^q = 1)) is ~0 for question %r; "
                "setting KDA_cont = 0 to avoid division by zero.",
                question[:80],
            )
            kda_score = 0.0
            degenerate = True
        else:
            kda_score = numerator / denominator
            degenerate = False

        mean_p_without_fact = sum(
            m["p_correct_without_fact"] for m in per_model.values()
        ) / len(per_model)
        mean_p_with_fact = sum(
            m["p_correct_with_fact"] for m in per_model.values()
        ) / len(per_model)

        return {
            "kda_score": kda_score,
            "numerator": numerator,
            "denominator": denominator,
            "zero_denominator": degenerate,
            "mean_p_correct_without_fact": mean_p_without_fact,
            "mean_p_correct_with_fact": mean_p_with_fact,
            "mean_probability_gain": mean_p_with_fact - mean_p_without_fact,
            "per_model": per_model,
        }

    def score_dataset(
        self,
        samples: Sequence[Dict],
        progress_callback: Optional[Callable[[str, int, int, int, int], None]] = None,
    ) -> List[Dict]:
        """Score a whole dataset with only one student resident at a time.

        Each model sweeps the entire dataset, its probability vectors are cached, the
        weights are released, and only then is the next model loaded. KDA_cont is
        aggregated across all models at the end.
        """
        cache: Dict[str, List[Tuple[List[float], List[float]]]] = {}

        for model_position, model_name in enumerate(self.model_names, start=1):
            self.logger.info("=" * 70)
            self.logger.info(
                "Sequential pass %d/%d: %s", model_position, len(self.model_names), model_name
            )
            self._log_vram("before load")
            student = make_student(model_name, self.device, self.logger)
            self.parameter_counts[model_name] = student.n_parameters
            self._log_vram("after load")

            pass_started = time.time()
            per_sample: List[Tuple[List[float], List[float]]] = []
            for position, sample in enumerate(samples, start=1):
                per_sample.append(
                    self._student_probabilities(
                        student, sample["passage"], sample["question"], sample["options"]
                    )
                )
                if progress_callback is not None:
                    progress_callback(
                        model_name, model_position, len(self.model_names), position, len(samples)
                    )
            elapsed = time.time() - pass_started
            self.logger.info(
                "Pass %d/%d complete: %d samples in %.2fs (%.4fs per sample)",
                model_position, len(self.model_names), len(samples), elapsed, elapsed / max(1, len(samples)),
            )

            cache[model_name] = per_sample
            self._release(student)

        self.logger.info("=" * 70)
        self.logger.info("All passes finished; aggregating KDA_cont across %d models", len(cache))

        results = []
        for index, sample in enumerate(samples):
            model_probabilities = {name: cache[name][index] for name in self.model_names}
            results.append(
                self._aggregate(model_probabilities, sample["answer_idx"], sample["question"])
            )
        return results

    def score(
        self,
        passage: str,
        question: str,
        options: Sequence[str],
        answer_idx: int,
    ) -> Dict:
        """Compute the ensemble KDA_cont score for a single question.

        Requires strategy="resident". Returns the score together with the full per-model
        probability log so the weighted average can be audited sample by sample.
        """
        if not self.students:
            raise RuntimeError(
                "score() needs the models resident in memory. This ensemble was built "
                "with strategy='sequential' -- use score_dataset(samples) instead."
            )
        options = list(options)
        model_probabilities = {
            student.model_name: self._student_probabilities(student, passage, question, options)
            for student in self.students
        }
        return self._aggregate(model_probabilities, answer_idx, question)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    scorer = KDATiny()
    demo = scorer.score(
        "Puppy is a baby of dog.",
        "What is the term for baby dog?",
        ["cat", "puppy", "kitty", "cow"],
        answer_idx=1,
    )
    print(f"KDA_cont = {demo['kda_score']:.4f}")
    for name, stats in demo["per_model"].items():
        print(
            f"  {name}: P(R^q=1)={stats['p_correct_without_fact']:.4f} "
            f"-> P(R^q+f=1)={stats['p_correct_with_fact']:.4f} "
            f"(weight {stats['weight']:.4f})"
        )
