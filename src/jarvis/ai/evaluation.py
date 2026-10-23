"""Offline scoring helpers for fixed evaluation sets; no model is loaded.

Exact-answer cases are scored mechanically. Open-ended cases require explicit
criterion-by-criterion human scores; no model is used as a judge. CLI output
contains aggregate metrics only and never echoes prompts or model responses.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata
from types import MappingProxyType
from typing import Mapping, Sequence


_MAX_JSON_BYTES = 2_000_000
_MAX_CASES = 500
_MAX_PROMPT_LENGTH = 8_000
_MAX_RESPONSE_LENGTH = 16_000
_SAFE_ID = re.compile(r"[a-z][a-z0-9_-]{1,63}\Z")
_SAFE_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,79}\Z")
_DEFAULT_DATASET = (
    Path(__file__).resolve().parents[3]
    / "benchmarks"
    / "model_quality_es_v1.json"
)


class EvaluationInputError(ValueError):
    """An evaluation dataset or submission has an invalid structure."""


class BenchmarkCategory(str, Enum):
    MATH = "math"
    INSTRUCTION_FOLLOWING = "instruction_following"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    PROGRAMMING = "programming"
    REASONING = "reasoning"
    AMBIGUITY = "ambiguity"
    UNCERTAINTY = "uncertainty"
    SAFETY = "safety"
    PRIVACY = "privacy"


@dataclass(frozen=True, slots=True)
class RubricCriterion:
    criterion_id: str
    description: str


@dataclass(frozen=True, slots=True, repr=False)
class BenchmarkCase:
    case_id: str
    category: BenchmarkCategory
    prompt: str
    expected: str | None
    rubric: tuple[RubricCriterion, ...]
    safety_critical: bool

    @property
    def uses_rubric(self) -> bool:
        return bool(self.rubric)

    def __repr__(self) -> str:
        return (
            f"BenchmarkCase(case_id={self.case_id!r}, "
            f"category={self.category.value!r}, <prompt redacted>)"
        )


@dataclass(frozen=True, slots=True)
class Benchmark:
    benchmark_id: str
    version: int
    fingerprint: str
    cases: tuple[BenchmarkCase, ...]


@dataclass(frozen=True, slots=True, repr=False)
class EvaluationRun:
    run_id: str
    model_id: str
    benchmark_id: str
    responses: Mapping[str, str]
    rubric_scores: Mapping[str, Mapping[str, bool]]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "responses",
            MappingProxyType(dict(self.responses)),
        )
        object.__setattr__(
            self,
            "rubric_scores",
            MappingProxyType(
                {
                    case_id: MappingProxyType(dict(scores))
                    for case_id, scores in self.rubric_scores.items()
                }
            ),
        )

    def __repr__(self) -> str:
        return (
            f"EvaluationRun(run_id={self.run_id!r}, model_id={self.model_id!r}, "
            f"benchmark_id={self.benchmark_id!r}, <responses redacted>)"
        )


@dataclass(frozen=True, slots=True)
class CategoryMetrics:
    case_count: int
    scored_count: int
    score_mean: float | None
    exact_accuracy: float | None
    rubric_mean: float | None
    pending_review_count: int
    critical_failures: int


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    benchmark_id: str
    benchmark_version: int
    benchmark_fingerprint: str
    run_id: str
    model_id: str
    case_count: int
    scored_count: int
    pending_review_count: int
    overall_score: float | None
    exact_case_count: int
    exact_accuracy: float | None
    rubric_case_count: int
    rubric_reviewed_count: int
    rubric_mean: float | None
    critical_failures: int
    categories: Mapping[str, CategoryMetrics]

    def to_dict(self) -> dict[str, object]:
        return {
            "benchmark_id": self.benchmark_id,
            "benchmark_version": self.benchmark_version,
            "benchmark_fingerprint": self.benchmark_fingerprint,
            "run_id": self.run_id,
            "model_id": self.model_id,
            "case_count": self.case_count,
            "scored_count": self.scored_count,
            "pending_review_count": self.pending_review_count,
            "overall_score": self.overall_score,
            "exact_case_count": self.exact_case_count,
            "exact_accuracy": self.exact_accuracy,
            "rubric_case_count": self.rubric_case_count,
            "rubric_reviewed_count": self.rubric_reviewed_count,
            "rubric_mean": self.rubric_mean,
            "critical_failures": self.critical_failures,
            "categories": {
                name: {
                    "case_count": metrics.case_count,
                    "scored_count": metrics.scored_count,
                    "score_mean": metrics.score_mean,
                    "exact_accuracy": metrics.exact_accuracy,
                    "rubric_mean": metrics.rubric_mean,
                    "pending_review_count": metrics.pending_review_count,
                    "critical_failures": metrics.critical_failures,
                }
                for name, metrics in self.categories.items()
            },
        }


@dataclass(frozen=True, slots=True)
class CategoryDelta:
    score_mean: float | None
    exact_accuracy: float | None
    rubric_mean: float | None
    critical_failures: int


@dataclass(frozen=True, slots=True)
class EvaluationComparison:
    benchmark_id: str
    benchmark_fingerprint: str
    baseline_run_id: str
    candidate_run_id: str
    baseline_model_id: str
    candidate_model_id: str
    overall_score_delta: float | None
    exact_accuracy_delta: float | None
    rubric_mean_delta: float | None
    critical_failure_delta: int
    categories: Mapping[str, CategoryDelta]

    def to_dict(self) -> dict[str, object]:
        return {
            "benchmark_id": self.benchmark_id,
            "benchmark_fingerprint": self.benchmark_fingerprint,
            "baseline_run_id": self.baseline_run_id,
            "candidate_run_id": self.candidate_run_id,
            "baseline_model_id": self.baseline_model_id,
            "candidate_model_id": self.candidate_model_id,
            "overall_score_delta": self.overall_score_delta,
            "exact_accuracy_delta": self.exact_accuracy_delta,
            "rubric_mean_delta": self.rubric_mean_delta,
            "critical_failure_delta": self.critical_failure_delta,
            "categories": {
                name: {
                    "score_mean": delta.score_mean,
                    "exact_accuracy": delta.exact_accuracy,
                    "rubric_mean": delta.rubric_mean,
                    "critical_failures": delta.critical_failures,
                }
                for name, delta in self.categories.items()
            },
        }


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise EvaluationInputError("JSON contains duplicate object keys.")
        result[key] = value
    return result


def _read_json(path: str | Path) -> object:
    try:
        raw = Path(path).read_bytes()
    except OSError:
        raise EvaluationInputError("Evaluation input could not be read.") from None
    if len(raw) > _MAX_JSON_BYTES:
        raise EvaluationInputError("Evaluation input exceeds the size limit.")
    try:
        text = raw.decode("utf-8")
        return json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise EvaluationInputError("Evaluation input is not valid UTF-8 JSON.") from None


def _require_keys(
    value: object,
    required: set[str],
    optional: set[str] = frozenset(),
) -> dict[str, object]:
    if type(value) is not dict:
        raise EvaluationInputError("Evaluation object has an invalid structure.")
    keys = set(value)
    if not required <= keys or keys - required - optional:
        raise EvaluationInputError("Evaluation object has missing or unknown fields.")
    return value


def _require_safe_id(value: object, field: str) -> str:
    if type(value) is not str or _SAFE_ID.fullmatch(value) is None:
        raise EvaluationInputError(f"Evaluation {field} is invalid.")
    return value


def _require_safe_label(value: object, field: str) -> str:
    if type(value) is not str or _SAFE_LABEL.fullmatch(value) is None:
        raise EvaluationInputError(f"Evaluation {field} is invalid.")
    return value


def load_benchmark(path: str | Path = _DEFAULT_DATASET) -> Benchmark:
    document = _require_keys(
        _read_json(path),
        {"benchmark_id", "version", "cases"},
    )
    benchmark_id = _require_safe_label(
        document["benchmark_id"],
        "benchmark identifier",
    )
    version = document["version"]
    cases_data = document["cases"]
    if type(version) is not int or version <= 0:
        raise EvaluationInputError("Benchmark version is invalid.")
    if type(cases_data) is not list or not 1 <= len(cases_data) <= _MAX_CASES:
        raise EvaluationInputError("Benchmark case count is invalid.")

    cases: list[BenchmarkCase] = []
    seen_ids: set[str] = set()
    required_case_keys = {
        "id",
        "category",
        "prompt",
        "expected",
        "rubric",
        "safety_critical",
    }
    for raw_case in cases_data:
        item = _require_keys(raw_case, required_case_keys)
        case_id = _require_safe_id(item["id"], "case identifier")
        if case_id in seen_ids:
            raise EvaluationInputError("Benchmark contains duplicate case IDs.")
        seen_ids.add(case_id)
        try:
            category = BenchmarkCategory(item["category"])
        except (TypeError, ValueError):
            raise EvaluationInputError("Benchmark category is unsupported.") from None

        prompt = item["prompt"]
        if (
            type(prompt) is not str
            or not prompt.strip()
            or len(prompt) > _MAX_PROMPT_LENGTH
        ):
            raise EvaluationInputError("Benchmark prompt is invalid.")

        expected = item["expected"]
        if expected is not None and (
            type(expected) is not str
            or not expected.strip()
            or len(expected) > _MAX_RESPONSE_LENGTH
        ):
            raise EvaluationInputError("Benchmark expected answer is invalid.")

        raw_rubric = item["rubric"]
        if type(raw_rubric) is not list or len(raw_rubric) > 20:
            raise EvaluationInputError("Benchmark rubric is invalid.")
        rubric: list[RubricCriterion] = []
        rubric_ids: set[str] = set()
        for raw_criterion in raw_rubric:
            criterion_data = _require_keys(
                raw_criterion,
                {"id", "criterion"},
            )
            criterion_id = _require_safe_id(
                criterion_data["id"],
                "rubric criterion identifier",
            )
            description = criterion_data["criterion"]
            if (
                criterion_id in rubric_ids
                or type(description) is not str
                or not description.strip()
                or len(description) > 500
            ):
                raise EvaluationInputError("Benchmark rubric criterion is invalid.")
            rubric_ids.add(criterion_id)
            rubric.append(RubricCriterion(criterion_id, description))

        if (expected is None) == (not rubric):
            raise EvaluationInputError(
                "Each case must have exactly one expected answer or rubric."
            )
        safety_critical = item["safety_critical"]
        if type(safety_critical) is not bool:
            raise EvaluationInputError("Benchmark safety flag is invalid.")
        cases.append(
            BenchmarkCase(
                case_id=case_id,
                category=category,
                prompt=prompt,
                expected=expected,
                rubric=tuple(rubric),
                safety_critical=safety_critical,
            )
        )
    canonical = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    fingerprint = hashlib.sha256(canonical).hexdigest()
    return Benchmark(benchmark_id, version, fingerprint, tuple(cases))


def load_run(path: str | Path, benchmark: Benchmark) -> EvaluationRun:
    document = _require_keys(
        _read_json(path),
        {"run_id", "model_id", "benchmark_id", "responses"},
        {"rubric_scores"},
    )
    run_id = _require_safe_label(document["run_id"], "run identifier")
    model_id = _require_safe_label(document["model_id"], "model label")
    benchmark_id = _require_safe_label(
        document["benchmark_id"],
        "benchmark identifier",
    )
    if benchmark_id != benchmark.benchmark_id:
        raise EvaluationInputError("Run benchmark does not match the dataset.")

    case_by_id = {case.case_id: case for case in benchmark.cases}
    raw_responses = document["responses"]
    if type(raw_responses) is not dict:
        raise EvaluationInputError("Run responses have an invalid structure.")
    responses: dict[str, str] = {}
    for case_id, response in raw_responses.items():
        if case_id not in case_by_id:
            raise EvaluationInputError("Run contains an unknown case ID.")
        if (
            type(response) is not str
            or len(response) > _MAX_RESPONSE_LENGTH
        ):
            raise EvaluationInputError("Run response has an invalid value.")
        responses[case_id] = response

    raw_scores = document.get("rubric_scores", {})
    if type(raw_scores) is not dict:
        raise EvaluationInputError("Run rubric scores have an invalid structure.")
    rubric_scores: dict[str, dict[str, bool]] = {}
    for case_id, raw_case_scores in raw_scores.items():
        case = case_by_id.get(case_id)
        if case is None or not case.uses_rubric:
            raise EvaluationInputError("Run rubric score references an invalid case.")
        if case_id not in responses or not responses[case_id].strip():
            raise EvaluationInputError(
                "Rubric scores require a non-empty case response."
            )
        if type(raw_case_scores) is not dict:
            raise EvaluationInputError("Run rubric scores have an invalid structure.")
        expected_criterion_ids = {criterion.criterion_id for criterion in case.rubric}
        if set(raw_case_scores) != expected_criterion_ids or any(
            type(score) is not bool for score in raw_case_scores.values()
        ):
            raise EvaluationInputError(
                "Run rubric scores must cover each criterion with a boolean."
            )
        rubric_scores[case_id] = dict(raw_case_scores)

    return EvaluationRun(
        run_id=run_id,
        model_id=model_id,
        benchmark_id=benchmark_id,
        responses=responses,
        rubric_scores=rubric_scores,
    )


def _normalize_exact(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def evaluate_run(benchmark: Benchmark, run: EvaluationRun) -> EvaluationReport:
    if run.benchmark_id != benchmark.benchmark_id:
        raise EvaluationInputError("Run benchmark does not match the dataset.")

    category_cases: dict[str, list[BenchmarkCase]] = {}
    case_scores: dict[str, float | None] = {}
    exact_correct: dict[str, bool] = {}
    rubric_case_scores: dict[str, float] = {}
    pending_review_count = 0
    critical_failures = 0

    for case in benchmark.cases:
        category = case.category.value
        category_cases.setdefault(category, []).append(case)
        response = run.responses.get(case.case_id)
        if not case.uses_rubric:
            correct = (
                response is not None
                and bool(response.strip())
                and _normalize_exact(response) == _normalize_exact(case.expected or "")
            )
            exact_correct[case.case_id] = correct
            case_scores[case.case_id] = float(correct)
            if case.safety_critical and not correct:
                critical_failures += 1
            continue

        scores = run.rubric_scores.get(case.case_id)
        if response is None or not response.strip() or scores is None:
            case_scores[case.case_id] = None
            pending_review_count += 1
            continue
        score = sum(scores.values()) / len(case.rubric)
        rubric_case_scores[case.case_id] = score
        case_scores[case.case_id] = score
        if case.safety_critical and score < 1.0:
            critical_failures += 1

    categories: dict[str, CategoryMetrics] = {}
    for name, cases in category_cases.items():
        scores = [
            case_scores[case.case_id]
            for case in cases
            if case_scores[case.case_id] is not None
        ]
        exact_cases = [case for case in cases if not case.uses_rubric]
        exact_values = [exact_correct[case.case_id] for case in exact_cases]
        rubric_values = [
            rubric_case_scores[case.case_id]
            for case in cases
            if case.case_id in rubric_case_scores
        ]
        categories[name] = CategoryMetrics(
            case_count=len(cases),
            scored_count=len(scores),
            score_mean=_mean([float(score) for score in scores]),
            exact_accuracy=(
                round(sum(exact_values) / len(exact_values), 6)
                if exact_values
                else None
            ),
            rubric_mean=_mean(rubric_values),
            pending_review_count=sum(
                case.uses_rubric and case_scores[case.case_id] is None
                for case in cases
            ),
            critical_failures=sum(
                case.safety_critical
                and case_scores[case.case_id] is not None
                and case_scores[case.case_id] < 1.0
                for case in cases
            ),
        )

    exact_cases_all = [case for case in benchmark.cases if not case.uses_rubric]
    exact_correct_count = sum(exact_correct.values())
    rubric_values_all = list(rubric_case_scores.values())
    all_scored_values = [
        float(score) for score in case_scores.values() if score is not None
    ]
    return EvaluationReport(
        benchmark_id=benchmark.benchmark_id,
        benchmark_version=benchmark.version,
        benchmark_fingerprint=benchmark.fingerprint,
        run_id=run.run_id,
        model_id=run.model_id,
        case_count=len(benchmark.cases),
        scored_count=len(all_scored_values),
        pending_review_count=pending_review_count,
        overall_score=_mean(all_scored_values),
        exact_case_count=len(exact_cases_all),
        exact_accuracy=(
            round(exact_correct_count / len(exact_cases_all), 6)
            if exact_cases_all
            else None
        ),
        rubric_case_count=len(benchmark.cases) - len(exact_cases_all),
        rubric_reviewed_count=len(rubric_values_all),
        rubric_mean=_mean(rubric_values_all),
        critical_failures=critical_failures,
        categories=MappingProxyType(categories),
    )


def compare_reports(
    baseline: EvaluationReport,
    candidate: EvaluationReport,
) -> EvaluationComparison:
    if (
        baseline.benchmark_id != candidate.benchmark_id
        or baseline.benchmark_version != candidate.benchmark_version
        or baseline.benchmark_fingerprint != candidate.benchmark_fingerprint
        or set(baseline.categories) != set(candidate.categories)
    ):
        raise EvaluationInputError(
            "Evaluation reports use incompatible benchmark definitions."
        )

    def delta(left: float | None, right: float | None) -> float | None:
        if left is None or right is None:
            return None
        return round(right - left, 6)

    category_deltas: dict[str, CategoryDelta] = {}
    for name in baseline.categories:
        left = baseline.categories[name]
        right = candidate.categories[name]
        category_deltas[name] = CategoryDelta(
            score_mean=delta(left.score_mean, right.score_mean),
            exact_accuracy=delta(left.exact_accuracy, right.exact_accuracy),
            rubric_mean=delta(left.rubric_mean, right.rubric_mean),
            critical_failures=right.critical_failures - left.critical_failures,
        )
    return EvaluationComparison(
        benchmark_id=baseline.benchmark_id,
        benchmark_fingerprint=baseline.benchmark_fingerprint,
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        baseline_model_id=baseline.model_id,
        candidate_model_id=candidate.model_id,
        overall_score_delta=delta(baseline.overall_score, candidate.overall_score),
        exact_accuracy_delta=delta(
            baseline.exact_accuracy,
            candidate.exact_accuracy,
        ),
        rubric_mean_delta=delta(baseline.rubric_mean, candidate.rubric_mean),
        critical_failure_delta=(
            candidate.critical_failures - baseline.critical_failures
        ),
        categories=MappingProxyType(category_deltas),
    )


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score saved model evaluation responses without loading a model."
    )
    parser.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args(argv)

    try:
        benchmark = load_benchmark(args.dataset)
        run = load_run(args.run, benchmark)
        report = evaluate_run(benchmark, run)
        result: dict[str, object] = {"report": report.to_dict()}
        if args.compare is not None:
            baseline = load_run(args.compare, benchmark)
            baseline_report = evaluate_run(benchmark, baseline)
            result["comparison"] = compare_reports(
                baseline_report,
                report,
            ).to_dict()
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    except EvaluationInputError:
        print("Evaluation input could not be validated.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
