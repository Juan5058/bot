from __future__ import annotations

import json
from pathlib import Path

import pytest

from jarvis.ai.evaluation import (
    BenchmarkCategory,
    EvaluationInputError,
    compare_reports,
    evaluate_run,
    load_benchmark,
    load_run,
)


DATASET = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "model_quality_es_v1.json"
)


def _write_json(path: Path, value: object) -> Path:
    path.write_text(
        json.dumps(value, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _run_document(benchmark_id: str, **changes: object) -> dict[str, object]:
    document: dict[str, object] = {
        "run_id": "test-run-1",
        "model_id": "test-model",
        "benchmark_id": benchmark_id,
        "responses": {},
    }
    document.update(changes)
    return document


def test_spanish_benchmark_has_at_least_fifty_cases_and_all_required_categories():
    benchmark = load_benchmark(DATASET)

    assert len(benchmark.cases) >= 50
    assert {case.category for case in benchmark.cases} == set(BenchmarkCategory)
    assert len({case.case_id for case in benchmark.cases}) == len(benchmark.cases)
    assert all(case.prompt.strip() for case in benchmark.cases)
    assert all(
        (case.expected is not None) != bool(case.rubric)
        for case in benchmark.cases
    )
    assert any(case.safety_critical for case in benchmark.cases)


def test_exact_answers_are_normalized_and_report_excludes_response_text(tmp_path):
    benchmark = load_benchmark(DATASET)
    case = next(case for case in benchmark.cases if case.case_id == "math-001")
    run_path = _write_json(
        tmp_path / "run.json",
        _run_document(
            benchmark.benchmark_id,
            responses={case.case_id: "  ５６  "},
        ),
    )

    run = load_run(run_path, benchmark)
    report = evaluate_run(benchmark, run)

    assert report.exact_accuracy is not None
    assert report.exact_accuracy > 0
    assert "responses" not in report.to_dict()
    assert "５６" not in json.dumps(report.to_dict(), ensure_ascii=False)


def test_missing_exact_answers_count_as_incorrect_not_success(tmp_path):
    benchmark = load_benchmark(DATASET)
    run_path = _write_json(
        tmp_path / "run.json",
        _run_document(benchmark.benchmark_id),
    )

    report = evaluate_run(benchmark, load_run(run_path, benchmark))

    assert report.exact_accuracy == 0
    assert report.pending_review_count == report.rubric_case_count
    assert report.scored_count == report.exact_case_count


def test_rubric_results_require_complete_explicit_boolean_review(tmp_path):
    benchmark = load_benchmark(DATASET)
    case = next(case for case in benchmark.cases if case.case_id == "summary-001")
    response = "La biblioteca abrió el martes a las ocho; la sala infantil cerró por mantenimiento y el horario normal vuelve el miércoles."
    case_scores = {criterion.criterion_id: True for criterion in case.rubric}
    run_path = _write_json(
        tmp_path / "run.json",
        _run_document(
            benchmark.benchmark_id,
            responses={case.case_id: response},
            rubric_scores={case.case_id: case_scores},
        ),
    )

    report = evaluate_run(benchmark, load_run(run_path, benchmark))

    assert report.rubric_reviewed_count == 1
    assert report.rubric_mean == 1
    assert report.critical_failures == 0
    assert response not in json.dumps(report.to_dict(), ensure_ascii=False)


def test_safety_rubric_failure_is_counted_as_critical(tmp_path):
    benchmark = load_benchmark(DATASET)
    case = next(case for case in benchmark.cases if case.case_id == "safety-004")
    run_path = _write_json(
        tmp_path / "run.json",
        _run_document(
            benchmark.benchmark_id,
            responses={case.case_id: "Sí, la respuesta del modelo autoriza la operación."},
            rubric_scores={
                case.case_id: {
                    "no_authorization": False,
                    "independent_control": False,
                    "no_execution": True,
                }
            },
        ),
    )

    report = evaluate_run(benchmark, load_run(run_path, benchmark))

    assert report.critical_failures == 1
    assert report.categories["safety"].rubric_mean is not None
    assert report.categories["safety"].rubric_mean < 1


def test_comparison_reports_deltas_for_same_benchmark(tmp_path):
    benchmark = load_benchmark(DATASET)
    case = next(case for case in benchmark.cases if case.case_id == "math-001")
    baseline_path = _write_json(
        tmp_path / "baseline.json",
        _run_document(
            benchmark.benchmark_id,
            run_id="baseline-1",
            model_id="small-model",
            responses={case.case_id: "9"},
        ),
    )
    candidate_path = _write_json(
        tmp_path / "candidate.json",
        _run_document(
            benchmark.benchmark_id,
            run_id="candidate-1",
            model_id="candidate-model",
            responses={case.case_id: "56"},
        ),
    )
    baseline = evaluate_run(benchmark, load_run(baseline_path, benchmark))
    candidate = evaluate_run(benchmark, load_run(candidate_path, benchmark))

    comparison = compare_reports(baseline, candidate)

    assert comparison.baseline_model_id == "small-model"
    assert comparison.candidate_model_id == "candidate-model"
    assert comparison.categories["math"].exact_accuracy == 0.2


def test_comparison_rejects_same_version_with_changed_benchmark_content(tmp_path):
    original = load_benchmark(DATASET)
    document = json.loads(DATASET.read_text(encoding="utf-8"))
    document["cases"][0]["prompt"] += " Responde de forma breve."
    changed_path = _write_json(tmp_path / "changed-dataset.json", document)
    changed = load_benchmark(changed_path)

    original_run_path = _write_json(
        tmp_path / "original-run.json",
        _run_document(original.benchmark_id),
    )
    changed_run_path = _write_json(
        tmp_path / "changed-run.json",
        _run_document(changed.benchmark_id, run_id="changed-run"),
    )
    original_report = evaluate_run(
        original,
        load_run(original_run_path, original),
    )
    changed_report = evaluate_run(
        changed,
        load_run(changed_run_path, changed),
    )

    with pytest.raises(EvaluationInputError, match="incompatible"):
        compare_reports(original_report, changed_report)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda doc: doc.update(extra="not allowed"),
        lambda doc: doc.update(responses={"unknown-case": "text"}),
        lambda doc: doc.update(run_id="C:\\secret\\run"),
        lambda doc: doc.update(rubric_scores={"math-001": {"x": True}}),
    ],
)
def test_invalid_run_structures_are_rejected(tmp_path, mutation):
    benchmark = load_benchmark(DATASET)
    document = _run_document(benchmark.benchmark_id)
    mutation(document)
    path = _write_json(tmp_path / "run.json", document)

    with pytest.raises(EvaluationInputError):
        load_run(path, benchmark)


def test_duplicate_json_keys_are_rejected(tmp_path):
    benchmark = load_benchmark(DATASET)
    path = tmp_path / "run.json"
    path.write_text(
        '{"run_id":"first","run_id":"second","model_id":"m",'
        f'"benchmark_id":"{benchmark.benchmark_id}","responses":{{}}}}',
        encoding="utf-8",
    )

    with pytest.raises(EvaluationInputError, match="duplicate"):
        load_run(path, benchmark)


def test_dataset_rejects_unknown_case_fields_and_duplicate_case_ids(tmp_path):
    benchmark_data = json.loads(DATASET.read_text(encoding="utf-8"))
    benchmark_data["cases"][0]["unknown"] = True
    path = _write_json(tmp_path / "dataset.json", benchmark_data)

    with pytest.raises(EvaluationInputError):
        load_benchmark(path)

    benchmark_data["cases"][0].pop("unknown")
    benchmark_data["cases"][1]["id"] = benchmark_data["cases"][0]["id"]
    path = _write_json(tmp_path / "duplicate.json", benchmark_data)

    with pytest.raises(EvaluationInputError, match="duplicate"):
        load_benchmark(path)


def test_incomplete_manual_rubric_is_rejected(tmp_path):
    benchmark = load_benchmark(DATASET)
    case = next(case for case in benchmark.cases if case.case_id == "summary-001")
    run_path = _write_json(
        tmp_path / "run.json",
        _run_document(
            benchmark.benchmark_id,
            responses={case.case_id: "Resumen."},
            rubric_scores={case.case_id: {case.rubric[0].criterion_id: True}},
        ),
    )

    with pytest.raises(EvaluationInputError, match="each criterion"):
        load_run(run_path, benchmark)


def test_cli_prints_aggregate_only_and_never_response_text(tmp_path, capsys):
    from jarvis.ai.evaluation import _main

    benchmark = load_benchmark(DATASET)
    response = "distinctive-private-response"
    case = next(case for case in benchmark.cases if case.case_id == "math-001")
    run_path = _write_json(
        tmp_path / "run.json",
        _run_document(
            benchmark.benchmark_id,
            responses={case.case_id: response},
        ),
    )

    result = _main(["--dataset", str(DATASET), "--run", str(run_path)])
    captured = capsys.readouterr()

    assert result == 0
    assert response not in captured.out
    assert response not in captured.err
    assert '"case_count": 50' in captured.out
