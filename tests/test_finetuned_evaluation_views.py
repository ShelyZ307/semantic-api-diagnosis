"""Regression checks for the canonical fine-tuned evaluation view registry."""

from pathlib import Path
import runpy


REQUIRED_VIEWS = {
    "seen_test",
    "unseen_family_test",
    "contract_dependent_seen_test",
    "contract_dependent_unseen_test",
    "seen_test_request_only",
    "seen_test_no_constraints",
    "unseen_family_test_request_only",
    "unseen_family_test_no_constraints",
}


def test_finetuned_evaluation_runner_includes_all_required_views() -> None:
    namespace = runpy.run_path("scripts/run_finetuned_evaluation.py", run_name="finetuned_evaluation_test")

    views = namespace["EVALUATION_VIEWS"]

    assert set(views) == REQUIRED_VIEWS
    assert all(Path(path).exists() for path in views.values())
