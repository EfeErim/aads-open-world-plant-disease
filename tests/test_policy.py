import pytest

from aads_public.policy import Candidate, Decision, SafetyPolicy

POLICY = SafetyPolicy(frozenset({"tomato__leaf", "grape__fruit"}))


def test_supported_confident_candidate_is_accepted() -> None:
    result = POLICY.decide(Candidate("tomato", "leaf", "late_blight", 0.82, 0.31, 0.22))
    assert result.decision is Decision.ACCEPT
    assert result.target_id == "tomato__leaf"
    assert result.disease == "late_blight"


@pytest.mark.parametrize(
    ("candidate", "reason"),
    [
        (None, "router_uncertain"),
        (Candidate("tomato", "fruit", "disease", 0.9, 0.4, 0.2), "unsupported_crop_or_part"),
        (Candidate("tomato", "leaf", "disease", 0.1, 0.4, 0.2), "low_similarity"),
        (Candidate("tomato", "leaf", "disease", 0.8, 0.01, 0.2), "low_margin"),
        (Candidate("tomato", "leaf", "disease", 0.8, 0.3, -0.1), "negative_prototype_too_close"),
        (Candidate("tomato", "leaf", "", 0.8, 0.3, 0.2), "empty_disease_label"),
    ],
)
def test_unsafe_candidates_fail_closed(candidate: Candidate | None, reason: str) -> None:
    result = POLICY.decide(candidate)
    assert result.decision is Decision.REVIEW
    assert result.reason == reason
    assert result.disease is None
