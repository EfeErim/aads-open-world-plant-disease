"""Fail-closed routing policy extracted from the thesis inference contract."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Decision(StrEnum):
    ACCEPT = "accept"
    REVIEW = "review"


@dataclass(frozen=True)
class Candidate:
    crop: str
    part: str
    disease: str
    similarity: float
    margin: float
    negative_gap: float

    @property
    def target_id(self) -> str:
        return f"{self.crop}__{self.part}"


@dataclass(frozen=True)
class PolicyResult:
    decision: Decision
    reason: str
    target_id: str | None = None
    disease: str | None = None


@dataclass(frozen=True)
class SafetyPolicy:
    supported_targets: frozenset[str]
    min_similarity: float = 0.20
    min_margin: float = 0.02
    min_negative_gap: float = 0.0

    def decide(self, candidate: Candidate | None) -> PolicyResult:
        if candidate is None:
            return PolicyResult(Decision.REVIEW, "router_uncertain")
        if candidate.target_id not in self.supported_targets:
            return PolicyResult(Decision.REVIEW, "unsupported_crop_or_part")
        if candidate.similarity < self.min_similarity:
            return PolicyResult(Decision.REVIEW, "low_similarity")
        if candidate.margin < self.min_margin:
            return PolicyResult(Decision.REVIEW, "low_margin")
        if candidate.negative_gap < self.min_negative_gap:
            return PolicyResult(Decision.REVIEW, "negative_prototype_too_close")
        if not candidate.disease.strip():
            return PolicyResult(Decision.REVIEW, "empty_disease_label")
        return PolicyResult(
            Decision.ACCEPT,
            "supported_and_confident",
            target_id=candidate.target_id,
            disease=candidate.disease,
        )
