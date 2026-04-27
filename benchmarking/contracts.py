from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class RunStatus(StrEnum):
    COMPLETED = "completed"
    RECOVERED = "recovered"
    FAILED = "failed"


@dataclass(frozen=True)
class AnswerPayload:
    short_answer_text: str = ""
    full_response_text: str = ""


@dataclass(frozen=True)
class FailureInfo:
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecoveryInfo:
    source: str
    scored: bool
    evaluable: bool = False
    reliability: str = "none"
    recovery_mode: str = "none"
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunnerResult:
    status: RunStatus
    answer: AnswerPayload
    raw: dict[str, Any]
    runner_meta: dict[str, Any]
    failure: FailureInfo | None = None
    recovery: RecoveryInfo | None = None

    def should_score(self) -> bool:
        if self.status is RunStatus.COMPLETED:
            return True
        return (
            self.status is RunStatus.RECOVERED
            and self.recovery is not None
            and self.recovery.scored
            and self.recovery.evaluable
        )

    @property
    def short_answer_text(self) -> str:
        return self.answer.short_answer_text

    @property
    def full_response_text(self) -> str:
        return self.answer.full_response_text
