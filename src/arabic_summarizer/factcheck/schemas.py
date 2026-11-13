from dataclasses import dataclass, field
from enum import Enum


class Verdict(str, Enum):
    SUPPORTED = "مدعوم"
    REFUTED = "مناقَض"
    INSUFFICIENT_EVIDENCE = "أدلة غير كافية"


@dataclass(frozen=True)
class Claim:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class Evidence:
    text: str
    url: str
    publisher: str
    date: str = ""
    kind: str = "news"
    rating: str = ""
    claim_date: str = ""
    title: str = ""
    retrieved_at: str = ""


@dataclass
class Assessment:
    claim: Claim
    verdict: Verdict = Verdict.INSUFFICIENT_EVIDENCE
    evidence: list[Evidence] = field(default_factory=list)
    relations: list[tuple[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
