"""Strict, serializable configuration shared by notebook and CLI."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Experiment:
    name: str
    max_steps: int
    train_samples: int
    learning_rate: float = 5e-5
    batch_size: int = 1
    accumulation: int = 16
    eval_steps: int = 100
    save_steps: int = 100
    smoke: bool = False

    def __post_init__(self):
        import re

        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", self.name):
            raise ValueError("Unsafe experiment name")
        for name in (
            "max_steps",
            "train_samples",
            "batch_size",
            "accumulation",
            "eval_steps",
            "save_steps",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.save_steps != self.eval_steps:
            raise ValueError("Aligned save/eval steps required for best-checkpoint recovery")
        if not 0 < self.learning_rate < 0.01:
            raise ValueError("learning_rate out of range")


@dataclass(frozen=True)
class Config:
    training_schema_version: int = 2
    model_id: str = "UBC-NLP/AraT5v2-base-1024"
    dataset_id: str = "csebuetnlp/xlsum"
    model_revision: str = "main"
    dataset_revision: str = "main"
    seed: int = 42
    source_length: int = 1024
    target_length: int = 128
    validation_samples: int = 256
    test_samples: int = 512
    max_hours: float = 7.0
    session_hours: float = 3.0
    max_storage_gb: float = 40.0
    reserve_gb: float = 7.0
    mixed_precision: str = "no"
    experiments: list[Experiment] = field(default_factory=list)

    def __post_init__(self):
        if self.training_schema_version != 2:
            raise ValueError("Old training schema: use a fresh run directory for the numerical fix")
        for name in (
            "seed",
            "source_length",
            "target_length",
            "validation_samples",
            "test_samples",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.model_id != "UBC-NLP/AraT5v2-base-1024" or self.dataset_id != "csebuetnlp/xlsum":
            raise ValueError("This pipeline is validated for the requested AraT5v2 and XL-Sum")
        if not 32 <= self.source_length <= 1024 or not 8 <= self.target_length <= 256:
            raise ValueError("Unsupported sequence lengths")
        if self.mixed_precision not in {"fp16", "bf16", "no"}:
            raise ValueError("Invalid mixed_precision")
        if any(
            not math.isfinite(x) or x <= 0
            for x in (
                self.validation_samples,
                self.test_samples,
                self.max_hours,
                self.session_hours,
                self.max_storage_gb,
                self.reserve_gb,
            )
        ):
            raise ValueError("Limits must be positive")
        if not self.experiments or not self.experiments[0].smoke:
            raise ValueError("The first experiment must be a smoke test")
        if len({e.name for e in self.experiments}) != len(self.experiments):
            raise ValueError("Duplicate experiment names")
        if not any(not e.smoke for e in self.experiments):
            raise ValueError("At least one real experiment is required")

    def to_dict(self):
        return asdict(self)

    @property
    def fingerprint(self):
        value = self.to_dict()
        # Budgets may be raised on restart; training semantics may not change in place.
        for key in ("max_hours", "session_hours", "max_storage_gb", "reserve_gb"):
            value.pop(key)
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def load_config(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    raw["experiments"] = [Experiment(**e) for e in raw.get("experiments", [])]
    return Config(**raw)
