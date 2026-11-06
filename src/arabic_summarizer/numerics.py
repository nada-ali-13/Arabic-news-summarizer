"""Fail before unsafe optimizer updates; persist exact numerical failure evidence.

This module diagnoses failures. It never replaces NaN/Infinity, filters a bad
batch, changes the loss, or silently retries under another precision.
"""

import math
import numbers
from pathlib import Path

from .io import atomic_json


class NumericalFailure(RuntimeError):
    """The recorded training/evaluation state is unsafe to continue."""


def json_values(value):
    """Represent nonfinite values explicitly in standards-compliant JSON."""
    import torch

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().item() if value.numel() == 1 else tensor_summary(value)
    if isinstance(value, numbers.Real):
        value = float(value)
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, dict):
        return {str(k): json_values(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_values(v) for v in value]
    return value


def nonfinite_values(values):
    import torch

    bad = {}
    for key, value in values.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            value = value.detach().item()
        if isinstance(value, numbers.Real) and not math.isfinite(value):
            bad[key] = repr(float(value))
    return bad


def tensor_summary(tensor):
    import torch

    tensor = tensor.detach()
    finite = torch.isfinite(tensor)
    good = tensor[finite]
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "nonfinite_count": int((~finite).sum().item()),
        "min_finite": good.min().item() if good.numel() else None,
        "max_finite": good.max().item() if good.numel() else None,
    }


def batch_summary(inputs):
    result = {}
    for key in ("input_ids", "attention_mask", "labels", "decoder_input_ids"):
        if key in inputs:
            tensor = inputs[key].detach()
            result[key] = tensor_summary(tensor)
            # Exact bounded token evidence makes a failing row replayable, without
            # logging a full article string or allocating vocabulary-sized logits.
            result[key]["first_rows"] = tensor[:2, :1024].cpu().tolist()
    if "labels" in inputs:
        result["supervised_tokens_per_row"] = (inputs["labels"] != -100).sum(-1).cpu().tolist()
    return result


def fail(output_dir, phase, step, values, *, batch=None, details=None):
    import torch
    import transformers

    path = Path(output_dir) / "numerical_failure.json"
    report = {
        "phase": phase,
        "completed_global_step": step,
        "attempted_optimizer_step": step + 1 if phase in {"train_loss", "gradient"} else None,
        "nonfinite": nonfinite_values(values),
        "values": json_values(values),
        "batch": batch,
        "details": json_values(details or {}),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    atomic_json(path, report)
    bad = report["nonfinite"] or report["details"]
    raise NumericalFailure(
        f"Numerical safety failure: phase={phase}, completed_global_step={step}, "
        f"values={bad}. Diagnostic: {path}. No successful result was recorded."
    )


def check_logs(output_dir, state, logs):
    bad = nonfinite_values(logs or {})
    if bad:
        fail(output_dir, "trainer_log", state.global_step, logs, details={"epoch": state.epoch})


class ValidatedSeq2SeqCollator:
    """Keep dynamic label padding at -100 and reject zero-denominator CE batches."""

    def __init__(self, tokenizer, model, pad_to_multiple_of=None):
        from transformers import DataCollatorForSeq2Seq

        self.pad_id = tokenizer.pad_token_id
        self.vocab_size = model.config.vocab_size
        self.collator = DataCollatorForSeq2Seq(
            tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=pad_to_multiple_of
        )

    def __call__(self, features):
        # Do not mutate Dataset's original lists. The tokenizer does not pre-pad
        # targets; any explicit pad IDs are also ignored before shifting labels.
        rows = [
            {**row, "labels": [-100 if value == self.pad_id else value for value in row["labels"]]}
            for row in features
        ]
        if any(not row["labels"] or all(value == -100 for value in row["labels"]) for row in rows):
            raise ValueError(
                "Every target needs a supervised token; all -100 labels cause NaN mean CE"
            )
        batch = self.collator(rows)
        validate_batch(batch, self.vocab_size)
        return batch


def validate_batch(inputs, vocab_size):
    import torch

    for key in ("input_ids", "labels"):
        values = inputs[key]
        if values.ndim != 2 or values.numel() == 0:
            raise ValueError(f"Invalid {key} shape: {list(values.shape)}")
        if values.dtype != torch.long:
            raise ValueError(f"{key} must be torch.long, got {values.dtype}")
        active = values[values != -100] if key == "labels" else values
        if active.numel() and (active.min().item() < 0 or active.max().item() >= vocab_size):
            raise ValueError(f"{key} contains IDs outside model vocabulary [0, {vocab_size})")
    if (inputs["labels"] != -100).sum(-1).min().item() == 0:
        raise ValueError("Every target needs a supervised token; all -100 labels cause NaN mean CE")
    if inputs["attention_mask"].sum(-1).min().item() == 0:
        raise ValueError("Every source needs an unmasked token")


def checked_trainer_class():
    """Keep heavy model imports lazy, as in the rest of the application."""
    import torch
    from accelerate.utils import DistributedType
    from transformers import Seq2SeqTrainer

    class CheckedSeq2SeqTrainer(Seq2SeqTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.last_inputs = {}
            if self.accelerator.distributed_type != DistributedType.NO:
                raise ValueError("Checked training currently supports single-device CPU/Colab GPU")
            # The Trainer clips just before optimizer.step; a callback on_log is
            # too late. Unscale exactly once using Accelerate's normal AMP path.
            self.accelerator.clip_grad_norm_ = self.checked_clip_grad_norm

        def checked_clip_grad_norm(self, parameters, max_norm, norm_type=2):
            parameters = list(parameters)
            self.accelerator.unscale_gradients()
            try:
                return torch.nn.utils.clip_grad_norm_(
                    parameters, max_norm, norm_type=norm_type, error_if_nonfinite=True
                )
            except RuntimeError as exc:
                norms = [
                    torch.linalg.vector_norm(p.grad.detach(), norm_type)
                    for p in parameters
                    if p.grad is not None
                ]
                if not norms:
                    raise
                total = torch.linalg.vector_norm(torch.stack(norms), norm_type)
                if torch.isfinite(total).item():
                    raise  # Do not mislabel unrelated runtime failures as NaN.
                details = {
                    name: tensor_summary(p.grad)
                    for name, p in self.model.named_parameters()
                    if p.grad is not None and not torch.isfinite(p.grad).all().item()
                }
                fail(
                    self.args.output_dir,
                    "gradient",
                    self.state.global_step,
                    {"grad_norm": total},
                    batch=batch_summary(self.last_inputs),
                    details={
                        "parameters_with_nonfinite_gradients": details,
                        "all_gradient_elements_finite": not bool(details),
                        "clipping_error": str(exc),
                        "fp16": self.args.fp16,
                        "bf16": self.args.bf16,
                    },
                )

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            self.last_inputs = {
                key: value.detach()
                for key, value in inputs.items()
                if isinstance(value, torch.Tensor)
            }
            validate_batch(inputs, model.config.vocab_size)
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
            if not torch.isfinite(loss.detach()).all().item():
                fail(
                    self.args.output_dir,
                    "train_loss" if model.training else "eval_loss",
                    self.state.global_step,
                    {"loss": loss},
                    batch=batch_summary(inputs),
                    details={
                        "fp16": self.args.fp16,
                        "bf16": self.args.bf16,
                        "logits": tensor_summary(outputs.logits),
                    },
                )
            return (loss, outputs) if return_outputs else loss

        def prediction_step(self, model, inputs, *args, **kwargs):
            validate_batch(inputs, model.config.vocab_size)
            loss, predictions, labels = super().prediction_step(model, inputs, *args, **kwargs)
            if loss is not None and not torch.isfinite(loss).all().item():
                fail(
                    self.args.output_dir,
                    "eval_loss",
                    self.state.global_step,
                    {"eval_loss": loss},
                    batch=batch_summary(inputs),
                )
            return loss, predictions, labels

    return CheckedSeq2SeqTrainer
