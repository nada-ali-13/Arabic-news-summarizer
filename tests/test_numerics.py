"""Numerical failures must be attributable and must precede optimizer mutation."""

from dataclasses import replace
from functools import wraps
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from arabic_summarizer.config import load_config
from arabic_summarizer.io import atomic_json, read_json
from arabic_summarizer.numerics import NumericalFailure, ValidatedSeq2SeqCollator, check_logs
from arabic_summarizer.training import build_trainer, resolve_revisions

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def components():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast, T5Config, T5ForConditionalGeneration

    torch.set_num_threads(1)
    backend = Tokenizer(
        WordLevel(
            {"<pad>": 0, "</s>": 1, "<unk>": 2, "خبر": 3, "مكتبة": 4, "جديدة": 5}, unk_token="<unk>"
        )
    )
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(single="$A </s>", special_tokens=[("</s>", 1)])
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        eos_token="</s>",
        unk_token="<unk>",
        model_input_names=["input_ids", "attention_mask"],
    )
    # Mirror the real checkpoint's architecture choices, rather than tied/relu T5 defaults.
    model = T5ForConditionalGeneration(
        T5Config(
            vocab_size=6,
            d_model=16,
            d_ff=32,
            d_kv=8,
            num_layers=1,
            num_decoder_layers=1,
            num_heads=2,
            pad_token_id=0,
            eos_token_id=1,
            decoder_start_token_id=0,
            feed_forward_proj="gated-gelu",
            tie_word_embeddings=False,
        )
    )
    model.config.use_cache = False
    model.generation_config.max_length = 8
    return model, tokenizer


def make_trainer(tmp_path, components):
    model, tokenizer = components
    config = replace(
        load_config(ROOT / "configs/dev.json"),
        source_length=32,
        target_length=8,
        mixed_precision="no",
    )
    rows = [{"input_ids": [3, 4, 5, 1], "attention_mask": [1, 1, 1, 1], "labels": [4, 5, 1]}] * 8
    return build_trainer(config, config.experiments[0], tmp_path, model, tokenizer, rows, rows[:2])


def test_padding_and_decoder_shift_are_correct(components):
    model, tokenizer = components
    collator = ValidatedSeq2SeqCollator(tokenizer, model, pad_to_multiple_of=8)
    features = [
        {"input_ids": [3, 4, 1], "attention_mask": [1, 1, 1], "labels": [4, 5, 1]},
        {"input_ids": [3, 1], "attention_mask": [1, 1], "labels": [3, 1]},
    ]
    batch = collator(features)
    assert batch["labels"].tolist() == [
        [4, 5, 1, -100, -100, -100, -100, -100],
        [3, 1, -100, -100, -100, -100, -100, -100],
    ]
    assert batch["decoder_input_ids"][0].tolist() == [0, 4, 5, 1, 0, 0, 0, 0]
    assert features[1]["labels"] == [3, 1]  # No in-place mutation of cached data.
    assert torch.isfinite(model(**batch).loss)


def test_zero_supervision_reproduces_native_nan_and_is_rejected(components):
    model, tokenizer = components
    logits = torch.randn(2, 6)
    assert torch.isnan(torch.nn.functional.cross_entropy(logits, torch.full((2,), -100)))
    collator = ValidatedSeq2SeqCollator(tokenizer, model)
    with pytest.raises(ValueError, match="supervised token"):
        collator([{"input_ids": [3, 1], "attention_mask": [1, 1], "labels": [0, 0]}])


@pytest.mark.parametrize("key,value", [("input_ids", [6, 1]), ("labels", [6, 1])])
def test_vocab_range_fails_before_device_assert(components, key, value):
    model, tokenizer = components
    row = {"input_ids": [3, 1], "attention_mask": [1, 1], "labels": [3, 1]}
    row[key] = value
    with pytest.raises(ValueError, match="vocabulary"):
        ValidatedSeq2SeqCollator(tokenizer, model)([row])


def test_log_failure_names_metric_step_and_keeps_values(tmp_path):
    state = SimpleNamespace(global_step=2, epoch=0.25)
    with pytest.raises(NumericalFailure, match="grad_norm.*inf"):
        check_logs(
            tmp_path,
            state,
            {"grad_norm": np.float32(np.inf), "loss": torch.tensor(2.0), "learning_rate": 0.0},
        )
    report = read_json(tmp_path / "numerical_failure.json")
    assert report["completed_global_step"] == 2
    assert report["nonfinite"] == {"grad_norm": "inf"}
    assert report["values"] == {"grad_norm": "inf", "loss": 2.0, "learning_rate": 0.0}


def test_nan_gradient_stops_before_adam_or_checkpoint(tmp_path, components):
    trainer = make_trainer(tmp_path, components)
    parameter = trainer.model.shared.weight
    before = parameter.detach().clone()
    hook = parameter.register_hook(lambda grad: torch.full_like(grad, float("nan")))
    try:
        with pytest.raises(NumericalFailure, match="phase=gradient"):
            trainer.train()
    finally:
        hook.remove()
    assert trainer.state.global_step == 0
    assert torch.equal(before, parameter.detach())
    assert not trainer.optimizer.state
    assert not list(tmp_path.glob("checkpoint-*"))
    report = read_json(tmp_path / "numerical_failure.json")
    assert report["attempted_optimizer_step"] == 1
    assert report["values"]["grad_norm"] == "nan"
    assert "shared.weight" in report["details"]["parameters_with_nonfinite_gradients"]
    assert report["batch"]["supervised_tokens_per_row"] == [3]


def test_bad_forward_loss_stops_before_backward(tmp_path, components, monkeypatch):
    trainer = make_trainer(tmp_path, components)
    original = trainer.model.forward

    @wraps(original)
    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        result.loss = result.loss * float("nan")
        return result

    monkeypatch.setattr(trainer.model, "forward", corrupt)
    # Keep Trainer's original signature-based input pruning unchanged.
    trainer._signature_columns = ["input_ids", "attention_mask", "labels", "decoder_input_ids"]
    with pytest.raises(NumericalFailure, match="phase=train_loss"):
        trainer.train()
    assert all(p.grad is None for p in trainer.model.parameters())
    assert read_json(tmp_path / "numerical_failure.json")["values"]["loss"] == "nan"


def test_fp32_two_step_smoke_matches_architecture_and_saves(tmp_path, components):
    trainer = make_trainer(tmp_path, components)
    trainer.train()
    assert trainer.state.global_step == 2
    assert all(torch.isfinite(p).all() for p in trainer.model.parameters())
    assert (tmp_path / "checkpoint-2/optimizer.pt").exists()
    assert trainer.model.config.feed_forward_proj == "gated-gelu"
    assert trainer.model.config.tie_word_embeddings is False


def test_legacy_run_lock_is_refused_without_modification(tmp_path):
    config = load_config(ROOT / "configs/dev.json")
    legacy = config.to_dict()
    legacy.pop("training_schema_version")
    path = tmp_path / "run_config.json"
    atomic_json(path, legacy)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="new run directory"):
        resolve_revisions(config, tmp_path)
    assert path.read_bytes() == before


def test_fp32_retry_clears_inherited_fp16_environment(tmp_path, components, monkeypatch):
    # Real pinned TrainingArguments/Accelerator objects, not mocked AMP state.
    monkeypatch.setenv("ACCELERATE_MIXED_PRECISION", "fp16")
    trainer = make_trainer(tmp_path, components)
    assert trainer.args.fp16 is False and trainer.args.bf16 is False
    assert trainer.accelerator.mixed_precision == "no"
    assert trainer.accelerator.native_amp is False
    assert trainer.accelerator.scaler is None
    precision = read_json(tmp_path / "precision.json")
    assert precision["accelerator"] == precision["effective"] == "no"
    assert precision["model_parameter_dtypes"] == ["torch.float32"]
    trainer.train()
    assert trainer.state.global_step == 2


def test_upstream_false_flags_do_not_clear_stale_amp_setting(tmp_path, monkeypatch):
    from accelerate import Accelerator
    from transformers import Seq2SeqTrainingArguments

    monkeypatch.setenv("ACCELERATE_MIXED_PRECISION", "fp16")
    args = Seq2SeqTrainingArguments(
        output_dir=str(tmp_path), fp16=False, bf16=False, use_cpu=True, report_to="none"
    )
    assert not args.fp16 and not args.bf16
    assert Accelerator(cpu=True).mixed_precision == "fp16"
    # This is a CPU configuration reproduction, not a GPU/FP16 training claim.


def test_zero_learning_rate_does_not_protect_against_nan_gradients():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    parameter.grad = torch.tensor([float("nan")])
    optimizer = torch.optim.AdamW([parameter], lr=0.0)
    assert torch.isnan(torch.nn.utils.clip_grad_norm_([parameter], 1.0))
    optimizer.step()
    assert torch.isnan(parameter).all()


def test_nonfinite_evaluation_loss_is_reported_before_metrics(tmp_path, components, monkeypatch):
    trainer = make_trainer(tmp_path, components)
    original = trainer.model.forward

    @wraps(original)
    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        if result.loss is not None:
            result.loss = result.loss * float("inf")
        return result

    monkeypatch.setattr(trainer.model, "forward", corrupt)
    trainer._signature_columns = ["input_ids", "attention_mask", "labels", "decoder_input_ids"]
    with pytest.raises(NumericalFailure, match="phase=eval_loss"):
        trainer.evaluate()
    assert read_json(tmp_path / "numerical_failure.json")["values"]["eval_loss"] == "inf"
