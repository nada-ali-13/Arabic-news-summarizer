"""Bounded supervised experiments with Drive-first full-state recovery."""

import gc
import json
import os
import shutil
import time
import uuid
from dataclasses import replace
from pathlib import Path

from .checkpoints import directory_bytes, latest_checkpoint, seal, valid
from .data import fixed_subset, load_official, tokenize
from .io import atomic_json, read_json
from .metrics import trainer_metrics
from .numerics import ValidatedSeq2SeqCollator, check_logs, checked_trainer_class


def resolve_revisions(config, root):
    """Freeze upstream commits once, and refuse incompatible in-place restarts."""
    from huggingface_hub import HfApi

    lock = root / "run_config.json"
    if lock.exists():
        previous = read_json(lock)
        for name in ("model_revision", "dataset_revision"):
            if getattr(config, name) not in {"main", previous[name]}:
                raise ValueError("Upstream revision changed: use a new run directory")
        config = replace(
            config,
            model_revision=previous["model_revision"],
            dataset_revision=previous["dataset_revision"],
        )
        # Compare serialized experiment dictionaries as well.
        current = config.to_dict()
        for key in ("max_hours", "session_hours", "max_storage_gb", "reserve_gb"):
            current.pop(key)
            previous.pop(key)
        if current != previous:
            raise ValueError("Configuration changed: use a new run directory")
    else:
        api = HfApi()
        config = replace(
            config,
            model_revision=api.model_info(config.model_id, revision=config.model_revision).sha,
            dataset_revision=api.dataset_info(
                config.dataset_id, revision=config.dataset_revision
            ).sha,
        )
    atomic_json(lock, config.to_dict())
    return config


def load_model(config, path=None):
    from transformers import AutoModelForSeq2SeqLM

    model = AutoModelForSeq2SeqLM.from_pretrained(
        path or config.model_id,
        **({} if path else {"revision": config.model_revision}),
        use_safetensors=bool(path),
        weights_only=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.generation_config.max_length = config.target_length
    model.generation_config.num_beams = 4
    model.generation_config.do_sample = False
    model.generation_config.no_repeat_ngram_size = 3
    return model


def build_trainer(
    config, experiment, output, model, tokenizer, train, validation, callbacks=(), references=None
):
    import torch
    from transformers import Seq2SeqTrainingArguments

    gpu = torch.cuda.is_available()
    if gpu and config.mixed_precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 requested but unsupported by this GPU; T4 requires no or fp16")
    effective_precision = config.mixed_precision if gpu else "no"
    # Transformers 4.57.1 preserves this variable when both precision flags are
    # false. A prior FP16 attempt otherwise silently makes an FP32 retry use AMP.
    os.environ["ACCELERATE_MIXED_PRECISION"] = effective_precision
    args = Seq2SeqTrainingArguments(
        output_dir=str(output),
        max_steps=experiment.max_steps,
        per_device_train_batch_size=experiment.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=experiment.accumulation,
        learning_rate=experiment.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.06,
        lr_scheduler_type="linear",
        max_grad_norm=1.0,
        optim="adamw_torch",
        fp16=effective_precision == "fp16",
        bf16=effective_precision == "bf16",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps",
        eval_steps=experiment.eval_steps,
        save_strategy="steps",
        save_steps=experiment.save_steps,
        save_total_limit=2,
        save_only_model=False,
        load_best_model_at_end=True,
        metric_for_best_model="rougeLsum",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=config.target_length,
        generation_num_beams=4,
        eval_accumulation_steps=1,
        logging_steps=1 if experiment.smoke else 10,
        logging_nan_inf_filter=False,
        seed=config.seed,
        data_seed=config.seed,
        full_determinism=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=gpu,
        report_to="none",
        save_safetensors=True,
    )
    trainer = checked_trainer_class()(
        model=model,
        args=args,
        processing_class=tokenizer,
        train_dataset=train,
        eval_dataset=validation,
        data_collator=ValidatedSeq2SeqCollator(
            tokenizer, model=model, pad_to_multiple_of=8 if gpu else None
        ),
        compute_metrics=trainer_metrics(tokenizer, references),
        callbacks=list(callbacks),
    )
    actual_precision = trainer.accelerator.mixed_precision
    if actual_precision != effective_precision:
        raise RuntimeError(
            f"Precision mismatch: requested={effective_precision}, actual={actual_precision}. "
            "Restart the Colab runtime; a new Drive directory alone does not reset runtime state."
        )
    precision = {
        "requested": config.mixed_precision,
        "effective": effective_precision,
        "accelerator": actual_precision,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "native_amp": trainer.accelerator.native_amp,
        "grad_scaler_enabled": trainer.accelerator.scaler is not None,
        "model_parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
        "model_config": {
            name: getattr(model.config, name, None)
            for name in (
                "model_type",
                "vocab_size",
                "feed_forward_proj",
                "tie_word_embeddings",
                "pad_token_id",
                "eos_token_id",
                "decoder_start_token_id",
                "use_cache",
            )
        },
        "tokenizer_pad_token_id": tokenizer.pad_token_id,
        "tokenizer_eos_token_id": tokenizer.eos_token_id,
    }
    atomic_json(Path(output) / "precision.json", precision)
    print(
        f"Precision verified: requested={config.mixed_precision}, "
        f"effective={effective_precision}, accelerator={actual_precision}, "
        f"native_amp={precision['native_amp']}, scaler={precision['grad_scaler_enabled']}"
    )
    return trainer


def budget_callback(config, root, fingerprint, ledger, session_start):
    from transformers import TrainerCallback

    class PersistentBudget(TrainerCallback):
        def __init__(self):
            self.last = time.monotonic()
            self.stopped = None

        def tick(self):
            now = time.monotonic()
            ledger["seconds"] += now - self.last
            self.last = now
            atomic_json(root / "budget.json", ledger)

        def on_step_end(self, args, state, control, **kwargs):
            self.tick()
            if ledger["seconds"] >= config.max_hours * 3600:
                self.stopped = "total time budget"
            elif time.monotonic() - session_start >= config.session_hours * 3600:
                self.stopped = "session time budget"
            elif state.global_step % 10 == 0:
                used = directory_bytes(root)
                free = shutil.disk_usage(root).free
                if used + config.reserve_gb * 1e9 > config.max_storage_gb * 1e9:
                    self.stopped = "storage budget (reserve retained for checkpoint)"
                elif free < config.reserve_gb * 1e9:
                    self.stopped = "Drive free-space reserve"
            if self.stopped:
                control.should_save = True
                control.should_training_stop = True
            return control

        def on_save(self, args, state, control, **kwargs):
            directory = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            atomic_json(directory / "run_config.json", config.to_dict())
            seal(directory, fingerprint)
            self.tick()

        def on_log(self, args, state, control, logs=None, **kwargs):
            check_logs(args.output_dir, state, logs)

    return PersistentBudget()


def predict_report(trainer, tokenized, rows, path):
    started = time.monotonic()
    output = trainer.predict(tokenized)
    predictions = trainer.processing_class.batch_decode(
        output.predictions, skip_special_tokens=True
    )
    report = {
        "metrics": output.metrics,
        "seconds": time.monotonic() - started,
        "ids": list(rows["id"]),
        "examples": [
            {
                "id": row["id"],
                "source": row["text"],
                "reference": row["summary"],
                "prediction": prediction,
            }
            for row, prediction in zip(rows, predictions)
        ],
    }
    atomic_json(path, report)
    return report


def run(config, root, allow_cpu=False, smoke_only=False):
    """Run again with the same config/root to recover, without editing experiments."""
    import torch
    from transformers import AutoTokenizer, EarlyStoppingCallback, set_seed

    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError("Select a Colab T4 GPU before training (Runtime > Change runtime type).")
    if shutil.disk_usage(root).free < config.reserve_gb * 1e9:
        raise RuntimeError("Insufficient Drive space for a full optimizer checkpoint")
    config = resolve_revisions(config, root)
    set_seed(config.seed)
    fingerprint = config.fingerprint
    session_start = time.monotonic()
    ledger = read_json(root / "budget.json") if (root / "budget.json").exists() else {"seconds": 0}
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id, revision=config.model_revision, use_fast=False
    )
    prepared = root / "prepared"
    if (prepared / "complete.json").exists():
        from datasets import load_from_disk

        if read_json(prepared / "complete.json")["fingerprint"] != fingerprint:
            raise ValueError("Prepared dataset configuration mismatch")
        ds = load_from_disk(str(prepared))
    else:
        ds = load_official(config, root / "cache")
        ds.save_to_disk(str(prepared))
        atomic_json(prepared / "complete.json", {"fingerprint": fingerprint})
    val_rows = fixed_subset(ds["validation"], config.validation_samples, config.seed)
    validation = tokenize(val_rows, tokenizer, config)
    atomic_json(root / "validation_ids.json", list(val_rows["id"]))
    state_path = root / "experiments.json"
    states = read_json(state_path) if state_path.exists() else {}

    def evaluate_model(model_path, rows, destination):
        started = time.monotonic()
        trainer = build_trainer(
            config,
            config.experiments[0],
            root / "evaluation_tmp",
            load_model(config, model_path),
            tokenizer,
            None,
            tokenize(rows, tokenizer, config),
            references=list(rows["summary"]),
        )
        try:
            result = predict_report(trainer, trainer.eval_dataset, rows, destination)
        finally:
            del trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        ledger["seconds"] += time.monotonic() - started
        atomic_json(root / "budget.json", ledger)
        return result

    for experiment in config.experiments:
        if smoke_only and not experiment.smoke:
            return {
                "status": "smoke_completed",
                "run_dir": str(root),
                "note": "No baseline, candidate training, or test evaluation was run.",
            }
        previous = states.get(experiment.name, {})
        if previous.get("status") == "completed":
            if not experiment.smoke and not valid(previous["export"], fingerprint):
                raise RuntimeError("Completed export is corrupt; refusing to silently replace it")
            continue
        if ledger["seconds"] >= config.max_hours * 3600:
            return {"status": "paused", "reason": "total time budget; raise max_hours to continue"}
        if time.monotonic() - session_start >= config.session_hours * 3600:
            return {"status": "paused", "reason": "session time budget; rerun to continue"}
        if directory_bytes(root) + config.reserve_gb * 1e9 > config.max_storage_gb * 1e9:
            return {
                "status": "paused",
                "reason": "storage budget; archive old runs or raise budget",
            }
        if not experiment.smoke and not (root / "baseline_validation.json").exists():
            evaluate_model(None, val_rows, root / "baseline_validation.json")
            if ledger["seconds"] >= config.max_hours * 3600:
                return {"status": "paused", "reason": "time budget reached during baseline"}
        output = root / "experiments" / experiment.name
        checkpoint = latest_checkpoint(output, fingerprint)
        train = tokenize(
            fixed_subset(ds["train"], experiment.train_samples, config.seed), tokenizer, config
        )
        monitor = budget_callback(config, root, fingerprint, ledger, session_start)
        experiment_started = ledger["seconds"]
        states[experiment.name] = {
            **previous,
            "status": "running",
            "config": config.to_dict(),
            "resume_from": checkpoint,
        }
        atomic_json(state_path, states)
        set_seed(config.seed)
        model = load_model(config)
        trainer = build_trainer(
            config,
            experiment,
            output,
            model,
            tokenizer,
            train,
            validation.select(range(min(2, len(validation)))) if experiment.smoke else validation,
            [monitor]
            + ([] if experiment.smoke else [EarlyStoppingCallback(early_stopping_patience=3)]),
            references=list(val_rows["summary"])[:2]
            if experiment.smoke
            else list(val_rows["summary"]),
        )
        try:
            trainer.train(resume_from_checkpoint=checkpoint)
            monitor.tick()
            if monitor.stopped:
                states[experiment.name].update(
                    status="paused",
                    reason=monitor.stopped,
                    runtime_seconds=previous.get("runtime_seconds", 0)
                    + ledger["seconds"]
                    - experiment_started,
                )
                atomic_json(state_path, states)
                return {"status": "paused", "reason": monitor.stopped}
            # Trainer has loaded the best validation checkpoint, not necessarily the last.
            metrics = trainer.evaluate()
            if "eval_rougeLsum" not in metrics or not all(
                __import__("math").isfinite(v)
                for v in metrics.values()
                if isinstance(v, (int, float))
            ):
                raise RuntimeError("Missing or nonfinite validation metrics")
            record = {
                **states[experiment.name],
                "status": "completed",
                "metrics": metrics,
                "global_step": trainer.state.global_step,
                "best_checkpoint": trainer.state.best_model_checkpoint,
                "runtime_seconds": previous.get("runtime_seconds", 0)
                + ledger["seconds"]
                - experiment_started,
                "training_loss": trainer.state.log_history,
            }
            if not experiment.smoke:
                export = root / "exports" / (experiment.name + "-" + uuid.uuid4().hex[:8])
                trainer.save_model(str(export))
                tokenizer.save_pretrained(str(export))
                atomic_json(export / "run_config.json", config.to_dict())
                seal(export, fingerprint, checkpoint=False)
                record["export"] = str(export)
            monitor.tick()
            record["runtime_seconds"] = (
                previous.get("runtime_seconds", 0) + ledger["seconds"] - experiment_started
            )
            states[experiment.name] = record
            atomic_json(state_path, states)
            update_best(root, states, fingerprint)
            # Only successful smoke weights are disposable; retain the audit record.
            if experiment.smoke:
                for path in output.glob("checkpoint-*"):
                    if path.is_dir() and path.name.split("-")[-1].isdigit():
                        shutil.rmtree(path)
        except Exception as exc:
            monitor.tick()
            states[experiment.name].update(
                status="failed",
                error_type=type(exc).__name__,
                error=str(exc),
                global_step=trainer.state.global_step,
                numerical_diagnostic=str(output / "numerical_failure.json")
                if (output / "numerical_failure.json").exists()
                else None,
            )
            atomic_json(state_path, states)
            raise
        finally:
            # Remove both model references before loading the next experiment.
            del trainer, model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    best = update_best(root, states, fingerprint)
    # Test evaluation begins only when the complete bounded plan is finished.
    report_path = root / "final_report.json"
    if report_path.exists():
        return read_json(report_path)
    rows = fixed_subset(ds["test"], config.test_samples, config.seed)
    atomic_json(root / "test_ids.json", list(rows["id"]))
    baseline_path = root / "baseline_test.json"
    if not baseline_path.exists():
        if (
            ledger["seconds"] >= config.max_hours * 3600
            or time.monotonic() - session_start >= config.session_hours * 3600
        ):
            return {"status": "paused", "reason": "time budget before baseline test evaluation"}
        evaluate_model(None, rows, baseline_path)
    tuned_path = root / "best_test.json"
    if not tuned_path.exists():
        if (
            ledger["seconds"] >= config.max_hours * 3600
            or time.monotonic() - session_start >= config.session_hours * 3600
        ):
            return {
                "status": "paused",
                "reason": "time budget before selected-model test evaluation",
            }
        evaluate_model(best["path"], rows, tuned_path)
    report = {
        "status": "completed",
        "config": config.to_dict(),
        "best": best,
        "baseline_validation": read_json(root / "baseline_validation.json")["metrics"],
        "baseline_test": read_json(baseline_path)["metrics"],
        "finetuned_test": read_json(tuned_path)["metrics"],
        "compute_seconds": ledger["seconds"],
        "qualitative_examples": str(tuned_path),
        "note": "Test subset fixed by seed; never used for selection. Manual review pending.",
    }
    atomic_json(report_path, report)
    return report


def update_best(root, states, fingerprint):
    candidates = [s for s in states.values() if s.get("status") == "completed" and s.get("export")]
    if not candidates:
        return None
    winner = max(candidates, key=lambda s: s["metrics"]["eval_rougeLsum"])
    pointer = {
        "path": winner["export"],
        "validation_rougeLsum": winner["metrics"]["eval_rougeLsum"],
        "fingerprint": fingerprint,
    }
    path = Path(root) / "best.json"
    if path.exists():
        old = read_json(path)
        if old["validation_rougeLsum"] >= pointer["validation_rougeLsum"]:
            return old
    atomic_json(path, pointer)
    return pointer


def main():
    import argparse

    from .config import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dev.json")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run(load_config(args.config), args.run_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
