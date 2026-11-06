"""Real CPU optimization/recovery on an offline, tiny random T5. No quality claims."""

from dataclasses import replace


def test_real_trainer_interruption_resume_export_and_final_evaluation(tmp_path, monkeypatch):
    import torch
    from datasets import Dataset, DatasetDict
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from transformers import (
        AutoTokenizer,
        PreTrainedTokenizerFast,
        T5Config,
        T5ForConditionalGeneration,
    )

    from arabic_summarizer import training
    from arabic_summarizer.checkpoints import valid
    from arabic_summarizer.config import Config, Experiment
    from arabic_summarizer.inference import Summarizer
    from arabic_summarizer.io import read_json

    torch.set_num_threads(1)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    vocab = {
        w: i
        for i, w in enumerate(
            [
                "<pad>",
                "</s>",
                "<unk>",
                "هذا",
                "خبر",
                "عربي",
                "عن",
                "افتتاح",
                "مكتبة",
                "جديدة",
                "للطلاب",
            ]
        )
    }
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(single="$A </s>", special_tokens=[("</s>", 1)])
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", eos_token="</s>", unk_token="<unk>"
    )
    tiny_config = T5Config(
        vocab_size=len(vocab),
        d_model=16,
        d_ff=32,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        d_kv=8,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
    )
    calls = []
    real_load_model = training.load_model
    real_tokenizer_loader = AutoTokenizer.from_pretrained

    def tiny_load(config, path=None):
        if path:
            return real_load_model(config, path)
        model = T5ForConditionalGeneration(tiny_config)
        model.config.use_cache = False
        model.generation_config.max_length = config.target_length
        model.generation_config.num_beams = 4
        return model

    monkeypatch.setattr(training, "load_model", tiny_load)
    monkeypatch.setattr(training, "resolve_revisions", lambda config, root: config)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)

    def dataset(config, cache):
        calls.append("dataset")

        def split(prefix):
            return Dataset.from_list(
                [
                    {
                        "id": f"{prefix}-{i}",
                        "text": "هذا خبر عربي عن افتتاح مكتبة جديدة للطلاب",
                        "summary": "افتتاح مكتبة جديدة",
                        "title": "مكتبة",
                        "url": "",
                    }
                    for i in range(4)
                ]
            )

        return DatasetDict({name: split(name) for name in ("train", "validation", "test")})

    monkeypatch.setattr(training, "load_official", dataset)
    config = Config(
        source_length=32,
        target_length=8,
        validation_samples=2,
        test_samples=2,
        mixed_precision="no",
        max_hours=1,
        session_hours=1,
        max_storage_gb=1,
        reserve_gb=0.000001,
        experiments=[
            Experiment("smoke", 1, 2, accumulation=1, eval_steps=1, save_steps=1, smoke=True),
            Experiment("candidate", 2, 4, accumulation=1, eval_steps=1, save_steps=1),
        ],
    )
    factory = training.budget_callback
    interrupted = False

    def stop_once(*args):
        callback = factory(*args)
        original = callback.on_step_end

        def on_step_end(arguments, state, control, **kwargs):
            nonlocal interrupted
            original(arguments, state, control, **kwargs)
            if (
                arguments.output_dir.endswith("candidate")
                and state.global_step == 1
                and not interrupted
            ):
                interrupted = True
                callback.stopped = "simulated disconnect"
                control.should_save = control.should_training_stop = True
            return control

        callback.on_step_end = on_step_end
        return callback

    monkeypatch.setattr(training, "budget_callback", stop_once)
    run_dir = tmp_path / "run"
    first = training.run(config, run_dir, allow_cpu=True)
    assert first["status"] == "paused"
    assert not (run_dir / "baseline_test.json").exists()
    checkpoint = run_dir / "experiments/candidate/checkpoint-1"
    assert valid(checkpoint, config.fingerprint)
    optimizer = torch.load(checkpoint / "optimizer.pt", weights_only=True)
    assert optimizer["state"]  # Actual Adam state, not a mock checkpoint.
    baseline_before = (run_dir / "baseline_validation.json").read_bytes()
    second = training.run(config, run_dir, allow_cpu=True)
    assert second["status"] == "completed"
    assert (run_dir / "baseline_validation.json").read_bytes() == baseline_before
    states = read_json(run_dir / "experiments.json")
    assert states["candidate"]["resume_from"].endswith("checkpoint-1")
    assert states["candidate"]["global_step"] == 2
    resumed_optimizer = torch.load(
        run_dir / "experiments/candidate/checkpoint-2/optimizer.pt", weights_only=True
    )
    assert all(int(value["step"]) == 2 for value in resumed_optimizer["state"].values())
    assert states["smoke"]["status"] == "completed"
    assert valid(second["best"]["path"], config.fingerprint)
    assert calls == ["dataset"]  # Prepared data reused on restart.
    assert training.run(replace(config, max_hours=2), run_dir, allow_cpu=True) == second
    # Independently load exported weights AND tokenizer through the actual loader.
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", real_tokenizer_loader)
    instance = Summarizer(second["best"]["path"], device="cpu", source_length=32)
    result = instance.summarize("هذا خبر عربي عن افتتاح مكتبة جديدة للطلاب")
    assert result["summary"] and result["seconds"] >= 0
    from arabic_summarizer.metrics import rouge

    saved = read_json(run_dir / "best_test.json")
    recomputed = rouge(
        [row["prediction"] for row in saved["examples"]],
        [row["reference"] for row in saved["examples"]],
    )
    assert all(saved["metrics"]["test_" + key] == value for key, value in recomputed.items())
