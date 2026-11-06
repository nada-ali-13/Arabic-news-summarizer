import pytest

from arabic_summarizer.inference import Summarizer


class Tokenizer:
    def num_special_tokens_to_add(self, pair=False):
        return 1

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))


def fake_summarizer(monkeypatch):
    instance = Summarizer(model=object(), tokenizer=Tokenizer(), source_length=32)
    calls = []

    def generate(ids, maximum, minimum):
        calls.append(len(ids))
        return "ملخص خبر عربي"

    monkeypatch.setattr(instance, "_generate", generate)
    return instance, calls


def test_mock_generation_and_counts(monkeypatch):
    instance, calls = fake_summarizer(monkeypatch)
    result = instance.summarize("خبر " * 12)
    assert result["summary_words"] == 3 and result["original_words"] == 12
    assert result["compression_percent"] == 75
    assert calls == [12]


def test_long_document_map_and_reduce(monkeypatch):
    instance, calls = fake_summarizer(monkeypatch)
    result = instance.summarize("خبر " * 80)
    assert result["chunks"] == 3 and result["warning"]
    assert calls == [31, 31, 18, 9]
    with pytest.raises(ValueError, match="16"):
        instance.summarize("خبر " * 600)


@pytest.mark.parametrize(
    "text,preference", [("", "متوسط"), (None, "متوسط"), ("خبر", "invalid"), ("x" * 100001, "قصير")]
)
def test_invalid_input(text, preference):
    with pytest.raises(ValueError):
        Summarizer().summarize(text, preference)


def test_missing_default_model_fails_clearly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ARABIC_SUMMARIZER_MODEL", raising=False)
    with pytest.raises(ValueError, match="ARABIC_SUMMARIZER_MODEL"):
        Summarizer().load()


def test_real_gradio_blocks_initialization(monkeypatch):
    import gradio as gr

    from app import build_app

    instance, _ = fake_summarizer(monkeypatch)
    demo = build_app(instance)
    assert isinstance(demo, gr.Blocks)
    assert "تلخيص المقال" in str(demo.config)
    assert "direction: rtl" in demo.css
    # Execute the actual event handler without starting a server or model download.
    handler = next(fn.fn for fn in demo.fns.values() if fn.fn.__name__ == "submit")
    summary, counts, _ = handler("خبر " * 10, "", "متوسط")
    assert summary == "ملخص خبر عربي" and "10" in counts
    with pytest.raises(gr.Error):
        handler("نص", "https://example.com", "متوسط")
    demo.close()
