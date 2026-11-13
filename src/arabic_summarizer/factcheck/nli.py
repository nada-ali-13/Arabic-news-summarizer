"""Opt-in, CPU-only local model: never downloads during an interactive request."""
import os
import threading

MODEL_ID = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"


class NliUnavailable(RuntimeError):
    pass


def label_mapping(config):
    mapping = {int(i): str(label).lower() for i, label in config.id2label.items()}
    if set(mapping.values()) != {"entailment", "neutral", "contradiction"}:
        raise NliUnavailable("تعريف تسميات NLI غير موثوق؛ لم يُصدر حكم.")
    return mapping


class Nli:
    def __init__(self):
        self.model = self.tokenizer = None
        self.lock = threading.Lock()
        self.failed = False

    def assess(self, premise, hypothesis):
        if os.environ.get("KHULASA_NLI_ENABLED") != "1" or self.failed:
            raise NliUnavailable("NLI غير مفعّل أو غير متاح؛ راجع دليل الإعداد.")
        with self.lock:
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                if self.model is None:
                    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
                    model = AutoModelForSequenceClassification.from_pretrained(
                        MODEL_ID, local_files_only=True, use_safetensors=True,
                    ).to("cpu").eval()
                    self.labels = label_mapping(model.config)
                    self.tokenizer, self.model = tokenizer, model
                inputs = self.tokenizer(premise, hypothesis, return_tensors="pt",
                                        truncation="only_first", max_length=384)
                with torch.inference_mode():
                    scores = self.model(**inputs).logits.softmax(-1)[0]
                index = int(scores.argmax())
                return self.labels[index] if float(scores[index]) >= .9 else "neutral"
            except Exception:
                self.model = self.tokenizer = None
                self.failed = True
                raise NliUnavailable("تعذر تشغيل NLI المحلي؛ لا توجد نتيجة نموذج بديلة.") from None
