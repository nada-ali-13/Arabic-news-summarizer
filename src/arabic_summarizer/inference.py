"""Independent CPU/CUDA inference and bounded hierarchical long-document handling."""

import os
import threading
import time
from pathlib import Path

from .text import normalize, word_count


class Summarizer:
    def __init__(
        self, model_path=None, device=None, source_length=1024, model=None, tokenizer=None
    ):
        self.model_path = model_path or os.getenv("ARABIC_SUMMARIZER_MODEL", "artifacts/best")
        self.device = device
        self.source_length = source_length
        self.model = model
        self.tokenizer = tokenizer
        self.lock = threading.Lock()
        if not 32 <= source_length <= 1024:
            raise ValueError("source_length must be between 32 and 1024")

    def load(self):
        if self.model is not None and self.tokenizer is not None:
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        if self.model_path == "artifacts/best" and not Path(self.model_path).is_dir():
            raise ValueError("لم يُعثر على النموذج المدرّب. حدد مساره في ARABIC_SUMMARIZER_MODEL.")
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_path,
            dtype=dtype,
            use_safetensors=self.model_path != "UBC-NLP/AraT5v2-base-1024",
            weights_only=True,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()
        self.model.config.use_cache = True

    def _generate(self, ids, max_tokens, min_tokens):
        import torch

        inputs = self.tokenizer.prepare_for_model(
            ids,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            prepend_batch_axis=True,
            return_tensors="pt",
        )
        if inputs["input_ids"].shape[-1] > self.source_length:
            raise ValueError("Internal source token budget exceeded")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                min_new_tokens=min(min_tokens, max(1, len(ids) // 6)),
                num_beams=4,
                do_sample=False,
                no_repeat_ngram_size=3,
                repetition_penalty=1.1,
                length_penalty=1.0,
                early_stopping=True,
            )
        return normalize(self.tokenizer.decode(output[0], skip_special_tokens=True))

    def summarize(self, text, preference="متوسط"):
        text = normalize(text)
        if not text:
            raise ValueError("أدخل نصًا عربيًا أو رابط مقال أولًا.")
        if len(text) > 100_000:
            raise ValueError("النص طويل جدًا؛ استخدم مقالًا أقصر من 100 ألف حرف.")
        lengths = {"قصير": (64, 8), "متوسط": (96, 16), "طويل": (128, 24)}
        if preference not in lengths:
            raise ValueError("اختر طولًا صالحًا للملخص.")
        with self.lock:
            self.load()
            started = time.monotonic()
            max_tokens, min_tokens = lengths[preference]
            budget = self.source_length - self.tokenizer.num_special_tokens_to_add(pair=False)
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            chunks = [ids[i : i + budget] for i in range(0, len(ids), budget)]
            if len(chunks) > 16:
                raise ValueError("المقال يتجاوز 16 مقطعًا؛ قسّمه إلى مقالات أصغر.")
            original_chunks = len(chunks)
            for _ in range(4):
                if len(chunks) > 16:
                    raise ValueError("تعذر ضغط المقاطع ضمن الحد؛ جرّب مقالًا أقصر.")
                summaries = [self._generate(chunk, max_tokens, min_tokens) for chunk in chunks]
                summary = normalize(" ".join(summaries))
                merged = self.tokenizer.encode(summary, add_special_tokens=False)
                if len(chunks) == 1:
                    break
                chunks = [merged[i : i + budget] for i in range(0, len(merged), budget)]
                if not chunks:
                    break
            else:
                raise ValueError("تعذر دمج المقاطع ضمن الحد؛ جرّب مقالًا أقصر.")
            if not summary:
                raise ValueError("لم ينتج النموذج ملخصًا؛ جرّب نصًا أطول أو نموذجًا مدرّبًا.")
            original_words, summary_words = word_count(text), word_count(summary)
            return {
                "summary": summary,
                "original_words": original_words,
                "summary_words": summary_words,
                "compression_percent": round(100 * (1 - summary_words / original_words), 1),
                "seconds": round(time.monotonic() - started, 3),
                "chunks": original_chunks,
                "warning": "لُخص المقال على مقاطع؛ قد تُفقد روابط بين أجزائه."
                if original_chunks > 1
                else "",
            }
