"""Unicode-aware Arabic ROUGE. The default rouge-score tokenizer drops Arabic."""

import re

from .text import normalize, sentences


class ArabicTokenizer:
    def tokenize(self, text):
        return re.findall(r"[^\W_]+", normalize(text).lower(), flags=re.UNICODE)


def rouge(predictions, references):
    from rouge_score.rouge_scorer import RougeScorer

    if not predictions or len(predictions) != len(references):
        raise ValueError("Nonempty aligned predictions/references required")
    scorer = RougeScorer(
        ["rouge1", "rouge2", "rougeL", "rougeLsum"],
        tokenizer=ArabicTokenizer(),
        use_stemmer=False,
        split_summaries=False,
    )
    values = {k: [] for k in ("rouge1", "rouge2", "rougeL", "rougeLsum")}
    for prediction, reference in zip(predictions, references):
        scores = scorer.score(
            "\n".join(sentences(normalize(reference))), "\n".join(sentences(normalize(prediction)))
        )
        for key in values:
            values[key].append(scores[key].fmeasure * 100)
    return {key: sum(v) / len(v) for key, v in values.items()}


def trainer_metrics(tokenizer, references=None):
    def compute(eval_prediction):
        import numpy as np

        predictions, labels = eval_prediction
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        return rouge(
            tokenizer.batch_decode(predictions, skip_special_tokens=True),
            references
            if references is not None
            else tokenizer.batch_decode(labels, skip_special_tokens=True),
        )

    return compute
