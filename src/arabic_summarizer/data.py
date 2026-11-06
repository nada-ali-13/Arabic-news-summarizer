"""Load official XL-Sum v2 JSONL files without executing a remote dataset script."""

import hashlib
import json
import statistics
import tarfile
from pathlib import Path

from .io import atomic_json
from .text import normalize


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_splits(raw):
    """Held-out data has priority: test > validation > train, no repartitioning.

    Remove exact normalized-source, ID and URL duplicates from lower-priority
    splits; preserve all usable test rows, including any internal test duplicates.
    """
    if set(raw) != {"train", "validation", "test"}:
        raise ValueError("Official train/validation/test splits required")
    result, audit = {}, {}
    seen_ids, seen_texts, seen_urls = set(), set(), set()
    for split in ("test", "validation", "train"):
        kept = []
        removed = 0
        for row in raw[split]:
            if not {"id", "url", "text", "summary", "title"} <= row.keys():
                raise ValueError(f"Invalid XL-Sum schema in {split}")
            if not all(isinstance(row[k], str) for k in row):
                raise ValueError("XL-Sum fields must be strings")
            row = {**row, "text": normalize(row["text"]), "summary": normalize(row["summary"])}
            if not row["id"] or not row["text"] or not row["summary"]:
                removed += 1
                continue
            key = digest(row["text"])
            duplicate = (
                row["id"] in seen_ids
                or key in seen_texts
                or bool(row["url"] and row["url"] in seen_urls)
            )
            if split != "test" and duplicate:
                removed += 1
                continue
            kept.append(row)
            seen_ids.add(row["id"])
            seen_texts.add(key)
            if row["url"]:
                seen_urls.add(row["url"])
        if not kept:
            raise ValueError(f"Empty usable {split} split")
        result[split] = kept
        audit[split] = {"raw": len(raw[split]), "kept": len(kept), "removed": removed}
    return result, audit


def load_official(config, cache_dir):
    from datasets import Dataset, DatasetDict
    from huggingface_hub import hf_hub_download

    archive = hf_hub_download(
        config.dataset_id,
        "data/arabic_XLSum_v2.0.tar.bz2",
        repo_type="dataset",
        revision=config.dataset_revision,
        cache_dir=str(cache_dir),
    )
    raw = {}
    names = {
        "train": "arabic_train.jsonl",
        "validation": "arabic_val.jsonl",
        "test": "arabic_test.jsonl",
    }
    with tarfile.open(archive, "r:bz2") as tar:
        for split, name in names.items():
            members = [m for m in tar.getmembers() if Path(m.name).name == name]
            if len(members) != 1 or not members[0].isfile() or members[0].size > 600_000_000:
                raise ValueError(f"Invalid archive member: {name}")
            # Read only the expected file, never extract archive paths to disk.
            with tar.extractfile(members[0]) as f:
                raw[split] = [json.loads(line) for line in f if line.strip()]
    clean, audit = clean_splits(raw)
    ds = DatasetDict({k: Dataset.from_list(v) for k, v in clean.items()})
    stats = {}
    for split, rows in clean.items():
        lengths = [len(r["text"].split()) for r in rows]
        stats[split] = {
            **audit[split],
            "mean_source_words": statistics.mean(lengths),
            "max_source_words": max(lengths),
            "example": rows[0],
        }
    atomic_json(Path(cache_dir).parent / "dataset_statistics.json", stats)
    return ds


def fixed_subset(dataset, count, seed):
    return dataset.shuffle(seed=seed).select(range(min(count, len(dataset))))


def tokenize(dataset, tokenizer, config):
    def batch(rows):
        inputs = tokenizer(rows["text"], max_length=config.source_length, truncation=True)
        inputs["labels"] = tokenizer(
            text_target=rows["summary"], max_length=config.target_length, truncation=True
        )["input_ids"]
        return inputs

    return dataset.map(
        batch,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing Arabic",
        load_from_cache_file=True,
    )
