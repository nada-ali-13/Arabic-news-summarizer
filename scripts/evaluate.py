"""Recompute Arabic ROUGE from saved, aligned evaluation examples."""

import argparse
import json
from pathlib import Path

from arabic_summarizer.metrics import rouge

parser = argparse.ArgumentParser()
parser.add_argument("report", help="baseline_test.json or best_test.json produced by training")
args = parser.parse_args()
examples = json.loads(Path(args.report).read_text(encoding="utf-8"))["examples"]
print(
    json.dumps(
        rouge([e["prediction"] for e in examples], [e["reference"] for e in examples]),
        ensure_ascii=False,
        indent=2,
    )
)
