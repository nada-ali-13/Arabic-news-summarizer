"""Measure actual model latency after loading, never report guessed timings."""

import argparse
import json
import resource
import statistics
import time
from pathlib import Path

from arabic_summarizer.inference import Summarizer

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--text-file", required=True)
parser.add_argument("--device", default="cpu")
parser.add_argument("--runs", type=int, default=3)
args = parser.parse_args()
if args.runs < 1:
    parser.error("--runs must be positive")
summarizer = Summarizer(args.model, device=args.device)
started = time.monotonic()
summarizer.load()
load_seconds = time.monotonic() - started
text = Path(args.text_file).read_text(encoding="utf-8")
samples = [summarizer.summarize(text)["seconds"] for _ in range(args.runs)]
print(
    json.dumps(
        {
            "device": args.device,
            "load_seconds": load_seconds,
            "generation_seconds": samples,
            "median_seconds": statistics.median(samples),
            "peak_rss_mb_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        },
        indent=2,
    )
)
