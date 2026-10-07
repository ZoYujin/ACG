#!/usr/bin/env python3
"""Score POPE JSONL outputs using the original word-based decision rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


NEGATIVE_WORDS = {"No", "not", "no", "NO"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def prediction(answer: str) -> int:
    words = answer.replace(".", "").replace(",", "").split(" ")
    is_negative = any(word in NEGATIVE_WORDS for word in words) or any(
        word.endswith("n't") for word in words
    )
    return 0 if is_negative else 1


def safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> None:
    args = parse_args()
    records = [
        json.loads(line)
        for line in args.input.read_text().splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"No predictions found in {args.input}.")

    predictions = [prediction(record["ans"]) for record in records]
    labels = [int(record["label"]) for record in records]

    tp = sum(pred == 1 and label == 1 for pred, label in zip(predictions, labels))
    tn = sum(pred == 0 and label == 0 for pred, label in zip(predictions, labels))
    fp = sum(pred == 1 and label == 0 for pred, label in zip(predictions, labels))
    fn = sum(pred == 0 and label == 1 for pred, label in zip(predictions, labels))

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    metrics = {
        "samples": len(records),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Accuracy": safe_divide(tp + tn, len(records)),
        "Precision": precision,
        "Recall": recall,
        "F1": safe_divide(2 * precision * recall, precision + recall),
        "YesRatio": safe_divide(sum(predictions), len(predictions)),
    }

    print(json.dumps(metrics, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
