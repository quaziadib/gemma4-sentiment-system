from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

from inference_sentiment import classify_text
from prompts import SENTIMENT_LABELS


def load_test_samples(path: Path) -> list[dict[str, str]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        samples: list[dict[str, str]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            text = record.get("text")
            label = record.get("label")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Each JSONL row must contain a non-empty 'text' field.")
            if label not in SENTIMENT_LABELS:
                raise ValueError(f"Invalid label in JSONL row: {label}")
            samples.append({"text": text, "label": label})
        return samples

    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if "text" not in (reader.fieldnames or []) or "label" not in (reader.fieldnames or []):
                raise ValueError("CSV file must contain 'text' and 'label' columns.")
            samples = []
            for row in reader:
                text = (row.get("text") or "").strip()
                label = (row.get("label") or "").strip()
                if not text:
                    raise ValueError("CSV rows must have a non-empty 'text' value.")
                if label not in SENTIMENT_LABELS:
                    raise ValueError(f"Invalid label in CSV row: {label}")
                samples.append({"text": text, "label": label})
            return samples

    raise ValueError("Unsupported test file format. Use .jsonl or .csv.")


def format_confusion_matrix(matrix: list[list[int]]) -> str:
    header = ["Actual \\ Pred"] + [label[:18] for label in SENTIMENT_LABELS]
    rows = [header]
    for label, values in zip(SENTIMENT_LABELS, matrix):
        rows.append([label[:18]] + [str(value) for value in values])

    widths = [max(len(row[col]) for row in rows) for col in range(len(header))]
    return "\n".join(
        " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)) for row in rows
    )


def evaluate(samples: list[dict[str, str]], args: argparse.Namespace) -> dict[str, Any]:
    predictions: list[dict[str, Any]] = []
    y_true: list[str] = []
    y_pred: list[str] = []

    for idx, sample in enumerate(samples, start=1):
        result = classify_text(
            sample["text"],
            api_url=args.api_url,
            model=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            retries=args.retries,
            retry_delay=args.retry_delay,
        )
        y_true.append(sample["label"])
        y_pred.append(result.sentiment)
        predictions.append(
            {
                "index": idx,
                "text": sample["text"],
                "ground_truth": sample["label"],
                "predicted_sentiment": result.sentiment,
                "predicted_label_id": result.label_id,
                "confidence": result.confidence,
                "correct": result.sentiment == sample["label"],
            }
        )
        print(f"[{idx}/{len(samples)}] true={sample['label']} pred={result.sentiment}")

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=SENTIMENT_LABELS,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=SENTIMENT_LABELS,
        digits=4,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=SENTIMENT_LABELS)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=SENTIMENT_LABELS, average="macro", zero_division=0),
        "weighted_f1": f1_score(
            y_true, y_pred, labels=SENTIMENT_LABELS, average="weighted", zero_division=0
        ),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred, labels=SENTIMENT_LABELS),
        "classification_report": report_dict,
        "confusion_matrix": {
            "labels": SENTIMENT_LABELS,
            "matrix": matrix.tolist(),
        },
    }

    return {
        "metrics": metrics,
        "classification_report_text": report_text,
        "confusion_matrix_table": format_confusion_matrix(matrix.tolist()),
        "predictions": predictions,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Bangla sentiment analysis predictions.")
    parser.add_argument("--test_file", required=True, help="Path to a JSONL or CSV file with text and label columns.")
    parser.add_argument(
        "--output_file",
        default="sentiment_evaluation_results.json",
        help="Where to save detailed predictions and aggregate metrics.",
    )
    parser.add_argument("--api_url", default="http://localhost:8001/v1/chat/completions")
    parser.add_argument("--model", default="gemma-4-26B-A4B")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry_delay", type=float, default=1.5)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    test_path = Path(args.test_file)
    output_path = Path(args.output_file)

    samples = load_test_samples(test_path)
    results = evaluate(samples, args)

    print("\nClassification Report")
    print(results["classification_report_text"])
    print("Confusion Matrix")
    print(results["confusion_matrix_table"])
    print(
        "\nAccuracy: {accuracy:.4f}\nMacro F1: {macro_f1:.4f}\nWeighted F1: {weighted_f1:.4f}\nCohen's Kappa: {cohen_kappa:.4f}".format(
            **results["metrics"]
        )
    )

    output_path.write_text(
        json.dumps(
            {
                "test_file": str(test_path),
                "sample_count": len(samples),
                "metrics": results["metrics"],
                "predictions": results["predictions"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved detailed evaluation results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
