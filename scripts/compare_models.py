"""
compare_models.py - Generate comparison artifacts for both models.

Run:
    python scripts/compare_models.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.config import (
    COMPARE_CSV,
    COMPARE_MD,
    METRICS_EVAL_CNN,
    METRICS_EVAL_MNET,
    METRICS_EVAL_TRANS,
    METRICS_EVAL_YAMNET,
    METRICS_TRAIN_CNN,
    METRICS_TRAIN_MNET,
    METRICS_TRAIN_TRANS,
    METRICS_TRAIN_YAMNET,
    TRAIN_HISTORY_CNN,
    TRAIN_HISTORY_MNET,
    TRAIN_HISTORY_TRANS,
    TRAIN_HISTORY_YAMNET,
)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _best_val_accuracy(history_path: Path) -> float | None:
    if not history_path.exists():
        return None
    with history_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        values = []
        for row in reader:
            if "val_accuracy" in row and row["val_accuracy"]:
                try:
                    values.append(float(row["val_accuracy"]))
                except ValueError:
                    continue
        return max(values) if values else None


def _fmt(val: float | None, digits: int = 4) -> str:
    if val is None:
        return "n/a"
    return f"{val:.{digits}f}"


def _assemble_row(name: str, train_path: Path, eval_path: Path, history_path: Path) -> dict:
    train = _read_json(train_path)
    eval_ = _read_json(eval_path)
    best_val = _best_val_accuracy(history_path)
    test_acc = eval_.get("accuracy")
    gen_gap = None
    if best_val is not None and test_acc is not None:
        gen_gap = float(best_val) - float(test_acc)
    return {
        "model": name,
        "test_accuracy": eval_.get("accuracy"),
        "f1_weighted": eval_.get("f1_weighted"),
        "train_seconds": train.get("train_seconds"),
        "best_val_accuracy": best_val,
        "generalization_gap": gen_gap,
    }


def write_csv(rows: list[dict]) -> None:
    COMPARE_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "test_accuracy",
        "f1_weighted",
        "train_seconds",
        "best_val_accuracy",
        "generalization_gap",
    ]
    with COMPARE_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_md(rows: list[dict]) -> None:
    COMPARE_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Model Comparison",
        "",
        "| Model | Test Acc | Weighted F1 | Train Seconds | Best Val Acc | Gen Gap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| {model} | {acc} | {f1} | {t} | {val} | {gap} |".format(
                model=r["model"],
                acc=_fmt(r.get("test_accuracy")),
                f1=_fmt(r.get("f1_weighted")),
                t=_fmt(r.get("train_seconds"), digits=2),
                val=_fmt(r.get("best_val_accuracy")),
                gap=_fmt(r.get("generalization_gap")),
            )
        )
    COMPARE_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    rows = [
        _assemble_row("mobilenetv2", METRICS_TRAIN_MNET, METRICS_EVAL_MNET, TRAIN_HISTORY_MNET),
        _assemble_row("custom_cnn", METRICS_TRAIN_CNN, METRICS_EVAL_CNN, TRAIN_HISTORY_CNN),
        _assemble_row("transformer", METRICS_TRAIN_TRANS, METRICS_EVAL_TRANS, TRAIN_HISTORY_TRANS),
        _assemble_row("yamnet", METRICS_TRAIN_YAMNET, METRICS_EVAL_YAMNET, TRAIN_HISTORY_YAMNET),
    ]
    write_csv(rows)
    write_md(rows)
    print(f"Wrote {COMPARE_CSV} and {COMPARE_MD}")


if __name__ == "__main__":
    main()
