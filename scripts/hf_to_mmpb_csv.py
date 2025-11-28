#!/usr/bin/env python3
"""
Dump only the CSV metadata part of the SNU-AIDAS/MMPB dataset into the same
column format used by MMPB_old, without copying any image/JSON assets.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

os.environ.setdefault("USE_TORCH", "0")

from datasets import load_dataset  # type: ignore
from tqdm import tqdm  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export only dataset.csv for MMPB_old format.")
    parser.add_argument(
        "--csv-path",
        required=True,
        type=Path,
        help="Destination CSV path (e.g., /mnt/.../mmpb_test/dataset.csv).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of rows to export (for smoke tests).",
    )
    return parser.parse_args()


def normalize_name(name: str) -> str:
    return name.strip()


def choose_prefix(category: str, l2: str, target: str | None) -> str:
    if category == "preference":
        if l2 == "awareness":
            if target:
                return "likes" if target.lower() == "like" else "dislikes"
            return "awareness"
        if l2 == "inconsistency":
            return "inconsistent"
        if l2 == "overconcept":
            return "negative"
    if category == "recognition":
        if target in ("Homo", "Hetro"):
            return "negative"
        return ""
    return f"{l2 or 'sample'}"


def build_rel_path(sample: dict, counters: Dict[Tuple[str, str, str, str, str], int]) -> str:
    attribute = sample["attribute"]
    name = normalize_name(sample["name"])
    category = sample["category"]
    concept = sample.get("concept")
    concept_component = concept.strip() if isinstance(concept, str) and concept.strip() else None
    l2 = sample["l2-category"] or ""
    target = sample.get("target")

    prefix = choose_prefix(category, l2, target)
    counter_key = (attribute, name, category, concept_component or "", prefix)
    idx = counters[counter_key]
    counters[counter_key] += 1

    filename = f"{prefix}{idx}.png" if prefix else f"{idx}.png"
    parts = [attribute, "test", name, category]
    if concept_component:
        parts.append(concept_component)
    return str(Path(*parts) / filename)


def main() -> None:
    args = parse_args()
    dataset = load_dataset("SNU-AIDAS/MMPB")["train"]
    iterator = dataset if args.limit is None else dataset.select(range(args.limit))
    iterable = tqdm(iterator, desc="Writing CSV", total=len(iterator))

    counters: Dict[Tuple[str, str, str, str, str], int] = defaultdict(int)
    csv_path = args.csv_path.resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "index",
        "question",
        "A",
        "B",
        "C",
        "D",
        "image_path",
        "answer",
        "attribute",
        "category",
        "l2-category",
        "concept",
        "target",
        "name",
        "preference",
        "description_simple",
        "description_moderate",
        "description_detailed",
        "description_super_detailed",
        "prompt",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for sample in iterable:
            rel_path = build_rel_path(sample, counters)
            writer.writerow(
                {
                    "index": sample["index"],
                    "question": sample["question"],
                    "A": sample["A"] or "",
                    "B": sample["B"] or "",
                    "C": sample["C"] or "",
                    "D": sample["D"] or "",
                    "image_path": rel_path,
                    "answer": sample["answer"],
                    "attribute": sample["attribute"],
                    "category": sample["category"],
                    "l2-category": sample["l2-category"],
                    "concept": sample["concept"] or "",
                    "target": sample["target"] or "",
                    "name": normalize_name(sample["name"]),
                    "preference": sample.get("preference") or "",
                    "description_simple": sample.get("description_simple") or "",
                    "description_moderate": sample.get("description_moderate") or "",
                    "description_detailed": sample.get("description_detailed") or "",
                    "description_super_detailed": sample.get("description_super_detailed") or "",
                    "prompt": sample.get("preference") or "",
                }
            )

    print(f"Wrote CSV to {csv_path}")


if __name__ == "__main__":
    main()
