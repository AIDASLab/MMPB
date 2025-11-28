#!/usr/bin/env python3
"""
Convert the Hugging Face version of MMPB (SNU-AIDAS/MMPB) into the on-disk layout.

The converter materializes:
  * Injection/reference images under <root>/<attribute>/train/<name>/<0-4>.png
  * Evaluation images under   <root>/<attribute>/test/<name>/<category>/<concept>/...
  * dataset.csv pointing to the saved evaluation images
  * Optional per-identity JSON files (awareness/inconsistency/overconcept) that
    mirror the structure stored in the backed-up dataset.

Usage:
    python hf_to_mmpb_old.py --output-root ./mmpb_dataset --csv-path ./dataset.csv

The script may be re-run; existing files are overwritten.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# Avoid importing torch when loading datasets.
os.environ.setdefault("USE_TORCH", "0")

from datasets import load_dataset  # type: ignore
from PIL import Image
from tqdm import tqdm  # type: ignore


ChoiceDict = Dict[str, str]
PrefJsonCache = Dict[Tuple[str, str, str, str], Dict[str, Dict[str, Dict[str, str]]]]
RecJsonCache = Dict[Tuple[str, str, str, str], Dict[str, List[Dict[str, str]]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize SNU-AIDAS/MMPB locally.")
    parser.add_argument(
        "--output-root",
        default="MMPB_old",
        type=Path,
        help="Root directory where the dataset will be written.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        help="Optional explicit path for dataset.csv (defaults to <output-root>/dataset.csv).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only this many examples (useful for smoke-testing).",
    )
    parser.add_argument(
        "--skip-json",
        action="store_true",
        help="If set, do not emit the auxiliary awareness/inconsistency/overconcept JSON files.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_name(name: str) -> str:
    return name.strip()


def save_image(img: Image.Image, path: Path) -> None:
    ensure_dir(path.parent)
    mode = "RGB" if img.mode not in ("RGB", "L") else img.mode
    img.convert(mode).save(path, format="PNG")


def has_choices(sample: dict) -> bool:
    return any(sample.get(opt) is not None for opt in ("A", "B", "C", "D"))


def format_answer(sample: dict) -> str:
    answer = sample.get("answer")
    if answer is None:
        return ""
    if has_choices(sample):
        mapping = {0: "A", 1: "B", 2: "C", 3: "D"}
        return mapping.get(int(answer), "")
    yn_map = {4: "Yes", 5: "No"}
    return yn_map.get(int(answer), str(answer))


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
        # Positive samples fall back to numeric filenames.
        return ""
    return f"{l2 or 'sample'}"


def build_rel_path(
    sample: dict,
    counters: Dict[Tuple[str, str, str, str, str], int],
) -> Path:
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
    return Path(*parts) / filename


def record_pref_json_entry(
    cache: PrefJsonCache,
    sample: dict,
    rel_path: Path,
    formatted_answer: str,
) -> None:
    concept = sample["concept"]
    if not concept:
        return
    key = (
        sample["attribute"],
        normalize_name(sample["name"]),
        sample["category"],
        sample["l2-category"],
    )
    concept_map = cache.setdefault(key, {})
    images_map = concept_map.setdefault(concept, {})
    images_map[f"./{rel_path.as_posix()}"] = {
        "Query": sample["question"],
        "Answer": formatted_answer,
    }


def record_rec_json_entry(
    cache: RecJsonCache,
    sample: dict,
    rel_path: Path,
    formatted_answer: str,
) -> None:
    key = (
        sample["attribute"],
        normalize_name(sample["name"]),
        sample["category"],
        sample["l2-category"],
    )
    path_key = f"./{rel_path.as_posix()}"
    entry = {
        "Query": sample["question"],
        "Answer": formatted_answer,
    }
    if sample.get("target"):
        entry["Target"] = sample["target"]
    cache.setdefault(key, {}).setdefault(path_key, []).append(entry)


def write_pref_jsons(cache: PrefJsonCache, root: Path) -> None:
    for (attribute, name, category, l2), concept_map in cache.items():
        dest = root / attribute / "test" / name / category / f"{l2}.json"
        ensure_dir(dest.parent)
        serializable = {
            concept: dict(sorted(paths.items(), key=lambda kv: kv[0]))
            for concept, paths in sorted(concept_map.items())
        }
        with dest.open("w", encoding="utf-8") as fh:
            json.dump(serializable, fh, indent=4, ensure_ascii=False)


def write_rec_jsons(cache: RecJsonCache, root: Path) -> None:
    for (attribute, name, category, l2), path_map in cache.items():
        dest = root / attribute / "test" / name / category / f"{l2}.json"
        ensure_dir(dest.parent)
        serializable = {path: entries for path, entries in sorted(path_map.items())}
        with dest.open("w", encoding="utf-8") as fh:
            json.dump(serializable, fh, indent=4, ensure_ascii=False)


def save_injection_images(sample: dict, root: Path, saved: set[Tuple[str, str]]) -> None:
    key = (sample["attribute"], normalize_name(sample["name"]))
    if key in saved:
        return
    target_dir = root / sample["attribute"] / "train" / normalize_name(sample["name"])
    ensure_dir(target_dir)
    for idx in range(1, 6):
        image = sample.get(f"injection_image_{idx}")
        if image is None:
            continue
        save_image(image, target_dir / f"{idx-1}.png")
    saved.add(key)


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    csv_path = (args.csv_path or (output_root / "dataset.csv")).resolve()
    ensure_dir(output_root)
    ensure_dir(csv_path.parent)

    dataset = load_dataset("SNU-AIDAS/MMPB")["train"]

    counters: Dict[Tuple[str, str, str, str, str], int] = defaultdict(int)
    injection_saved: set[Tuple[str, str]] = set()
    pref_json_cache: PrefJsonCache = {}
    rec_json_cache: RecJsonCache = {}
    csv_rows: List[Dict[str, str]] = []

    iterator = dataset if args.limit is None else dataset.select(range(args.limit))

    for sample in tqdm(iterator, desc="Converting samples"):
        save_injection_images(sample, output_root, injection_saved)
        rel_path = build_rel_path(sample, counters)
        save_image(sample["image_path"], output_root / rel_path)
        formatted_answer = format_answer(sample)

        csv_rows.append(
            {
                "index": str(sample["index"]),
                "question": sample["question"],
                "A": sample["A"] or "",
                "B": sample["B"] or "",
                "C": sample["C"] or "",
                "D": sample["D"] or "",
                "image_path": rel_path.as_posix(),
                "answer": formatted_answer,
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

        if not args.skip_json:
            if sample["category"] == "preference":
                record_pref_json_entry(pref_json_cache, sample, rel_path, formatted_answer)
            else:
                record_rec_json_entry(rec_json_cache, sample, rel_path, formatted_answer)

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
        writer.writerows(csv_rows)

    if not args.skip_json:
        write_pref_jsons(pref_json_cache, output_root)
        write_rec_jsons(rec_json_cache, output_root)

    print(f"Wrote {len(csv_rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
