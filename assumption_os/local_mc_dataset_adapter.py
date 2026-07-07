"""Build local multiple-choice transfer JSONL files from bundled eval zips.

The transfer runner intentionally consumes a plain local JSONL file so HLE,
MMLU, C-Eval, and CMMLU handling stay decoupled.  This adapter creates that
JSONL from already-downloaded zip archives without touching HuggingFace or
source-search APIs.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
import zipfile
from pathlib import Path
from typing import Any, Iterable

from .autonomy_journal import stable_hash


DEFAULT_MMLU_ZIP = Path(
    "reference/repos/ReMA-public/src/360-LLaMA-Factory/evaluation/mmlu/mmlu.zip"
)
DEFAULT_CEVAL_ZIP = Path(
    "reference/repos/ReMA-public/src/360-LLaMA-Factory/evaluation/ceval/ceval.zip"
)
DEFAULT_CMMLU_ZIP = Path(
    "reference/repos/ReMA-public/src/360-LLaMA-Factory/evaluation/cmmlu/cmmlu.zip"
)
DEFAULT_OUT = Path("phase four/assumption_graph/local_transfer_datasets/local_mc_transfer.jsonl")


def build_local_mc_jsonl_from_zip(
    *,
    root: Path,
    dataset: str,
    zip_path: Path | None = None,
    split: str | None = None,
    output_jsonl: Path,
    sample_size: int = 24,
    seed: int = 0,
    subjects: Iterable[str] | None = None,
    max_per_subject: int | None = None,
    exclude_eval_paths: Iterable[Path] | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    dataset = dataset.lower().strip()
    if dataset not in {"mmlu", "ceval", "cmmlu"}:
        raise ValueError(f"unsupported dataset: {dataset}")
    split = (split or _default_split(dataset)).lower().strip()
    zip_path = zip_path or _default_zip_for_dataset(dataset)
    zip_path = zip_path if zip_path.is_absolute() else root / zip_path
    output_jsonl = output_jsonl if output_jsonl.is_absolute() else root / output_jsonl
    selected_subjects = {subject.strip() for subject in (subjects or []) if subject.strip()}
    excluded_hashes = _load_excluded_hashes(exclude_eval_paths or [], root=root)

    rows = list(_iter_dataset_rows(dataset=dataset, zip_path=zip_path, split=split))
    if selected_subjects:
        rows = [row for row in rows if row["subject"] in selected_subjects]
    before_exclude = len(rows)
    rows = [row for row in rows if not _row_matches_excluded_hash(row, excluded_hashes)]
    rows = _deterministic_sample(
        rows,
        sample_size=sample_size,
        seed=seed,
        max_per_subject=max_per_subject,
    )

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    by_subject: dict[str, int] = {}
    for row in rows:
        by_subject[row["subject"]] = by_subject.get(row["subject"], 0) + 1
    return {
        "dataset": dataset,
        "split": split,
        "zip_path": str(zip_path),
        "output_jsonl": str(output_jsonl),
        "available_after_filters_before_exclude": before_exclude,
        "excluded_hash_count": len(excluded_hashes),
        "sample_size_requested": sample_size,
        "sample_size_written": len(rows),
        "seed": seed,
        "subjects_requested": sorted(selected_subjects),
        "subjects_written": sorted(by_subject),
        "by_subject": by_subject,
        "output_hash": stable_hash({"rows": [_row_public_id(row) for row in rows]}),
        "raw_content_in_output_jsonl": True,
    }


def _default_zip_for_dataset(dataset: str) -> Path:
    return {
        "mmlu": DEFAULT_MMLU_ZIP,
        "ceval": DEFAULT_CEVAL_ZIP,
        "cmmlu": DEFAULT_CMMLU_ZIP,
    }[dataset]


def _default_split(dataset: str) -> str:
    if dataset == "ceval":
        return "val"
    return "test"


def _iter_dataset_rows(*, dataset: str, zip_path: Path, split: str) -> Iterable[dict[str, Any]]:
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        for member, subject in _iter_members(dataset=dataset, archive=archive, split=split):
            text = archive.read(member).decode("utf-8-sig")
            yield from _parse_member_rows(dataset=dataset, split=split, member=member, subject=subject, text=text)


def _iter_members(*, dataset: str, archive: zipfile.ZipFile, split: str) -> Iterable[tuple[str, str]]:
    names = sorted(name for name in archive.namelist() if name.endswith(".csv"))
    if dataset == "mmlu":
        split_dir = {"dev": "dev", "train": "dev", "val": "val", "validation": "val", "test": "test"}[split]
        pattern = re.compile(rf"^data/{re.escape(split_dir)}/(.+)_{re.escape(split_dir)}\.csv$")
    elif dataset == "ceval":
        split_dir = {"dev": "dev", "train": "dev", "val": "val", "validation": "val", "test": "test"}[split]
        pattern = re.compile(rf"^{re.escape(split_dir)}/(.+)_{re.escape(split_dir)}\.csv$")
    else:
        split_dir = {"dev": "dev", "train": "dev", "test": "test"}[split]
        pattern = re.compile(rf"^{re.escape(split_dir)}/(.+)\.csv$")
    for name in names:
        match = pattern.match(name)
        if match:
            yield name, match.group(1)


def _parse_member_rows(
    *,
    dataset: str,
    split: str,
    member: str,
    subject: str,
    text: str,
) -> Iterable[dict[str, Any]]:
    if dataset == "mmlu":
        reader = csv.reader(io.StringIO(text))
        for index, row in enumerate(reader):
            if len(row) < 6:
                continue
            question, a, b, c, d, answer = [cell.strip() for cell in row[:6]]
            normalized = _normalized_row(
                dataset=dataset,
                split=split,
                subject=subject,
                row_id=str(index),
                question=question,
                choices={"A": a, "B": b, "C": c, "D": d},
                answer=answer,
            )
            if normalized is not None:
                yield normalized
        return

    reader = csv.DictReader(io.StringIO(text))
    for index, row in enumerate(reader):
        question = str(row.get("question") or row.get("Question") or "").strip()
        answer = str(row.get("answer") or row.get("Answer") or "").strip()
        row_id = str(row.get("id") or row.get("") or index).strip()
        normalized = _normalized_row(
            dataset=dataset,
            split=split,
            subject=subject,
            row_id=row_id,
            question=question,
            choices={
                "A": str(row.get("A") or "").strip(),
                "B": str(row.get("B") or "").strip(),
                "C": str(row.get("C") or "").strip(),
                "D": str(row.get("D") or "").strip(),
            },
            answer=answer,
        )
        if normalized is not None:
            yield normalized


def _normalized_row(
    *,
    dataset: str,
    split: str,
    subject: str,
    row_id: str,
    question: str,
    choices: dict[str, str],
    answer: str,
) -> dict[str, Any] | None:
    answer = answer.upper()[:1]
    if answer not in choices:
        return None
    if not question or any(not choices.get(label, "").strip() for label in ("A", "B", "C", "D")):
        return None
    public_id = f"{dataset}:{split}:{subject}:{row_id}"
    return {
        "id": public_id,
        "dataset": dataset,
        "source": f"{dataset}_local_zip",
        "split": split,
        "category": f"{dataset}_transfer",
        "subject": subject,
        "question": question,
        "choices": {label: choices[label].strip() for label in ("A", "B", "C", "D")},
        "answer": answer,
    }


def _deterministic_sample(
    rows: list[dict[str, Any]],
    *,
    sample_size: int,
    seed: int,
    max_per_subject: int | None = None,
) -> list[dict[str, Any]]:
    rows = list(rows)
    rng = random.Random(seed)
    rng.shuffle(rows)
    if max_per_subject is not None and max_per_subject > 0:
        rows = _cap_per_subject(rows, max_per_subject=max_per_subject)
    if sample_size > 0:
        rows = rows[:sample_size]
    return sorted(rows, key=_row_public_id)


def _cap_per_subject(rows: list[dict[str, Any]], *, max_per_subject: int) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    capped: list[dict[str, Any]] = []
    for row in rows:
        subject = row["subject"]
        if counts.get(subject, 0) >= max_per_subject:
            continue
        capped.append(row)
        counts[subject] = counts.get(subject, 0) + 1
    return capped


def _row_public_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or "")


def _load_excluded_hashes(paths: Iterable[Path], *, root: Path) -> set[str]:
    hashes: set[str] = set()
    for raw_path in paths:
        path = raw_path if raw_path.is_absolute() else root / raw_path
        if not path.exists():
            continue
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        _collect_hashes_from_obj(json.loads(line), hashes)
        else:
            _collect_hashes_from_obj(json.loads(path.read_text(encoding="utf-8")), hashes)
    return hashes


def _collect_hashes_from_obj(obj: Any, hashes: set[str]) -> None:
    if isinstance(obj, dict):
        for key in ("question_hash", "problem_id_hash", "id_hash"):
            value = obj.get(key)
            if isinstance(value, str) and value:
                hashes.add(value)
        sampling = obj.get("sampling")
        if isinstance(sampling, dict):
            for value in sampling.get("sample_problem_hashes") or []:
                if isinstance(value, str) and value:
                    hashes.add(value)
        for key in ("rows", "items", "examples"):
            value = obj.get(key)
            if isinstance(value, list):
                for item in value:
                    _collect_hashes_from_obj(item, hashes)
    elif isinstance(obj, list):
        for item in obj:
            _collect_hashes_from_obj(item, hashes)


def _row_matches_excluded_hash(row: dict[str, Any], excluded_hashes: set[str]) -> bool:
    if not excluded_hashes:
        return False
    question = _question_with_option_lines(row)
    question_hash = stable_hash({"local_mc_question": question})
    id_hash = stable_hash({"local_mc_id": row["id"], "question_hash": question_hash})
    return question_hash in excluded_hashes or id_hash in excluded_hashes


def _question_with_option_lines(row: dict[str, Any]) -> str:
    question = str(row.get("question") or "").strip()
    if re.search(r"(?m)^\s*[A-D][\).\:]\s+\S+", question):
        return question
    choices = row.get("choices") or {}
    lines = [f"{label}. {choices.get(label, '')}" for label in ("A", "B", "C", "D")]
    return question + "\n" + "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert bundled MC eval zips to local transfer JSONL.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--dataset", choices=["mmlu", "ceval", "cmmlu"], required=True)
    parser.add_argument("--zip-path", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--sample-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--subjects", default="")
    parser.add_argument("--max-per-subject", type=int, default=None)
    parser.add_argument("--exclude-eval-json", action="append", default=[])
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    root = Path(args.root).resolve()
    payload = build_local_mc_jsonl_from_zip(
        root=root,
        dataset=args.dataset,
        zip_path=Path(args.zip_path) if args.zip_path else None,
        split=args.split,
        output_jsonl=Path(args.out),
        sample_size=args.sample_size,
        seed=args.seed,
        subjects=[item.strip() for item in args.subjects.split(",") if item.strip()],
        max_per_subject=args.max_per_subject,
        exclude_eval_paths=[Path(item) for item in args.exclude_eval_json],
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
