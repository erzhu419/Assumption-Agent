"""Download gated HLE once for local offline evaluation.

The saved directory contains the raw gated HLE test split. Keep it local and
out of git. Downstream HLE runners use it when HLE_DATASET_LOCAL_PATH points to
the saved directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from .autonomy_journal import PAPER_DIR
from .hle_smoke_eval import DATASET_NAME, _hf_token


DEFAULT_OUT = PAPER_DIR / "hle_dataset_cache" / "test"


def download_hle_dataset_cache(
    *,
    out: Path,
    dataset_name: str = DATASET_NAME,
    split: str = "test",
    overwrite: bool = False,
) -> dict[str, Any]:
    token = _hf_token()
    if not token:
        raise RuntimeError("HF_TOKEN or HUGGINGFACE_HUB_TOKEN is required to download gated HLE.")
    out = out.expanduser()
    if out.exists():
        if not overwrite:
            raise FileExistsError(f"{out} already exists; pass --overwrite to replace it.")
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split, token=token)
    dataset.save_to_disk(str(out))
    manifest = {
        "dataset": dataset_name,
        "split": split,
        "out": str(out),
        "row_count": int(getattr(dataset, "num_rows", 0) or len(dataset)),
        "feature_names": sorted(str(name) for name in getattr(dataset, "features", {}).keys()),
        "raw_hle_content_persisted_locally": True,
        "git_safe": False,
        "usage": f"export HLE_DATASET_LOCAL_PATH='{out}'",
    }
    manifest_path = out.with_suffix(out.suffix + ".manifest.json") if out.suffix else Path(str(out) + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download cais/hle once and save it for offline local HLE runs.")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--split", default="test")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = download_hle_dataset_cache(
        out=Path(args.out),
        dataset_name=args.dataset,
        split=args.split,
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
