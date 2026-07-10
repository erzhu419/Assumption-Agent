from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 in the WSL research environment.
    import tomli as tomllib

from ..models import stable_hash
from ..splits import (
    AccessPhase,
    BenchmarkItem,
    SplitAccessGuard,
    build_family_out_manifest,
    build_instance_holdout_manifest,
)


class SkillLearnBenchAdapter:
    """Inventory SkillLearnBench without exposing verifier or solution content."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.tasks_root = self.root / "tasks"
        if not self.tasks_root.is_dir():
            raise FileNotFoundError(f"SkillLearnBench tasks directory not found: {self.tasks_root}")
        self._items: dict[str, BenchmarkItem] | None = None
        self._required_env_by_item: dict[str, tuple[str, ...]] | None = None

    def discover(self) -> list[BenchmarkItem]:
        if self._items is not None:
            return list(self._items.values())
        items: dict[str, BenchmarkItem] = {}
        required_env_by_item: dict[str, tuple[str, ...]] = {}
        for config_path in sorted(self.tasks_root.glob("*/*/task.toml")):
            instance_dir = config_path.parent
            family = instance_dir.parent.name
            item_id = instance_dir.name
            config = tomllib.loads(config_path.read_text(encoding="utf-8"))
            metadata = config.get("metadata", {}) if isinstance(config.get("metadata"), dict) else {}
            environment_config = (
                config.get("environment", {})
                if isinstance(config.get("environment"), dict)
                else {}
            )
            required_env_by_item[item_id] = tuple(
                sorted(
                    {
                        str(value).strip()
                        for value in environment_config.get("required_env", []) or []
                        if str(value).strip()
                    }
                )
            )
            instruction_path = instance_dir / "instruction.md"
            if not instruction_path.is_file():
                raise FileNotFoundError(f"missing instruction for {item_id}")
            verifier_files = sorted(
                str(path.relative_to(instance_dir))
                for path in (instance_dir / "tests").rglob("*")
                if path.is_file()
            )
            environment_files = sum(1 for path in (instance_dir / "environment").rglob("*") if path.is_file())
            features = {
                "benchmark": "skilllearnbench",
                "family": family,
                "category": str(metadata.get("category") or ""),
                "difficulty": str(metadata.get("difficulty") or ""),
                "tags": tuple(str(value) for value in metadata.get("tags", [])),
                "environment_file_count": environment_files,
                "has_container_environment": (instance_dir / "environment" / "Dockerfile").is_file(),
            }
            items[item_id] = BenchmarkItem(
                id=item_id,
                family=family,
                features=features,
                content_ref=str(instruction_path.relative_to(self.root)),
                verifier_ref_hash=stable_hash({"item_id": item_id, "verifier_file_refs": verifier_files}),
            )
        if len(items) != 100:
            raise ValueError(f"expected 100 SkillLearnBench instances, found {len(items)}")
        self._items = items
        self._required_env_by_item = required_env_by_item
        return list(items.values())

    def required_env_by_item(self) -> dict[str, tuple[str, ...]]:
        self.discover()
        return dict(self._required_env_by_item or {})

    def credential_independent_items(self) -> list[BenchmarkItem]:
        """Exclude complete families that require any external task credential."""

        items = self.discover()
        requirements = self.required_env_by_item()
        excluded_families = {
            item.family for item in items if requirements.get(item.id, ())
        }
        return [item for item in items if item.family not in excluded_families]

    def credential_independent_summary(self) -> dict[str, Any]:
        items = self.discover()
        requirements = self.required_env_by_item()
        excluded_families = sorted(
            {item.family for item in items if requirements.get(item.id, ())}
        )
        excluded_items = [item for item in items if item.family in excluded_families]
        return {
            "policy": "exclude_external_credentials_by_family_v1",
            "eligible_instance_count": len(items) - len(excluded_items),
            "excluded_instance_count": len(excluded_items),
            "excluded_families": excluded_families,
            "excluded_required_env_names": sorted(
                {
                    name
                    for item in excluded_items
                    for name in requirements.get(item.id, ())
                }
            ),
            "secret_value_persisted": False,
        }

    def load_instruction(self, item_id: str, *, phase: AccessPhase, guard: SplitAccessGuard) -> str:
        guard.authorize(item_id, phase)
        items = self._items or {item.id: item for item in self.discover()}
        item = items[item_id]
        return (self.root / item.content_ref).read_text(encoding="utf-8")

    def inventory_summary(self) -> dict[str, Any]:
        items = self.discover()
        families = sorted({item.family for item in items})
        categories = sorted({str(item.features.get("category") or "") for item in items})
        return {
            "benchmark": "skilllearnbench",
            "instance_count": len(items),
            "family_count": len(families),
            "category_count": len(categories),
            "families": families,
            "categories": categories,
            "all_verifier_refs_hashed": all(item.verifier_ref_hash for item in items),
            "verifier_content_exposed": False,
            "raw_content_persisted": False,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a sealed SkillLearnBench split manifest.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol", choices=("instance_holdout", "family_out"), default="instance_holdout")
    parser.add_argument("--seed", default="skilllearnbench-v1")
    parser.add_argument(
        "--credential-independent",
        action="store_true",
        help="Exclude every family containing a task that declares required_env.",
    )
    args = parser.parse_args()

    adapter = SkillLearnBenchAdapter(args.root)
    items = (
        adapter.credential_independent_items()
        if args.credential_independent
        else adapter.discover()
    )
    if args.protocol == "family_out":
        manifest = build_family_out_manifest(items, benchmark="skilllearnbench", seed=args.seed)
    else:
        manifest = build_instance_holdout_manifest(items, benchmark="skilllearnbench", seed=args.seed)
    manifest.write(args.out)
    print(
        json.dumps(
            {
                "inventory": adapter.inventory_summary(),
                "benchmark_subset": (
                    adapter.credential_independent_summary()
                    if args.credential_independent
                    else {"policy": "full_inventory_v1"}
                ),
                "manifest": manifest.to_dict(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
