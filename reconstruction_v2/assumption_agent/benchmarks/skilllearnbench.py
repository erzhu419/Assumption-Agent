from __future__ import annotations

import argparse
import json
import re
import shlex
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


OFFLINE_BLOCKED_FAMILIES = frozenset(
    {
        "fix-security-bug",
        "nlp-paper-reproduction",
        "python-scala-translation",
    }
)
OFFLINE_BLOCKED_ITEM_IDS = frozenset({"weighted-gdp-calculation-2"})
OFFLINE_READY_SUBSET_POLICY = (
    "exclude_external_credentials_and_offline_blockers_v1"
)
TRAIN_ACTION_ENVIRONMENT_PROFILE_VERSION = (
    "train_nonoracle_allowlisted_environment_facts_v2"
)

_TASK_LOCAL_PATH_PATTERN = re.compile(r"/root(?:/[A-Za-z0-9._+-]+)+")
_SAFE_RELATIVE_ENVIRONMENT_PATH_PATTERN = re.compile(
    r"[A-Za-z0-9._+-]+(?:/[A-Za-z0-9._+-]+)*"
)
_FORBIDDEN_NONORACLE_PATH_PATTERN = re.compile(
    r"(?:^|[\s/_.-])(?:tests?|verifier|oracle|solutions?)(?:$|[\s/_.-])",
    re.IGNORECASE,
)
_SENSITIVE_PATH_COMPONENT_PATTERN = re.compile(
    r"(?:^|[/_.-])(?:api[-_]?keys?|access[-_]?tokens?|auth|credentials?|"
    r"env|id[-_]?rsa|netrc|npmrc|passwords?|passwds?|private|pypirc|"
    r"secrets?|sk-[A-Za-z0-9_-]{8,}|tokens?|keys?)"
    r"(?:$|[/_.-])",
    re.IGNORECASE,
)
_SENSITIVE_ENVIRONMENT_TEXT_PATTERN = re.compile(
    r"(?ix)(?:"
    r"(?:^|\s)(?:-H|--header|--api[-_]?key|--access[-_]?token|--auth|"
    r"--password|--passwd|--secret|--token)(?:[=\s]|$)|"
    r"\bBearer\s+|"
    r"\b[A-Z0-9_]*(?:API[-_]?KEY|ACCESS[-_]?TOKEN|AUTH[-_]?TOKEN|"
    r"PASSWORD|PASSWD|SECRET|TOKEN)\s*=|"
    r"\bsk-[A-Za-z0-9_-]{8,}\b|"
    r"https?://[^\s/@:]+:[^\s/@]+@|"
    r"(?:\?|&)[A-Za-z0-9_.~-]+=[^\s&]+"
    r")"
)
_ENVIRONMENT_NOTE_FACT_PATTERNS = {
    "local_cache_declared": re.compile(r"\bcache\b", re.IGNORECASE),
    "local_database_declared": re.compile(
        r"\b(?:database|sqlite|db)\b", re.IGNORECASE
    ),
    "offline_assets_declared": re.compile(r"\boffline\b", re.IGNORECASE),
    "structured_csv_artifact_declared": re.compile(r"\bcsv\b", re.IGNORECASE),
    "structured_json_artifact_declared": re.compile(r"\bjson\b", re.IGNORECASE),
    "pdf_artifact_declared": re.compile(r"\bpdf\b", re.IGNORECASE),
}
_SKILL_INSTALL_PATH_MARKERS = (
    "/root/.agents/skills",
    "/root/.claude/skills",
    "/root/.codex/skills",
    "/root/.factory/skills",
    "/root/.gemini/skills",
    "/root/.goose/skills",
    "/root/.opencode/skill",
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
            _reject_symlink_path(
                config_path,
                anchor=self.root,
                label="task inventory",
            )
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

    def offline_ready_items(self) -> list[BenchmarkItem]:
        """Return the preregistered subset with an offline verifier runtime."""

        return [
            item
            for item in self.credential_independent_items()
            if item.family not in OFFLINE_BLOCKED_FAMILIES
            and item.id not in OFFLINE_BLOCKED_ITEM_IDS
        ]

    def offline_ready_summary(self) -> dict[str, Any]:
        items = self.discover()
        ready = self.offline_ready_items()
        credential_summary = self.credential_independent_summary()
        excluded_families = sorted(
            {
                *credential_summary["excluded_families"],
                *OFFLINE_BLOCKED_FAMILIES,
            }
        )
        return {
            "policy": OFFLINE_READY_SUBSET_POLICY,
            "eligible_instance_count": len(ready),
            "eligible_family_count": len({item.family for item in ready}),
            "excluded_instance_count": len(items) - len(ready),
            "excluded_families": excluded_families,
            "excluded_item_ids": sorted(OFFLINE_BLOCKED_ITEM_IDS),
            "excluded_required_env_names": credential_summary[
                "excluded_required_env_names"
            ],
            "offline_blocked_families": sorted(OFFLINE_BLOCKED_FAMILIES),
            "offline_blocked_item_ids": sorted(OFFLINE_BLOCKED_ITEM_IDS),
            "secret_value_persisted": False,
        }

    def selected_payload_fingerprint(
        self,
        item_ids: Any,
    ) -> dict[str, Any]:
        """Hash selected task/runtime/verifier bytes without exposing content."""

        items = {item.id: item for item in self.discover()}
        selected = tuple(sorted(str(item_id) for item_id in item_ids))
        unknown = sorted(set(selected) - set(items))
        if unknown:
            raise ValueError("benchmark fingerprint contains unknown item IDs")
        file_rows: list[dict[str, str]] = []
        for item_id in selected:
            instance = self.root / items[item_id].content_ref
            instance = instance.parent
            paths = [instance / "task.toml", instance / "instruction.md"]
            for directory_name in ("environment", "tests"):
                directory = instance / directory_name
                if directory.is_dir():
                    paths.extend(path for path in directory.rglob("*") if path.is_file())
            for path in sorted(set(paths)):
                if not path.is_file():
                    raise FileNotFoundError(
                        f"selected benchmark payload is missing for {item_id}"
                    )
                file_rows.append(
                    {
                        "item_id_hash": stable_hash({"item_id": item_id}),
                        "path": str(path.relative_to(instance)),
                        "content_hash": stable_hash({"bytes": path.read_bytes().hex()}),
                    }
                )
        return {
            "policy": "selected_task_environment_verifier_bytes_v1",
            "selected_item_count": len(selected),
            "file_count": len(file_rows),
            "tree_hash": stable_hash(file_rows),
            "raw_content_persisted": False,
        }

    def load_instruction(self, item_id: str, *, phase: AccessPhase, guard: SplitAccessGuard) -> str:
        guard.authorize(item_id, phase)
        items = self._items or {item.id: item for item in self.discover()}
        item = items[item_id]
        return self._bound_item_content_path(item).read_text(encoding="utf-8")

    def load_action_design_context(
        self,
        item_id: str,
        *,
        phase: AccessPhase,
        guard: SplitAccessGuard,
    ) -> dict[str, Any]:
        """Return a compact TRAIN-only inventory of non-oracle runtime facts.

        The profile deliberately reads only the task's public environment directory.
        Verifier, solution, and test content are outside the traversal root and are
        never inspected.  The proposal model receives normalized capabilities, not
        an unrestricted filesystem handle.
        """

        if phase is not AccessPhase.PROPOSAL:
            raise PermissionError(
                "action design context is restricted to the proposal phase"
            )
        guard.authorize(item_id, phase)
        items = self._items or {item.id: item for item in self.discover()}
        item = items[item_id]
        instance_dir = self._bound_item_content_path(item).parent
        environment_dir = _contained_environment_directory(instance_dir)
        docker_text, dockerfile_present = _read_contained_environment_file(
            environment_dir,
            "Dockerfile",
        )
        notes_text, readme_present = _read_contained_environment_file(
            environment_dir,
            "README.md",
        )
        logical_lines = _dockerfile_logical_lines(docker_text)
        fact_lines = [
            line
            for line in logical_lines
            if not _SENSITIVE_ENVIRONMENT_TEXT_PATTERN.search(line)
            and not _FORBIDDEN_NONORACLE_PATH_PATTERN.search(line)
        ]
        working_directory = _docker_working_directory(fact_lines)
        os_packages = _docker_installed_packages(fact_lines, manager="apt")
        python_packages = _docker_installed_packages(fact_lines, manager="pip")
        declared_paths = {
            path.rstrip(".,;:)")
            for line in fact_lines
            for path in _TASK_LOCAL_PATH_PATTERN.findall(line)
            if not any(marker in path for marker in _SKILL_INSTALL_PATH_MARKERS)
            and _safe_task_local_path(path) is not None
        }
        copied_files = _docker_task_copy_destinations(fact_lines)
        declared_paths.update(copied_files)
        environment_files = _contained_environment_source_files(environment_dir)
        profile = {
            "policy": TRAIN_ACTION_ENVIRONMENT_PROFILE_VERSION,
            "working_directory": working_directory,
            "declared_os_packages": os_packages,
            "declared_python_packages": python_packages,
            "declared_task_local_paths": sorted(declared_paths),
            "copied_task_files": copied_files,
            "environment_source_files": environment_files,
            "environment_note_facts": _allowlisted_environment_note_facts(
                notes_text
            ),
            "setup_operation_facts": _allowlisted_setup_operation_facts(
                fact_lines
            ),
            "setup_facts_role": "build_time_provenance_not_runtime_command",
            "dockerfile_present": dockerfile_present,
            "dockerfile_hash": (
                stable_hash({"dockerfile": docker_text}) if docker_text else ""
            ),
            "readme_present": readme_present,
            "readme_hash": (
                stable_hash({"readme": notes_text}) if notes_text else ""
            ),
            "source_scope": "train_task_environment_only",
            "verifier_content_used": False,
            "solution_content_used": False,
            "test_content_used": False,
            "raw_environment_content_persisted": False,
            "raw_environment_notes_persisted": False,
            "raw_setup_recipe_persisted": False,
        }
        return profile

    def _bound_item_content_path(self, item: BenchmarkItem) -> Path:
        expected_content_ref = (
            Path("tasks") / item.family / item.id / "instruction.md"
        )
        if Path(item.content_ref) != expected_content_ref:
            raise PermissionError(
                "task content path is not bound to its item identity"
            )
        content_path = self.root / expected_content_ref
        _reject_symlink_path(
            content_path,
            anchor=self.root,
            label="task instance",
        )
        resolved_content = content_path.resolve(strict=True)
        if not _path_is_within(
            resolved_content,
            self.tasks_root.resolve(strict=True),
        ):
            raise PermissionError(
                "task content resolves outside the benchmark task root"
            )
        return resolved_content

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


def _dockerfile_logical_lines(text: str) -> list[str]:
    lines: list[str] = []
    pending = ""
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        pending = f"{pending} {stripped}".strip()
        if pending.endswith("\\"):
            pending = pending[:-1].rstrip()
            continue
        lines.append(" ".join(pending.split()))
        pending = ""
    if pending:
        lines.append(" ".join(pending.split()))
    return lines


def _docker_working_directory(lines: list[str]) -> str:
    for line in reversed(lines):
        if line.upper().startswith("WORKDIR "):
            return _safe_task_local_path(line.split(maxsplit=1)[1].strip()) or ""
    return ""


def _docker_installed_packages(lines: list[str], *, manager: str) -> list[str]:
    if manager == "apt":
        pattern = re.compile(
            r"\bapt(?:-get)?\s+install\s+(?:-[A-Za-z0-9-]+\s+)*(.*?)(?=\s*(?:&&|;|$))",
            re.IGNORECASE,
        )
    elif manager == "pip":
        pattern = re.compile(
            r"\b(?:pip3?|python3?\s+-m\s+pip)\s+install\s+(.*?)(?=\s*(?:&&|;|$))",
            re.IGNORECASE,
        )
    else:
        raise ValueError(f"unsupported Docker package manager: {manager}")
    packages: set[str] = set()
    for line in lines:
        for match in pattern.finditer(line):
            if _SENSITIVE_ENVIRONMENT_TEXT_PATTERN.search(match.group(1)):
                continue
            try:
                tokens = shlex.split(match.group(1))
            except ValueError:
                tokens = match.group(1).split()
            for token in tokens:
                value = token.strip().rstrip("\\")
                if (
                    not value
                    or value.startswith("-")
                    or value in {"apt-get", "apt", "pip", "pip3", "python", "python3"}
                    or "$" in value
                    or "/" in value
                    or not re.fullmatch(
                        r"[A-Za-z0-9][A-Za-z0-9._+-]*(?:==[A-Za-z0-9][A-Za-z0-9._+-]*)?",
                        value,
                    )
                    or _SENSITIVE_ENVIRONMENT_TEXT_PATTERN.search(value)
                    or _SENSITIVE_PATH_COMPONENT_PATTERN.search(value)
                ):
                    continue
                packages.add(value)
    return sorted(packages)


def _docker_task_copy_destinations(lines: list[str]) -> list[str]:
    destinations: set[str] = set()
    for line in lines:
        if not line.upper().startswith("COPY "):
            continue
        try:
            tokens = shlex.split(line)
        except ValueError:
            tokens = line.split()
        values = [value for value in tokens[1:] if not value.startswith("--")]
        if len(values) < 2:
            continue
        destination = values[-1]
        sources = values[:-1]
        if any("skills" in source.lower() for source in sources):
            continue
        for source in sources:
            basename = Path(source).name
            resolved = (
                f"{destination.rstrip('/')}/{basename}"
                if destination.endswith("/")
                else destination
            )
            safe_destination = _safe_task_local_path(resolved)
            if safe_destination is not None:
                destinations.add(safe_destination)
    return sorted(destinations)


def _allowlisted_environment_note_facts(text: str) -> list[str]:
    """Map README text to fixed labels; never return README fragments."""

    if not text:
        return []
    return sorted(
        fact
        for fact, pattern in _ENVIRONMENT_NOTE_FACT_PATTERNS.items()
        if pattern.search(text)
    )


def _allowlisted_setup_operation_facts(lines: list[str]) -> list[str]:
    """Map Docker operations to fixed labels; never return Dockerfile lines."""

    operation_patterns = {
        "apt_package_install": re.compile(r"\bapt(?:-get)?\s+install\b", re.I),
        "pip_package_install": re.compile(
            r"\b(?:pip3?|python3?\s+-m\s+pip)\s+install\b", re.I
        ),
        "task_file_copy": re.compile(r"^COPY\s+", re.I),
        "local_directory_create": re.compile(r"\bmkdir\b", re.I),
        "local_symlink_create": re.compile(r"\bln\s+-s\b", re.I),
        "local_archive_extract": re.compile(r"\b(?:tar|unzip)\b", re.I),
    }
    facts: set[str] = set()
    for line in lines:
        if _SENSITIVE_ENVIRONMENT_TEXT_PATTERN.search(line):
            continue
        if re.search(r"^COPY\s+.*skills", line, re.IGNORECASE):
            continue
        facts.update(
            fact for fact, pattern in operation_patterns.items() if pattern.search(line)
        )
    return sorted(facts)


def _contained_environment_directory(instance_dir: Path) -> Path:
    candidate = instance_dir / "environment"
    if candidate.is_symlink():
        raise PermissionError("environment directory symlinks are forbidden")
    if not candidate.exists():
        return candidate
    resolved = candidate.resolve(strict=True)
    if not resolved.is_dir() or not _path_is_within(resolved, instance_dir):
        raise PermissionError("environment directory resolves outside task instance")
    return resolved


def _read_contained_environment_file(
    environment_dir: Path,
    filename: str,
) -> tuple[str, bool]:
    candidate = environment_dir / filename
    if candidate.is_symlink():
        raise PermissionError(f"environment metadata symlink is forbidden: {filename}")
    if not candidate.exists():
        return "", False
    resolved = candidate.resolve(strict=True)
    if not _path_is_within(resolved, environment_dir) or not resolved.is_file():
        raise PermissionError(f"environment metadata escapes containment: {filename}")
    return resolved.read_text(encoding="utf-8", errors="replace"), True


def _contained_environment_source_files(environment_dir: Path) -> list[str]:
    if not environment_dir.is_dir():
        return []
    paths: list[str] = []
    for path in environment_dir.rglob("*"):
        if path.is_symlink():
            raise PermissionError("environment source symlinks are forbidden")
        if not path.is_file():
            continue
        resolved = path.resolve(strict=True)
        if not _path_is_within(resolved, environment_dir):
            raise PermissionError("environment source resolves outside containment")
        relative = str(resolved.relative_to(environment_dir))
        if relative in {"Dockerfile", "README.md"} or "skills" in resolved.parts:
            continue
        if (
            not _SAFE_RELATIVE_ENVIRONMENT_PATH_PATTERN.fullmatch(relative)
            or _FORBIDDEN_NONORACLE_PATH_PATTERN.search(relative)
            or _SENSITIVE_PATH_COMPONENT_PATTERN.search(relative)
            or len(relative) > 240
        ):
            continue
        paths.append(relative)
    return sorted(set(paths))


def _safe_task_local_path(value: str) -> str | None:
    candidate = value.strip().rstrip(".,;:)]}\"'")
    if candidate == "/root":
        return candidate
    if not candidate.startswith("/root/") or len(candidate) > 300:
        return None
    if not _TASK_LOCAL_PATH_PATTERN.fullmatch(candidate):
        return None
    if any(part in {"", ".", ".."} for part in candidate.split("/")[2:]):
        return None
    if _FORBIDDEN_NONORACLE_PATH_PATTERN.search(candidate):
        return None
    if _SENSITIVE_PATH_COMPONENT_PATTERN.search(candidate):
        return None
    return candidate


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _reject_symlink_path(path: Path, *, anchor: Path, label: str) -> None:
    try:
        relative = path.relative_to(anchor)
    except ValueError as exc:
        raise PermissionError(f"{label} path is outside its anchor") from exc
    current = anchor
    for part in relative.parts:
        if part in {"", ".", ".."}:
            raise PermissionError(f"{label} path contains an unsafe component")
        current = current / part
        if current.is_symlink():
            raise PermissionError(f"{label} path symlinks are forbidden")


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
    parser.add_argument(
        "--offline-ready",
        action="store_true",
        help="Use the preregistered credential-free subset with offline verifiers.",
    )
    args = parser.parse_args()

    adapter = SkillLearnBenchAdapter(args.root)
    if args.offline_ready:
        items = adapter.offline_ready_items()
        subset_summary = adapter.offline_ready_summary()
    elif args.credential_independent:
        items = adapter.credential_independent_items()
        subset_summary = adapter.credential_independent_summary()
    else:
        items = adapter.discover()
        subset_summary = {"policy": "full_inventory_v1"}
    if args.protocol == "family_out":
        manifest = build_family_out_manifest(items, benchmark="skilllearnbench", seed=args.seed)
    else:
        manifest = build_instance_holdout_manifest(items, benchmark="skilllearnbench", seed=args.seed)
    manifest.write(args.out)
    print(
        json.dumps(
            {
                "inventory": adapter.inventory_summary(),
                "benchmark_subset": subset_summary,
                "manifest": manifest.to_dict(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
