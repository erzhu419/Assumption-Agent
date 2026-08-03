from __future__ import annotations

from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = PROJECT_ROOT / "docs"
DISCLOSURE_VALUES = {
    "same_admin_controller": True,
    "organizational_independence": False,
    "independent_human_actors": False,
    "technical_role_independence": True,
    "owner_accepted_threat_model": True,
    "remote_attestation": False,
    "hardware_key_nonexportability": False,
}


def test_every_normative_doc_that_discloses_admin_control_names_all_seven_fields() -> None:
    """Prevent a shortened four-field disclosure from looking authoritative."""

    candidates: list[Path] = []
    missing_or_wrong_by_path: dict[str, list[str]] = {}
    for path in sorted(DOCS_ROOT.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        if "same_admin_controller" not in text:
            continue
        candidates.append(path)
        missing_or_wrong = [
            key
            for key, expected in DISCLOSURE_VALUES.items()
            if re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(key)}\s*[:=]\s*"
                rf"{str(expected).lower()}(?![A-Za-z0-9_])",
                text,
            )
            is None
        ]
        if missing_or_wrong:
            missing_or_wrong_by_path[path.relative_to(PROJECT_ROOT).as_posix()] = (
                missing_or_wrong
            )

    assert candidates, "no technical-actor disclosure document was found"
    assert not missing_or_wrong_by_path, (
        "incomplete or incorrect seven-field disclosures: "
        f"{missing_or_wrong_by_path}"
    )
