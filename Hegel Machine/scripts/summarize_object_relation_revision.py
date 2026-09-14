#!/usr/bin/env python3
"""Keep independent apparatus and stage-level costs for object relation repair."""

from pathlib import Path
import summarize_dynamic_research_replication as study


ROOT = Path(__file__).resolve().parents[1]
study.DEFAULT = ROOT / "artifacts/object_relation_revision_20260913"
study.PAIRS = (("hegel_revised", "hegel_v079"), ("hegel_revised", "code"), ("hegel_v079", "code"))


if __name__ == "__main__":
    study.main()
