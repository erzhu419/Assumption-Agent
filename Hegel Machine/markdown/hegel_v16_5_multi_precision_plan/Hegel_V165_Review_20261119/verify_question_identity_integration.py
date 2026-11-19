#!/usr/bin/env python3
"""Reproduce the fixed-787 question identity integration control, offline.

Only Python's standard library and the exact question_semantics.py input are
needed. The script does not fetch files, import the repository as a package,
write bytecode beside the source, or modify a checkout. Its pinned hashes bind
the input bytes to the file audited at the fixed commit; this is not a claim
about the HEAD of whichever checkout supplies the file.

Example (from the review package directory):
  python verify_question_identity_integration.py \
    --source '/path/to/checkout/Hegel Machine/src/hegel_machine/cognition/question_semantics.py' \
    --output question_identity_integration_probe.json

For a previously downloaded snapshot, --source accepts that file too. Optional
--source-metadata accepts the audit snapshot's .meta.json and additionally
checks its repo, ref, path, and Git blob ID. No source file is bundled here.
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import types


REPOSITORY = "erzhu419/Assumption-Agent"
COMMIT = "787256bd415515905359626f6a5498174ad5db3c"
REPO_PATH = "Hegel Machine/src/hegel_machine/cognition/question_semantics.py"
SOURCE_URL = (
    "https://github.com/erzhu419/Assumption-Agent/blob/"
    + COMMIT + "/Hegel%20Machine/src/hegel_machine/cognition/question_semantics.py"
)
EXPECTED_SHA256 = "ea25e8664614c0c0efb98b72d42e4242867a34e8598eb36cc7e30f6c7731b0de"
EXPECTED_GIT_BLOB_SHA1 = "d40fb11e330c62851e1b63b26da3108b6b754969"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True,
                        help="Exact fixed-787 question_semantics.py, from checkout or snapshot")
    parser.add_argument("--source-metadata", type=Path,
                        help="Optional original audit .meta.json; if supplied it must match")
    parser.add_argument("--output", type=Path,
                        default=Path("question_identity_integration_probe.json"))
    args = parser.parse_args()

    raw = args.source.read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    blob_sha1 = hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()
    if sha256 != EXPECTED_SHA256 or blob_sha1 != EXPECTED_GIT_BLOB_SHA1:
        parser.error(
            "Refusing to execute a different source version. "
            f"Expected SHA256 {EXPECTED_SHA256} / Git blob {EXPECTED_GIT_BLOB_SHA1}; "
            f"received {sha256} / {blob_sha1}. Obtain the exact file from {SOURCE_URL}."
        )

    metadata_verified = False
    if args.source_metadata:
        metadata = json.loads(args.source_metadata.read_text(encoding="utf-8"))
        expected = {"repo": REPOSITORY, "ref": COMMIT,
                    "path": REPO_PATH, "sha": EXPECTED_GIT_BLOB_SHA1}
        mismatches = {key: {"expected": value, "actual": metadata.get(key)}
                      for key, value in expected.items() if metadata.get(key) != value}
        if mismatches:
            parser.error("Snapshot provenance metadata mismatch: " + json.dumps(mismatches))
        metadata_verified = True

    # Execute only the hash-verified module in memory, without creating __pycache__.
    module = types.ModuleType("fixed_787_question_semantics_control")
    module.__file__ = str(args.source.resolve())
    exec(compile(raw, str(args.source), "exec"), module.__dict__)

    question = {
        "id": "review_question", "kind": "mechanism", "answer_type": "relation",
        "answer_resolution_ref": {"kind": "fixed"},
        "query_semantics": {"mode": "model_query",
                            "query_expression": {"op": "HypothesisTest"}},
        "protocol_spec": {"reset_requirement": "cold"},
        "readout_spec": {"raw_channel": "I"},
        "scale_spec": {"coarse_graining_op": "identity"},
        "semantic_answer": {"status": "answered", "value": True},
    }
    base = module.adopt_question_semantics(None, question)
    changes = {
        "protocol_spec": {"reset_requirement": "warm"},
        "readout_spec": {"raw_channel": "flux"},
        "scale_spec": {"coarse_graining_op": "window_average"},
    }
    results = {}
    for field, replacement in changes.items():
        updated = module.adopt_question_semantics(
            deepcopy(base), {"id": question["id"], field: replacement})
        if updated.get(field) != replacement:
            raise RuntimeError(f"Control invalid: requested update was not applied for {field}")
        results[field] = {
            "meaning_version_before": base["meaning_version"],
            "meaning_version_after": updated["meaning_version"],
            "old_semantic_answer_retained": (
                updated.get("semantic_answer") == base["semantic_answer"]),
        }

    reproduced = all(item == {"meaning_version_before": 1,
                              "meaning_version_after": 1,
                              "old_semantic_answer_retained": True}
                     for item in results.values())
    report = {
        "base_commit": COMMIT,
        "scope": (
            "Controlled dictionary update of exact fixed source; no model or scientific experiment. "
            "New top-level fields are hypothetical V16.5 additions; this is an integration risk, "
            "not an allegation of current V16.5 runtime behavior."),
        "provenance": {
            "repository": REPOSITORY, "fixed_source_url": SOURCE_URL,
            "repository_path": REPO_PATH, "local_input": str(args.source.resolve()),
            "source_sha256": sha256, "git_blob_sha1": blob_sha1,
            "exact_fixed_source_content_verified": True,
            "optional_snapshot_metadata_verified": metadata_verified,
            "checkout_head_claim": "Not inspected; only the exact pinned file content is verified.",
        },
        "results": results,
        "expected_fixed_version_behavior_reproduced": reproduced,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()),
                      "source_sha256": sha256, "reproduced": reproduced}, ensure_ascii=False))
    return 0 if reproduced else 1


if __name__ == "__main__":
    sys.exit(main())
