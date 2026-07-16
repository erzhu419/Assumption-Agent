from __future__ import annotations

"""Authorized Replication-C sealed extraction and dual-oracle preparation."""

import argparse
import concurrent.futures
import json
from pathlib import Path
import shutil
from typing import Any, Callable, Mapping, Sequence

from replication_runtime.financial_semantic_v2 import oracle_pandas, oracle_streaming
from replication_runtime.financial_semantic_v2.pack import (
    build_consensus_gold,
    build_measurement_view,
    partition_items,
    payload_hash,
    read_json,
    sha256_file,
    verify_consensus_gold,
    verify_measurement_view,
    verify_public_pack,
    write_json,
)

from .sealed_access import (
    ACCESS_VERSION,
    CLAIM_FILENAME,
    COMPLETION_FILENAME,
    FAILURE_FILENAME,
    read_authorized_private_pack_v1,
)


PREPARATION_VERSION = "financial_sec13f_replication_c_sealed_preparation_v1"
SEALED_PAYLOAD_VERSION = "financial_sec13f_replication_c_sealed_payload_v1"
SEALED_PAYLOAD_FILENAME = "sealed.payload.private.json"
SEALED_GOLD_FILENAME = "sealed.gold.private.json"
PREPARATION_FILENAME = "sealed.preparation.json"


class SealedPreparationError(RuntimeError):
    """Sealed preparation failed closed after authorization."""


Oracle = Callable[..., Mapping[str, Any]]


def _access_journal_binding_v1(
    journal_root: str | Path,
    *,
    authorization_hash: str,
    private_pack_hash: str,
) -> dict[str, Any]:
    """Bind only the durable, content-free access receipts."""

    journal = Path(journal_root).expanduser().resolve(strict=True)
    if journal.is_symlink() or not journal.is_dir():
        raise SealedPreparationError("sealed access journal is unavailable")
    if {path.name for path in journal.iterdir()} != {
        CLAIM_FILENAME,
        COMPLETION_FILENAME,
    } or (journal / FAILURE_FILENAME).exists():
        raise SealedPreparationError("sealed access journal is incomplete")
    claim_path = journal / CLAIM_FILENAME
    completion_path = journal / COMPLETION_FILENAME
    claim = read_json(claim_path)
    completion = read_json(completion_path)
    claim_body = dict(claim)
    claim_hash = claim_body.pop("claim_hash", None)
    completion_body = dict(completion)
    completion_hash = completion_body.pop("completion_hash", None)
    if (
        claim_hash != payload_hash(claim_body)
        or completion_hash != payload_hash(completion_body)
        or claim.get("access_version") != ACCESS_VERSION
        or completion.get("access_version") != ACCESS_VERSION
        or claim.get("authorization_manifest_hash") != authorization_hash
        or completion.get("authorization_manifest_hash") != authorization_hash
        or completion.get("claim_hash") != claim_hash
        or claim.get("private_pack_hash") != private_pack_hash
        or completion.get("expected_private_pack_hash") != private_pack_hash
        or claim.get("access_claimed_before_path_probe") is not True
        or completion.get("access_completed") is not True
        or completion.get("raw_file_sha256_matches_precommit") is not True
        or completion.get("verified_public_pack_hash_matches_commitment") is not True
        or claim.get("private_path_persisted") is not False
        or completion.get("private_path_persisted") is not False
        or claim.get("private_content_persisted") is not False
        or completion.get("private_content_persisted") is not False
    ):
        raise SealedPreparationError("sealed access journal identity drifted")
    return {
        "access_version": ACCESS_VERSION,
        "authorization_manifest_hash": authorization_hash,
        "private_pack_hash": private_pack_hash,
        "claim_hash": claim_hash,
        "claim_file_sha256": sha256_file(claim_path),
        "completion_hash": completion_hash,
        "completion_file_sha256": sha256_file(completion_path),
        "access_claimed_before_path_probe": True,
        "access_completed": True,
        "raw_file_sha256_matches_precommit": True,
        "verified_public_pack_hash_matches_commitment": True,
        "private_path_persisted": False,
        "private_content_persisted": False,
    }


def _sealed_commitments(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "item_id": item["item_id"],
            "template": item["template"],
            "fold": item["fold"],
            "instruction_sha256": item["instruction_sha256"],
            "query_commitment_hash": payload_hash(item["query"]),
            "full_item_commitment_hash": payload_hash(item),
        }
        for item in items
    ]


def verify_sealed_payload_v1(
    value: Mapping[str, Any],
    *,
    measurement_view: Mapping[str, Any],
) -> dict[str, Any]:
    view = verify_measurement_view(measurement_view)
    payload = dict(value)
    body = dict(payload)
    declared = body.pop("sealed_payload_hash", None)
    items = payload.get("sealed_items")
    if (
        declared != payload_hash(body)
        or payload.get("sealed_payload_version") != SEALED_PAYLOAD_VERSION
        or payload.get("private_pack_hash") != view["private_pack_hash"]
        or payload.get("measurement_view_hash") != view["measurement_view_hash"]
        or payload.get("sources") != view["sources"]
        or payload.get("container_roots") != view["container_roots"]
        or not isinstance(items, list)
        or len(items) != 4
        or any(
            not isinstance(item, Mapping)
            or item.get("partition") != "sealed"
            or item.get("fold") is not None
            for item in items or ()
        )
        or _sealed_commitments(items or ()) != view["sealed_item_commitments"]
        or payload.get("item_count") != 4
        or payload.get("model_calls") != 0
        or payload.get("network_calls") != 0
    ):
        raise SealedPreparationError("sealed payload drifted")
    return payload


def prepare_sealed_partition_v1(
    *,
    private_pack_path: str | Path,
    measurement_view: Mapping[str, Any],
    authorization: Mapping[str, Any],
    journal_root: str | Path,
    previous_source: str | Path,
    current_source: str | Path,
    output_root: str | Path,
    study_id: str,
    candidate_id: str,
    pandas_oracle: Oracle = oracle_pandas.evaluate_partition,
    streaming_oracle: Oracle = oracle_streaming.evaluate_partition,
) -> dict[str, Any]:
    """Open the private pack once, then form sealed consensus gold locally."""

    view = verify_measurement_view(measurement_view)
    destination = Path(output_root).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.mkdir(parents=True, mode=0o700)
    try:
        private = read_authorized_private_pack_v1(
            unresolved_private_pack_path=private_pack_path,
            journal_root=journal_root,
            authorization=authorization,
            expected_study_id=study_id,
            expected_private_pack_hash=str(view["private_pack_hash"]),
            expected_measurement_view_hash=str(view["measurement_view_hash"]),
            expected_candidate_id=candidate_id,
        )
        pack = verify_public_pack(private)
        access_journal = _access_journal_binding_v1(
            journal_root,
            authorization_hash=str(authorization["manifest_hash"]),
            private_pack_hash=str(pack["pack_hash"]),
        )
        # This recomputation proves every public sealed commitment before an
        # oracle is called and does not expose any private field in the receipt.
        if build_measurement_view(pack) != view:
            raise SealedPreparationError("private pack differs from public commitments")
        sealed_items = list(partition_items(pack, "sealed"))

        def run(oracle: Oracle) -> dict[str, Any]:
            return dict(
                oracle(
                    pack=pack,
                    previous_source=previous_source,
                    current_source=current_source,
                    partition="sealed",
                )
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = (executor.submit(run, pandas_oracle), executor.submit(run, streaming_oracle))
            left, right = (future.result() for future in futures)
        gold = build_consensus_gold(
            pack=pack,
            left=left,
            right=right,
            partition="sealed",
        )
        verify_consensus_gold(gold, pack=pack, expected_partition="sealed")
        payload_body = {
            "sealed_payload_version": SEALED_PAYLOAD_VERSION,
            "private_pack_hash": pack["pack_hash"],
            "measurement_view_hash": view["measurement_view_hash"],
            "sources": pack["sources"],
            "container_roots": pack["container_roots"],
            "sealed_items": sealed_items,
            "item_count": 4,
            "model_calls": 0,
            "network_calls": 0,
        }
        payload = {
            **payload_body,
            "sealed_payload_hash": payload_hash(payload_body),
        }
        verify_sealed_payload_v1(payload, measurement_view=view)
        payload_path = write_json(destination / SEALED_PAYLOAD_FILENAME, payload)
        gold_path = write_json(destination / SEALED_GOLD_FILENAME, gold)
        payload_path.chmod(0o600)
        gold_path.chmod(0o600)
        receipt_body = {
            "preparation_version": PREPARATION_VERSION,
            "study_id_hash": payload_hash({"study_id": study_id}),
            "authorization_hash": authorization["manifest_hash"],
            "access_journal": access_journal,
            "private_pack_hash": pack["pack_hash"],
            "measurement_view_hash": view["measurement_view_hash"],
            "sealed_payload_hash": payload["sealed_payload_hash"],
            "sealed_gold_hash": gold["gold_hash"],
            "sealed_item_count": 4,
            "oracle_ids": gold["oracle_ids"],
            "oracle_output_hashes": gold["oracle_output_hashes"],
            "oracle_call_count": 2,
            "oracle_max_workers": 2,
            "cross_oracle_agreement": True,
            "model_calls": 0,
            "network_calls": 0,
            "online_judge_calls": 0,
            "private_path_persisted": False,
            "sealed_content_persisted_in_receipt": False,
            "gold_content_persisted_in_receipt": False,
            "candidate_imports": 0,
        }
        receipt = {
            **receipt_body,
            "preparation_hash": payload_hash(receipt_body),
        }
        write_json(destination / PREPARATION_FILENAME, receipt)
        return receipt
    except Exception:
        # Preparation roots are private, but a partial payload/gold must not be
        # mistaken for a complete authorized preparation.
        shutil.rmtree(destination, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-pack", type=Path, required=True)
    parser.add_argument("--measurement-view", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--journal-root", type=Path, required=True)
    parser.add_argument("--previous-source", type=Path, required=True)
    parser.add_argument("--current-source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--candidate-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = prepare_sealed_partition_v1(
        private_pack_path=args.private_pack,
        measurement_view=read_json(args.measurement_view),
        authorization=read_json(args.authorization),
        journal_root=args.journal_root,
        previous_source=args.previous_source,
        current_source=args.current_source,
        output_root=args.output_root,
        study_id=args.study_id,
        candidate_id=args.candidate_id,
    )
    print(json.dumps({"preparation_hash": receipt["preparation_hash"], "sealed_item_count": 4}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
