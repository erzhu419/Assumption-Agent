from __future__ import annotations

"""Durable candidate backend for the public SEC-13F contract operator."""

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from assumption_agent.benchmarks.financial_sec13f_contract_integration_v2 import (
    INTEGRATION_VERSION,
    FinancialSec13FContractSubprocessBackendV2,
)
from assumption_agent.benchmarks.financial_sec13f_contract_operator_v2 import (
    EXTRACTION_RECEIPT_VERSION,
    NUMERIC_ENGINE,
    OPERATION_ORDER_BY_TEMPLATE,
    OPERATOR_VERSION,
    PARSER_MODE,
    QUERY_RECEIPT_VERSION,
    payload_hash,
    sha256_file,
)
from assumption_agent.benchmarks.skilllearn_lifecycle import (
    SkillLearnTrialObservation,
    SkillLearnTrialRequest,
    TrialVariant,
)
from assumption_agent.models import stable_hash

from replication_runtime.financial_semantic_v2.backends import (
    WORK_STAGE_ORDER_V2,
    DurableRawSubprocessBackendV2,
    FinancialSemanticReplicationBackendError,
    _DurableBackendMixinV2,
    backend_runtime_identity_v2,
    future_terminal_semantics_v2,
    initialize_work_state_v2,
)
from replication_runtime.financial_semantic_v2.durable_state import (
    atomic_write_hashed_json_v2,
)


_RUNTIME_EVIDENCE_FIELDS = {
    "runtime_version",
    "request_hash",
    "candidate_id",
    "asset_manifest_hash",
    "contract_hash",
    "planner_hash",
    "backend_instance_hash",
    "plan_hash",
    "extraction_receipt_hash",
    "extraction_receipt",
    "query_receipt_hash",
    "query_receipt",
    "output_sha256",
    "answers_payload_persisted",
    "operator_source_sha256",
    "program_id",
    "treatment_hash",
    "external_skill_source_receipt_hash",
    "container_operator_readback_sha256",
    "container_asset_readback_sha256",
    "container_plan_readback_sha256",
    "executed_after_agent_exit",
    "executed_before_verifier_materialization",
    "online_calls",
    "raw_instruction_persisted",
    "raw_entity_persisted_in_durable_evidence",
    "ephemeral_plan_deleted_before_verifier",
    "gold_content_accessed",
    "pack_content_accessed",
    "evidence_hash",
}
_EXTRACTION_RECEIPT_FIELDS = {
    "receipt_version",
    "parser_mode",
    "candidate_id",
    "asset_manifest_hash",
    "contract_hash",
    "template_grammar_hash",
    "plan_hash",
    "instruction_sha256",
    "template_id",
    "question_hashes",
    "entity_hashes",
    "semantic_assignment",
    "raw_instruction_persisted",
    "raw_entity_persisted_in_receipt",
    "model_calls",
    "online_calls",
    "receipt_hash",
}
_QUERY_RECEIPT_FIELDS = {
    "receipt_version",
    "operator_version",
    "candidate_id",
    "asset_manifest_hash",
    "contract_hash",
    "operator_source_sha256",
    "plan_hash",
    "numeric_engine",
    "input_file_receipts",
    "input_set_hash",
    "pre_output_exists",
    "pre_output_sha256",
    "post_output_sha256",
    "output_changed",
    "answer_key_set_hash",
    "answers_payload_persisted_in_receipt",
    "raw_entity_persisted_in_receipt",
    "network_calls",
    "model_calls",
    "verifier_content_accessed",
    "gold_content_accessed",
    "pack_content_accessed",
    "receipt_hash",
}
_FORBIDDEN_RAW_CONTENT_KEYS = {
    "answer",
    "answers",
    "answers_payload",
    "entity",
    "expected_answer",
    "expected_output",
    "gold",
    "gold_payload",
    "instruction",
    "output_payload",
    "plan",
    "query",
    "question",
    "raw_answer",
    "raw_entity",
    "raw_instruction",
    "raw_plan",
}


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _reject_nested_raw_content(value: object) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if (
                not isinstance(key, str)
                or key.casefold() in _FORBIDDEN_RAW_CONTENT_KEYS
            ):
                raise FinancialSemanticReplicationBackendError(
                    "contract runtime evidence contains forbidden raw content"
                )
            _reject_nested_raw_content(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _reject_nested_raw_content(nested)


class DurableFinancialSec13FContractBackendV2(
    _DurableBackendMixinV2,
    FinancialSec13FContractSubprocessBackendV2,
):
    """Persist causal operator evidence before the offline verifier starts."""

    def __init__(
        self,
        *args: Any,
        durable_state_root: str | Path,
        durable_work_unit_hash: str,
        durable_request_hash: str,
        expected_precomputed_plan_hash: str,
        expected_program_set_hash: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.durable_state_root = Path(durable_state_root).resolve()
        self.durable_work_unit_hash = durable_work_unit_hash
        self.durable_request_hash = durable_request_hash
        self.durable_arm = "candidate"
        self.expected_precomputed_plan_hash = expected_precomputed_plan_hash
        self.expected_program_set_hash = expected_program_set_hash

    def _validated_runtime_evidence_v2(
        self, evidence: Mapping[str, Any], *, state: Any
    ) -> dict[str, Any]:
        """Validate the complete durable payload before writing any bytes."""

        if set(evidence) != _RUNTIME_EVIDENCE_FIELDS:
            raise FinancialSemanticReplicationBackendError(
                "contract runtime evidence schema drifted"
            )
        _reject_nested_raw_content(evidence)
        evidence_body = dict(evidence)
        evidence_hash = evidence_body.pop("evidence_hash", None)
        if not _is_sha256(evidence_hash) or stable_hash(evidence_body) != (
            evidence_hash
        ):
            raise FinancialSemanticReplicationBackendError(
                "contract runtime evidence hash drifted"
            )

        extraction = evidence.get("extraction_receipt")
        query = evidence.get("query_receipt")
        if (
            not isinstance(extraction, Mapping)
            or set(extraction) != _EXTRACTION_RECEIPT_FIELDS
            or not isinstance(query, Mapping)
            or set(query) != _QUERY_RECEIPT_FIELDS
        ):
            raise FinancialSemanticReplicationBackendError(
                "contract runtime receipt schema drifted"
            )
        extraction_body = dict(extraction)
        extraction_hash = extraction_body.pop("receipt_hash", None)
        query_body = dict(query)
        query_hash = query_body.pop("receipt_hash", None)
        if (
            not _is_sha256(extraction_hash)
            or payload_hash(extraction_body) != extraction_hash
            or not _is_sha256(query_hash)
            or payload_hash(query_body) != query_hash
        ):
            raise FinancialSemanticReplicationBackendError(
                "contract runtime nested receipt hash drifted"
            )

        asset = self.planner.asset
        template_id = extraction.get("template_id")
        expected_operations = OPERATION_ORDER_BY_TEMPLATE.get(str(template_id))
        question_hashes = extraction.get("question_hashes")
        entity_hashes = extraction.get("entity_hashes")
        if (
            expected_operations is None
            or not isinstance(question_hashes, list)
            or not isinstance(entity_hashes, list)
            or len(question_hashes) != len(expected_operations)
            or len(entity_hashes) != len(expected_operations)
            or not all(_is_sha256(value) for value in question_hashes)
            or not all(_is_sha256(value) for value in entity_hashes)
            or extraction.get("semantic_assignment")
            != list(expected_operations)
            or extraction.get("receipt_version")
            != EXTRACTION_RECEIPT_VERSION
            or extraction.get("parser_mode") != PARSER_MODE
            or extraction.get("candidate_id") != asset["candidate_id"]
            or extraction.get("asset_manifest_hash")
            != asset["manifest_hash"]
            or extraction.get("contract_hash") != asset["contract_hash"]
            or extraction.get("template_grammar_hash")
            != asset["template_grammar_hash"]
            or extraction.get("plan_hash")
            != self.expected_precomputed_plan_hash
            or not _is_sha256(extraction.get("instruction_sha256"))
            or extraction.get("raw_instruction_persisted") is not False
            or extraction.get("raw_entity_persisted_in_receipt") is not False
            or extraction.get("model_calls") != 0
            or extraction.get("online_calls") != 0
        ):
            raise FinancialSemanticReplicationBackendError(
                "contract extraction receipt identity drifted"
            )

        input_receipts = query.get("input_file_receipts")
        expected_inputs = [
            ("previous", "COVERPAGE.tsv"),
            ("previous", "INFOTABLE.tsv"),
            ("current", "COVERPAGE.tsv"),
            ("current", "INFOTABLE.tsv"),
        ]
        if not isinstance(input_receipts, list) or len(input_receipts) != len(
            expected_inputs
        ):
            raise FinancialSemanticReplicationBackendError(
                "contract input receipt set drifted"
            )
        for row, (role, table) in zip(input_receipts, expected_inputs):
            if (
                not isinstance(row, Mapping)
                or set(row) != {"role", "table", "size_bytes", "file_sha256"}
                or row.get("role") != role
                or row.get("table") != table
                or isinstance(row.get("size_bytes"), bool)
                or not isinstance(row.get("size_bytes"), int)
                or row["size_bytes"] <= 0
                or not _is_sha256(row.get("file_sha256"))
            ):
                raise FinancialSemanticReplicationBackendError(
                    "contract input receipt row drifted"
                )
        pre_output_exists = query.get("pre_output_exists")
        pre_output_sha256 = query.get("pre_output_sha256")
        if (
            not isinstance(pre_output_exists, bool)
            or (
                pre_output_exists
                and not _is_sha256(pre_output_sha256)
            )
            or (not pre_output_exists and pre_output_sha256 is not None)
            or not isinstance(query.get("output_changed"), bool)
            or query.get("receipt_version") != QUERY_RECEIPT_VERSION
            or query.get("operator_version") != OPERATOR_VERSION
            or query.get("candidate_id") != asset["candidate_id"]
            or query.get("asset_manifest_hash") != asset["manifest_hash"]
            or query.get("contract_hash") != asset["contract_hash"]
            or query.get("operator_source_sha256")
            != asset["operator_source_sha256"]
            or query.get("plan_hash") != self.expected_precomputed_plan_hash
            or query.get("numeric_engine") != NUMERIC_ENGINE
            or query.get("input_set_hash") != payload_hash(input_receipts)
            or not _is_sha256(query.get("post_output_sha256"))
            or not _is_sha256(query.get("answer_key_set_hash"))
            or query.get("answers_payload_persisted_in_receipt") is not False
            or query.get("raw_entity_persisted_in_receipt") is not False
            or query.get("network_calls") != 0
            or query.get("model_calls") != 0
            or query.get("verifier_content_accessed") is not False
            or query.get("gold_content_accessed") is not False
            or query.get("pack_content_accessed") is not False
        ):
            raise FinancialSemanticReplicationBackendError(
                "contract query receipt identity drifted"
            )

        plan_file_bytes = (
            json.dumps(
                state.plan,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        expected_values = {
            "runtime_version": INTEGRATION_VERSION,
            "request_hash": self.durable_request_hash,
            "candidate_id": asset["candidate_id"],
            "asset_manifest_hash": asset["manifest_hash"],
            "contract_hash": asset["contract_hash"],
            "planner_hash": self.planner.planner_hash,
            "backend_instance_hash": self.financial_backend_instance_hash,
            "plan_hash": self.expected_precomputed_plan_hash,
            "extraction_receipt_hash": extraction_hash,
            "query_receipt_hash": query_hash,
            "output_sha256": query["post_output_sha256"],
            "answers_payload_persisted": False,
            "operator_source_sha256": asset["operator_source_sha256"],
            "program_id": self.expected_program_id,
            "treatment_hash": self.expected_treatment_hash,
            "external_skill_source_receipt_hash": (
                self.expected_external_skill_source_receipt_hash
            ),
            "container_operator_readback_sha256": asset[
                "operator_source_sha256"
            ],
            "container_asset_readback_sha256": sha256_file(
                self.planner.asset_path
            ),
            "container_plan_readback_sha256": hashlib.sha256(
                plan_file_bytes
            ).hexdigest(),
            "executed_after_agent_exit": True,
            "executed_before_verifier_materialization": True,
            "online_calls": 0,
            "raw_instruction_persisted": False,
            "raw_entity_persisted_in_durable_evidence": False,
            "ephemeral_plan_deleted_before_verifier": True,
            "gold_content_accessed": False,
            "pack_content_accessed": False,
        }
        if (
            evidence.get("request_hash") != getattr(state, "request_hash", None)
            or any(evidence.get(key) != value for key, value in expected_values.items())
        ):
            raise FinancialSemanticReplicationBackendError(
                "contract runtime evidence identity drifted"
            )
        for field in (
            "request_hash",
            "candidate_id",
            "asset_manifest_hash",
            "contract_hash",
            "planner_hash",
            "backend_instance_hash",
            "plan_hash",
            "extraction_receipt_hash",
            "query_receipt_hash",
            "output_sha256",
            "operator_source_sha256",
            "program_id",
            "treatment_hash",
            "external_skill_source_receipt_hash",
            "container_operator_readback_sha256",
            "container_asset_readback_sha256",
            "container_plan_readback_sha256",
        ):
            if not _is_sha256(evidence.get(field)):
                raise FinancialSemanticReplicationBackendError(
                    "contract runtime evidence hash field is malformed"
                )
        return dict(evidence)

    def _execute_contract_plan_before_verifier_v2(
        self,
        *,
        delegate: Any,
        container_name: str,
    ) -> None:
        state = getattr(self._contract_local, "state", None)
        if state is None or state.plan.get("plan_hash") != (
            self.expected_precomputed_plan_hash
        ):
            raise FinancialSemanticReplicationBackendError(
                "runtime contract plan differs from the precomputed plan"
            )
        request = getattr(self, "_active_request", None)
        if not isinstance(request, SkillLearnTrialRequest):
            raise FinancialSemanticReplicationBackendError(
                "contract operator started without an active request"
            )
        chain = self._durable_chain()
        if [row.stage for row in chain] != list(WORK_STAGE_ORDER_V2[:2]):
            raise FinancialSemanticReplicationBackendError(
                "contract operator started at an unexpected durable stage"
            )
        self._transition_next(
            "agent_completed",
            self._agent_completion_payload(
                request,
                reconciled_after_backend_return=False,
            ),
        )
        super()._execute_contract_plan_before_verifier_v2(
            delegate=delegate,
            container_name=container_name,
        )
        evidence = getattr(state, "runtime_evidence", None)
        if not isinstance(evidence, Mapping):
            raise FinancialSemanticReplicationBackendError(
                "contract runtime evidence was not produced"
            )
        validated_evidence = self._validated_runtime_evidence_v2(
            evidence, state=state
        )
        evidence_receipt = atomic_write_hashed_json_v2(
            self.durable_state_root / "semantic_runtime_evidence.json",
            {
                "request_hash": self.durable_request_hash,
                "evidence": validated_evidence,
                "evidence_hash": validated_evidence["evidence_hash"],
                "persisted_before_verifier": True,
                "raw_plan_persisted": False,
                "answers_payload_persisted": False,
            },
            hash_field="receipt_hash",
        )
        self._transition_next(
            "operator_completed",
            {
                "arm": "candidate",
                "applicable": True,
                "plan_hash": self.expected_precomputed_plan_hash,
                "semantic_evidence_hash": validated_evidence[
                    "evidence_hash"
                ],
                "semantic_evidence_receipt_hash": evidence_receipt[
                    "receipt_hash"
                ],
                "operator_calls": 1,
                "online_calls": 0,
                "persisted_before_verifier": True,
                "raw_plan_persisted": False,
                "answers_payload_persisted": False,
            },
        )

    def run(
        self,
        request: SkillLearnTrialRequest,
        *,
        skill_source_dir: Path | None,
        trace_id: str,
    ) -> SkillLearnTrialObservation:
        if request.request_hash != self.durable_request_hash:
            raise FinancialSemanticReplicationBackendError(
                "candidate request no longer matches durable state"
            )
        if (
            request.variant is not TrialVariant.POLICY_ON
            or skill_source_dir is None
            or request.program_id != self.expected_program_id
            or request.program_set_hash != self.expected_program_set_hash
            or request.treatment_hash != self.expected_treatment_hash
            or request.external_skill_source_receipt_hash
            != self.expected_external_skill_source_receipt_hash
        ):
            raise FinancialSemanticReplicationBackendError(
                "contract candidate arm identity or source drifted"
            )
        self._active_request = request
        try:
            observation = super().run(
                request,
                skill_source_dir=skill_source_dir,
                trace_id=trace_id,
            )
            self._complete_after_observation(request, observation)
            return observation
        finally:
            self._active_request = None


__all__ = [
    "WORK_STAGE_ORDER_V2",
    "DurableFinancialSec13FContractBackendV2",
    "DurableRawSubprocessBackendV2",
    "FinancialSemanticReplicationBackendError",
    "backend_runtime_identity_v2",
    "future_terminal_semantics_v2",
    "initialize_work_state_v2",
]
