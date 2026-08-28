#!/usr/bin/env python3
"""Authorized, finite Q0.5b shadow retry campaign with external trace binding.

``--preflight`` is read-only and derives an exact execution capsule.  ``--run``
requires a separate immutable owner-direction record for that capsule.  Each
of the three ordered slots is one-shot and uses a distinct clone, Cargo cache,
artifact, trajectory namespace, and raw-stream namespace.  Only an exact,
fully closed ``HELD_ACTOR_GAP_EXCEEDED`` outcome can open the next slot.
"""

from __future__ import annotations

import argparse
from collections import deque
from copy import deepcopy
from dataclasses import asdict, is_dataclass
import errno
import fcntl
from hashlib import sha256
import importlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import select
import selectors
import signal
import stat
import subprocess
import sys
import threading
import time
import types
from typing import Any, Callable, Final, Mapping, NoReturn, Sequence


PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
WORKTREE_ROOT: Final = PROJECT_ROOT.parent
TOOL_PATH: Final = Path(__file__).resolve()
CONFIG_PATH: Final = (
    PROJECT_ROOT / "config/phase3_q05b_shadow_retry_campaign_v2.json"
)
CORE_PATH: Final = (
    PROJECT_ROOT
    / "src/hegel_machine/phase3_q05b_shadow_retry_campaign_v2.py"
)
CORE_TEST_PATH: Final = (
    PROJECT_ROOT / "tests/test_phase3_q05b_shadow_retry_campaign_v2_core.py"
)
TOOL_TEST_PATH: Final = (
    PROJECT_ROOT / "tests/test_phase3_q05b_shadow_retry_campaign_v2_tool.py"
)
DOC_PATH: Final = (
    PROJECT_ROOT / "docs/Hegel_Machine_Q05b_Shadow_Retry_Campaign_v2.md"
)
V1_SAFETY_KERNEL_PATH: Final = (
    PROJECT_ROOT / "src/hegel_machine/phase3_q05b_shadow_nonadoption_trial_v1.py"
)
_STATE_ROOT_V2: Final = "/home/erzhu419/.local/state/hegel-machine"
_V2R8_STATE_PREFIX: Final = f"{_STATE_ROOT_V2}/q05b-shadow-retry-campaign-v2r8"
_PRIOR_V2_STATE_PREFIX: Final = f"{_STATE_ROOT_V2}/q05b-shadow-retry-campaign-v2"
_PRIOR_V2_CONTRACT_ID: Final = "hegel-phase3a-q05b-shadow-retry-campaign-v2"
_PRIOR_V2_CAPSULE_DIGEST: Final = (
    "sha256:238b79a5ddf5adb379df9ac1b0ae7e762f7616f1cfbf19b93474315a9197ab62"
)
_PRIOR_V2_AUTHORIZATION_DIGEST: Final = (
    "sha256:0f59cc4aa2f605bdd1388baf769961ccfebf2123c6d6c9ff467f7f61add29cea"
)
_PRIOR_V2R1_STATE_PREFIX: Final = (
    f"{_STATE_ROOT_V2}/q05b-shadow-retry-campaign-v2r1"
)
_PRIOR_V2R1_WORKTREE_ROOT: Final = "/tmp/hegel-q05b-shadow-retry-v2r1-worktree"
_PRIOR_V2R1_CONTRACT_ID: Final = (
    "hegel-phase3a-q05b-shadow-retry-campaign-v2r1"
)
_PRIOR_V2R1_CONTRACT_DIGEST: Final = (
    "sha256:b18548d74ce744a5326ae42fa957b297963a3b2b187131591c5541c562ef3210"
)
_PRIOR_V2R1_CAPSULE_DIGEST: Final = (
    "sha256:62d6d7a56107439261ce66788e5d803f7d1370733673416076131e6d20a16e61"
)
_PRIOR_V2R1_AUTHORIZATION_DIGEST: Final = (
    "sha256:2a410cff6c59895f941075af5cc6035f61dea4ba4734d82eb0719c5b23a1bb90"
)
_PRIOR_V2R1_TERMINAL_LEDGER_RECORD_DIGEST: Final = (
    "sha256:b910df6dfea1e0bd8fd4bd11d8aa20d682f43898b495f8c643b668b970f8108b"
)
_PRIOR_V2R1_COMPLETION_LEDGER_RECORD_DIGEST: Final = (
    "sha256:b9e63585d7a6eb48d6195fa188ce1d40f2c79a169848f58cd10b82328be4702b"
)
_PRIOR_V2R2_STATE_PREFIX: Final = (
    f"{_STATE_ROOT_V2}/q05b-shadow-retry-campaign-v2r2"
)
_PRIOR_V2R2_WORKTREE_ROOT: Final = "/tmp/hegel-q05b-shadow-retry-v2r2-worktree"
_PRIOR_V2R2_CONTRACT_ID: Final = (
    "hegel-phase3a-q05b-shadow-retry-campaign-v2r2"
)
_PRIOR_V2R2_CONTROLLER_COMMIT: Final = (
    "33ff8ba17addf914f06eb440b8b7a40e93654816"
)
_PRIOR_V2R2_CONTROLLER_TREE: Final = (
    "bafb4291cdc003e514e6ae36800588f3e37242e7"
)
_PRIOR_V2R2_TARGET_COMMIT: Final = (
    "7bc2044212e278443b1ca703115c7cb17591ac14"
)
_PRIOR_V2R2_CAMPAIGN_SOURCE_BINDING_DIGEST: Final = (
    "sha256:f1bf112b268c5fdf1a104e9e66f92e21bc8187347657e0556ea47c3bd8815079"
)
_PRIOR_V2R2_CONTRACT_DIGEST: Final = (
    "sha256:02e297682d99e2efa7818b2b58562dd6b6dc3ef468e20b980240e2ab1e19c871"
)
_PRIOR_V2R2_CAPSULE_DIGEST: Final = (
    "sha256:325fc2d916e021d811b5ca0bce2bba4a2f89282e5529d9471670a4c8cd34648d"
)
_PRIOR_V2R2_OWNER_CONFIRMATION_DIGEST: Final = (
    "sha256:a1e8b8d688d239aec60b30094e34a470bb889290d7101a8683a9a8fd951c0f52"
)
_PRIOR_V2R2_AUTHORIZATION_DIGEST: Final = (
    "sha256:079730f604279e5da1e2aa81c22ae4a00806f5981659622fdec3da2efde836c9"
)
_PRIOR_V2R2_TERMINAL_BODY_DIGEST: Final = (
    "sha256:61d2b63a73e781351a65d49c638498ddc3a66fe8972eaa1ed214b73f0e33edde"
)
_PRIOR_V2R2_TERMINAL_LEDGER_RECORD_DIGEST: Final = (
    "sha256:bb312d9774a57723a4b59e3a31a4adfd0ad6abef6c3b12cd3bc852e9b6280f78"
)
_PRIOR_V2R2_COMPLETION_BODY_DIGEST: Final = (
    "sha256:55b8b25cc990b56bbda03930d90bda14e9329546d136812ca8714a8b2d0cef85"
)
_PRIOR_V2R2_COMPLETION_LEDGER_RECORD_DIGEST: Final = (
    "sha256:1d3c632731315906eea3ef104d4f73628ab8aeeb62af693055d69be02da03d09"
)
_PRIOR_V2R2_TERMINAL_STATUS: Final = (
    "CAMPAIGN_RETRY_EXHAUSTED_TERMINAL_NO_SLOT_04"
)
_PRIOR_V2R3_STATE_PREFIX: Final = (
    f"{_STATE_ROOT_V2}/q05b-shadow-retry-campaign-v2r3"
)
_PRIOR_V2R3_WORKTREE_ROOT: Final = "/tmp/hegel-q05b-shadow-retry-v2r3-worktree"
_PRIOR_V2R3_CONTRACT_ID: Final = (
    "hegel-phase3a-q05b-shadow-retry-campaign-v2r3"
)
_PRIOR_V2R3_CONTROLLER_COMMIT: Final = (
    "bcf22f321478ace834b2bc4c0953960812f74d45"
)
_PRIOR_V2R3_CONTROLLER_TREE: Final = (
    "ef5f42336b5b6490396491062fe6b699c65b96d5"
)
_PRIOR_V2R3_TARGET_COMMIT: Final = (
    "fdf8e22f6811fe3062b4409dad48cf9f044f2919"
)
_PRIOR_V2R3_TARGET_TREE: Final = (
    "7d3daf0088e24e223857f3f0daf789e2ce73c710"
)
_PRIOR_V2R3_CAMPAIGN_SOURCE_BINDING_DIGEST: Final = (
    "sha256:2fe4c2b80e0218dd1aaefe99cf431701ef0350477f828254c1bb6f4a7cb81de9"
)
_PRIOR_V2R3_CONTRACT_DIGEST: Final = (
    "sha256:4be13ab3045cc67ff9087ab3e24c6e6327ec1a8efc0be99e4d2befb391282828"
)
_PRIOR_V2R3_CAPSULE_DIGEST: Final = (
    "sha256:150e92c65f870df8591655fba3d6798d60bfecf52693e9b30719385e93f0cc24"
)
_PRIOR_V2R3_OWNER_CONFIRMATION_DIGEST: Final = (
    "sha256:43dc31c26c9853a5d8d507c64eb402d6e3b10d882e4df640fd75fbde16543ac7"
)
_PRIOR_V2R3_AUTHORIZATION_DIGEST: Final = (
    "sha256:7a32ebc4fe0da30df71d74698733e7734080dc15138e3429af22bf54d5f25969"
)
_PRIOR_V2R3_TERMINAL_STATUS: Final = (
    "CAMPAIGN_TRAJECTORY_UNCERTAIN_TERMINAL_NO_RESUME"
)
_PRIOR_V2R3_TERMINAL_BODY_DIGEST: Final = (
    "sha256:bd372c290bb4809479e3344da3a090a8f03a426bf04549d26174b41c72af089f"
)
_PRIOR_V2R3_TERMINAL_LEDGER_RECORD_DIGEST: Final = (
    "sha256:6b4542757feefa7d26e4c537a66eae3f5486a0d1f3e30f74c4c340f0c833646f"
)
_PRIOR_V2R3_COMPLETION_BODY_DIGEST: Final = (
    "sha256:41280c9afdf42ab951617408b97f5526233b8eea94c63f09adc0fa6a8c229c96"
)
_PRIOR_V2R3_COMPLETION_LEDGER_RECORD_DIGEST: Final = (
    "sha256:b0adfb9cc65d5422918fb498f10fac534bb6c60ea71d6611c701553628984007"
)
_PRIOR_V2R4_STATE_PREFIX: Final = (
    f"{_STATE_ROOT_V2}/q05b-shadow-retry-campaign-v2r4"
)
_PRIOR_V2R4_WORKTREE_ROOT: Final = "/tmp/hegel-q05b-shadow-retry-v2r4-worktree"
_PRIOR_V2R4_CONTRACT_ID: Final = (
    "hegel-phase3a-q05b-shadow-retry-campaign-v2r4"
)
_PRIOR_V2R4_CONTROLLER_COMMIT: Final = (
    "a2fac3ae10789eb2be34ad33f8ef9785d312eb53"
)
_PRIOR_V2R4_CONTROLLER_TREE: Final = (
    "7d9e3c36ce10d00f2458c83d8bf16d60512f007e"
)
_PRIOR_V2R4_CONTROLLER_PARENT_COMMIT: Final = (
    "abaedf57e9795515d9b7ca6b04f905d87f30d5e3"
)
_PRIOR_V2R4_TARGET_COMMIT: Final = _PRIOR_V2R4_CONTROLLER_PARENT_COMMIT
_PRIOR_V2R4_TARGET_TREE: Final = (
    "c9a88e1223faa703860d5f1cbd5fe3e83210f9c3"
)
_PRIOR_V2R4_TARGET_SOURCE_BINDING_DIGEST: Final = (
    "sha256:5a8fb98578c945e71b046304679c49f0f614b3b06a389b47c671ec4ada3887d2"
)
_PRIOR_V2R4_TARGET_CONTRACT_DIGEST: Final = (
    "sha256:2b05dcd39c1d2074f89a6e88f337a3303a33951923392b096904bbecd3b32761"
)
_PRIOR_V2R4_CAMPAIGN_SOURCE_BINDING_DIGEST: Final = (
    "sha256:76f03e0c55c54b95a9288d83d636f25bd04eb6a599bc6515e05231d92af7ce18"
)
_PRIOR_V2R4_CONTRACT_DIGEST: Final = (
    "sha256:09e2f40cceed32d38bd0e3d4a04f620593b76e8ff68f62e86ff35dd7ffa68cb8"
)
_PRIOR_V2R4_CAPSULE_DIGEST: Final = (
    "sha256:8646ad08037d681e8e619de66f5d23b15ebedccee0234108881488e758129a46"
)
_PRIOR_V2R4_OWNER_CONFIRMATION_DIGEST: Final = (
    "sha256:cd083fef0b3f10636d972e4d24ffce2eb1152bf794bc40e5afdef5e474114742"
)
_PRIOR_V2R4_AUTHORIZATION_DIGEST: Final = (
    "sha256:1bc5b4da4e67368a633ab4cb708099c0e8f09761e002fdb654397dd4e4d70e26"
)
_PRIOR_V2R4_INTENT_LEDGER_RECORD_DIGEST: Final = (
    "sha256:053a657e8e9b7ba2ec64bd8b058e811649a1b8c6afe7e7506d55c68ab14582a8"
)
_PRIOR_V2R4_AUTHORIZATION_CONSUMED_LEDGER_RECORD_DIGEST: Final = (
    "sha256:d3aa14b06f216a8e8997cc7618ca7ed2bbffa9ad8340c5d81f1182ec19816210"
)
_PRIOR_V2R4_SLOT_01_INTENT_LEDGER_RECORD_DIGEST: Final = (
    "sha256:f53277df9c4e65e6d62d4d4d58f0b78956f8c75fbddc84636d6b835892818cad"
)
_PRIOR_V2R4_TERMINAL_STATUS: Final = "CAMPAIGN_FAILED_CLOSED_TERMINAL"
_PRIOR_V2R4_TERMINAL_ERROR_CODE: Final = "FAIL_CAMPAIGN_SCHEDULING_ADMISSION"
_PRIOR_V2R4_TERMINAL_BODY_DIGEST: Final = (
    "sha256:0fdc27653795b98df13639af9d9ad255a00b287bf1931cc74d36729da693cb83"
)
_PRIOR_V2R4_TERMINAL_LEDGER_RECORD_DIGEST: Final = (
    "sha256:b64f61ef3cf57419fa902dca6b58cf1bb49ad45b87137f8264099640305341bf"
)
_PRIOR_V2R4_COMPLETION_BODY_DIGEST: Final = (
    "sha256:805bb7678abeb0640e23473ffe6c6aeda2123a6362dac86ef44c6e3b40474094"
)
_PRIOR_V2R4_COMPLETION_LEDGER_RECORD_DIGEST: Final = (
    "sha256:7d3a30a7fc8749e9a05a06f8c47ee01bcc06223925f20be50f49e3874ab8c74c"
)
_PRIOR_V2R4_PROTECTED_POLICY_FACTS: Final = {
    "prior_v2r4_campaign_contract_id": _PRIOR_V2R4_CONTRACT_ID,
    "prior_v2r4_controller_commit": _PRIOR_V2R4_CONTROLLER_COMMIT,
    "prior_v2r4_controller_tree": _PRIOR_V2R4_CONTROLLER_TREE,
    "prior_v2r4_controller_parent_commit": _PRIOR_V2R4_CONTROLLER_PARENT_COMMIT,
    "prior_v2r4_target_commit": _PRIOR_V2R4_TARGET_COMMIT,
    "prior_v2r4_target_tree": _PRIOR_V2R4_TARGET_TREE,
    "prior_v2r4_target_source_binding_digest": (
        _PRIOR_V2R4_TARGET_SOURCE_BINDING_DIGEST
    ),
    "prior_v2r4_target_contract_digest": _PRIOR_V2R4_TARGET_CONTRACT_DIGEST,
    "prior_v2r4_campaign_source_binding_digest": (
        _PRIOR_V2R4_CAMPAIGN_SOURCE_BINDING_DIGEST
    ),
    "prior_v2r4_contract_digest": _PRIOR_V2R4_CONTRACT_DIGEST,
    "prior_v2r4_capsule_digest": _PRIOR_V2R4_CAPSULE_DIGEST,
    "prior_v2r4_owner_confirmation_digest": (
        _PRIOR_V2R4_OWNER_CONFIRMATION_DIGEST
    ),
    "prior_v2r4_authorization_digest": _PRIOR_V2R4_AUTHORIZATION_DIGEST,
    "prior_v2r4_intent_ledger_record_digest": (
        _PRIOR_V2R4_INTENT_LEDGER_RECORD_DIGEST
    ),
    "prior_v2r4_authorization_consumed_ledger_record_digest": (
        _PRIOR_V2R4_AUTHORIZATION_CONSUMED_LEDGER_RECORD_DIGEST
    ),
    "prior_v2r4_slot_01_intent_ledger_record_digest": (
        _PRIOR_V2R4_SLOT_01_INTENT_LEDGER_RECORD_DIGEST
    ),
    "prior_v2r4_terminal_status": _PRIOR_V2R4_TERMINAL_STATUS,
    "prior_v2r4_terminal_error_code": _PRIOR_V2R4_TERMINAL_ERROR_CODE,
    "prior_v2r4_terminal_body_digest": _PRIOR_V2R4_TERMINAL_BODY_DIGEST,
    "prior_v2r4_terminal_ledger_record_digest": (
        _PRIOR_V2R4_TERMINAL_LEDGER_RECORD_DIGEST
    ),
    "prior_v2r4_completion_body_digest": _PRIOR_V2R4_COMPLETION_BODY_DIGEST,
    "prior_v2r4_completion_ledger_record_digest": (
        _PRIOR_V2R4_COMPLETION_LEDGER_RECORD_DIGEST
    ),
    "prior_v2r4_ledger_record_count": 5,
    "prior_v2r4_authorization_consumed_once": True,
    "prior_v2r4_supervisor_invocation_count": 0,
    "prior_v2r4_last_slot_id_or_null": None,
    "prior_v2r4_last_slot_outcome_digest_or_null": None,
    "prior_v2r4_slot_01_intent_staged": True,
    "prior_v2r4_slot_01_consumed": False,
    "prior_v2r4_slot_01_supervisor_invoked": False,
    "prior_v2r4_slot_01_trace_created": False,
    "prior_v2r4_slot_01_artifact_created": False,
    "prior_v2r4_slot_02_invoked": False,
    "prior_v2r4_slot_03_invoked": False,
    "prior_v2r4_scheduling_admission_window_count": 3,
    "prior_v2r4_scheduling_admission_selected_window_ordinal_or_null": None,
    "prior_v2r4_scheduling_admission_timed_records_durable_by_window": [
        570, 405, 490,
    ],
    "prior_v2r4_scheduling_admission_retained_records_by_window": [
        5_981, 5_816, 5_901,
    ],
    "prior_v2r4_scheduling_admission_cleanup_removed_records_by_window": [
        5_981, 5_816, 5_901,
    ],
    "prior_v2r4_scheduling_admission_cleanup_absence_by_window": [
        True, True, True,
    ],
    "prior_v2r4_terminal_is_historical_nonresumable_evidence_only": True,
    "prior_v2r4_campaign_root_open_allowed": False,
    "prior_v2r4_campaign_root_access_allowed": False,
    "prior_v2r4_capsule_authorization_or_ledger_reuse_allowed": False,
}
_PRIOR_V2R5_STATE_PREFIX: Final = (
    f"{_STATE_ROOT_V2}/q05b-shadow-retry-campaign-v2r5"
)
_PRIOR_V2R5_WORKTREE_ROOT: Final = "/tmp/hegel-q05b-shadow-retry-v2r5-worktree"
_PRIOR_V2R5_LEDGER_RECORD_NAMES_IN_SEQUENCE_ORDER: Final = (
    "campaign.intent.json",
    "campaign.authorization-consumed.json",
    "slot-01.intent.json",
    "slot-01.consumed.json",
    "slot-01.trajectory-binding.json",
    "slot-01.outcome.json",
    "slot-02.intent.json",
    "slot-02.consumed.json",
    "slot-02.trajectory-binding.json",
    "slot-02.outcome.json",
    "campaign.terminal.json",
    "campaign.completion.json",
)
_PRIOR_V2R5_LEDGER_RECORD_DIGESTS_IN_SEQUENCE_ORDER: Final = (
    "sha256:0743479056deb481b8e21b8ce2d762649bb8da2c936e76c4af3846f0aa471a23",
    "sha256:63c244676d3299fcc7734cff7fac778ac2f2fdfbacd03ef54bcde79a40604205",
    "sha256:3902326411865fc5d1605253609c48f252b412dcf70379c8d3bd1b68fed9bc86",
    "sha256:662fe117e2df380b9d6aa99cde270c92288702a14b199465916637ffc44a300a",
    "sha256:f29a472141104a7412dc6eed4f7e6ba9f8a63677191cb849e2d97eca7140e3e5",
    "sha256:7ce8823768e904e6c21ca9700e9b08256ed538d2868b54196ab231218f8220c0",
    "sha256:c70532147bea7851bb074a1074b4434704942d89b6e85de61b66fe9ac2816897",
    "sha256:0fbd04a991026bdd883e852a28dc7321263e3d195fc8052860add353e3953e8b",
    "sha256:ac24254fe6d6b145f8e6a0695490ec7fff724c82fb8b738cea4e841a1176972f",
    "sha256:edf59bc70c98d6ad2fd1ddc540e478a9aa4bc2134c71a2fec1312c3ff0ee3683",
    "sha256:e3b5ab05808234a5d56acd9c09240dc84b4b4ba29d4b293e43286ab6d7732797",
    "sha256:a2ddd0e6493ed8023143052df672fa7425c39f2d5a4f0ba59df7d59bc9fd1f42",
)
_PRIOR_V2R5_PROTECTED_POLICY_FACTS: Final = {
    "prior_v2r5_campaign_contract_id": (
        "hegel-phase3a-q05b-shadow-retry-campaign-v2r5"
    ),
    "prior_v2r5_controller_commit": (
        "8c1018518bb885ba4821ccaf10f2a3fb17f86a35"
    ),
    "prior_v2r5_controller_tree": (
        "968818406af8d3ca5d9d507c69f0ef32ff8159a2"
    ),
    "prior_v2r5_controller_parent_commit": (
        "abaedf57e9795515d9b7ca6b04f905d87f30d5e3"
    ),
    "prior_v2r5_target_commit": "abaedf57e9795515d9b7ca6b04f905d87f30d5e3",
    "prior_v2r5_target_tree": "c9a88e1223faa703860d5f1cbd5fe3e83210f9c3",
    "prior_v2r5_target_source_binding_digest": (
        "sha256:5a8fb98578c945e71b046304679c49f0f614b3b06a389b47c671ec4ada3887d2"
    ),
    "prior_v2r5_target_contract_digest": (
        "sha256:2b05dcd39c1d2074f89a6e88f337a3303a33951923392b096904bbecd3b32761"
    ),
    "prior_v2r5_campaign_source_binding_digest": (
        "sha256:c290da621e9016d79ea33398d55ae0687dbf4b4e3b4179f70727e22203c3a65a"
    ),
    "prior_v2r5_contract_digest": (
        "sha256:eba17a7bf118657259aa0279fd8b62263db19b55755e20b08bb078625c4312c9"
    ),
    "prior_v2r5_capsule_digest": (
        "sha256:ba1d195b17fea2cc461a50db50bc08fd45f63d2858ba608257a733ad3846d74d"
    ),
    "prior_v2r5_owner_confirmation_digest": (
        "sha256:a7cc4cf63f7ccca3f5d75eb5f90c80d9ac37dd1b33e12335081f6d2780baefc9"
    ),
    "prior_v2r5_authorization_digest": (
        "sha256:5b0355f87aec83834599360ee2c35a3581f61cabe547caa21e9a92256e8402bf"
    ),
    "prior_v2r5_ledger_record_names_in_sequence_order": list(
        _PRIOR_V2R5_LEDGER_RECORD_NAMES_IN_SEQUENCE_ORDER
    ),
    "prior_v2r5_ledger_record_digests_in_sequence_order": list(
        _PRIOR_V2R5_LEDGER_RECORD_DIGESTS_IN_SEQUENCE_ORDER
    ),
    "prior_v2r5_ledger_record_count": 12,
    "prior_v2r5_authorization_consumed_once": True,
    "prior_v2r5_supervisor_invocation_count": 2,
    "prior_v2r5_last_slot_id": "SLOT_02",
    "prior_v2r5_last_slot_outcome_digest": (
        "sha256:1a7d985561b6742b7eeb0fb89388774e2d18c07385e1594f78bbf4d99523ed1e"
    ),
    "prior_v2r5_slot_01_decision": "RETRY_EXACT_GAP",
    "prior_v2r5_slot_01_outcome_digest": (
        "sha256:859522773dc644391c9acd534a15f193dce63994baefbc8e65fb8fbe7d510659"
    ),
    "prior_v2r5_slot_01_trajectory_binding_digest": (
        "sha256:06831072700cca655d0aab4b26c68f1cf3ef41cc8c6dcadad7a7a3858174632a"
    ),
    "prior_v2r5_slot_01_observation_digest": (
        "sha256:d1107f1b064d48e0f50ae3fb922ed1d15f7dc4e15701bb1c8184d547426eb3f1"
    ),
    "prior_v2r5_slot_01_partition_certificate_digest": (
        "sha256:bb08fa35f16ea17022b8fd53fa2de73fd5d6a0697e2aaaef18e426d10d547b04"
    ),
    "prior_v2r5_slot_01_trajectory_status": "COMPLETE_FAIL",
    "prior_v2r5_slot_01_trajectory_event_count": 6_338,
    "prior_v2r5_slot_01_failure_rows_retained": 5,
    "prior_v2r5_slot_01_cleanup_rows_retained": 6,
    "prior_v2r5_slot_01_unpartitioned_event_count": 0,
    "prior_v2r5_slot_01_terminal_error_code": "HELD_ACTOR_GAP_EXCEEDED",
    "prior_v2r5_slot_01_gap_origin_actor": "RUST_ENDPOINT",
    "prior_v2r5_slot_01_gap_origin_role_id": 2,
    "prior_v2r5_slot_01_gap_origin_sample_ordinal": 1_578,
    "prior_v2r5_slot_01_gap_ns": 290_330_097,
    "prior_v2r5_slot_01_maximum_gap_ns": 250_000_000,
    "prior_v2r5_slot_01_target_wait_ns": 125_000_000,
    "prior_v2r5_slot_01_previous_duration_ns": 301_367_353,
    "prior_v2r5_slot_02_decision": "TERMINAL_NONRETRYABLE",
    "prior_v2r5_slot_02_outcome_digest": (
        "sha256:1a7d985561b6742b7eeb0fb89388774e2d18c07385e1594f78bbf4d99523ed1e"
    ),
    "prior_v2r5_slot_02_trajectory_binding_digest": (
        "sha256:61c6840ab0ace4717fe2ce862687bf144c85feaab545abd0e3cb73cbec64a3e7"
    ),
    "prior_v2r5_slot_02_observation_digest": (
        "sha256:b0bad116471ebd4b8d404948bb5ea57b7cb4e51282621c283fc360370671f0ec"
    ),
    "prior_v2r5_slot_02_partition_certificate_digest": (
        "sha256:69351d9d9d1a6e3720be578ddc2e3bce14a3de5c4a43829519459e1cadcdacf7"
    ),
    "prior_v2r5_slot_02_trajectory_status": "COMPLETE_FAIL",
    "prior_v2r5_slot_02_trajectory_event_count": 13_602,
    "prior_v2r5_slot_02_failure_rows_retained": 2,
    "prior_v2r5_slot_02_cleanup_rows_retained": 4,
    "prior_v2r5_slot_02_unpartitioned_event_count": 0,
    "prior_v2r5_slot_02_terminal_error_code": "FAIL_Q05B_DUAL_ARTIFACT",
    "prior_v2r5_slot_02_terminal_error_detail": (
        "candidate evidence strict replay failed: No module named 'tomli'"
    ),
    "prior_v2r5_slot_02_terminal_origin_actor_or_null": None,
    "prior_v2r5_slot_02_terminal_stage": (
        "STAGE_08_BUILD_AND_STRICTLY_DECODE_NINETEEN_ROW_CANDIDATE"
    ),
    "prior_v2r5_slot_03_invoked": False,
    "prior_v2r5_each_invoked_slot_scheduling_admission_passed": True,
    "prior_v2r5_each_invoked_slot_selected_window_ordinal": 1,
    "prior_v2r5_each_invoked_slot_window_count": 1,
    "prior_v2r5_each_invoked_slot_burst_count": 16,
    "prior_v2r5_each_invoked_slot_timed_records_durable_by_deadline": 4_800,
    "prior_v2r5_each_invoked_slot_retained_records": 10_211,
    "prior_v2r5_each_invoked_slot_cleanup_removed_records": 10_211,
    "prior_v2r5_each_invoked_slot_cleanup_absence": True,
    "prior_v2r5_each_invoked_slot_artifact_state": "ABSENT",
    "prior_v2r5_each_invoked_slot_stderr_empty": True,
    "prior_v2r5_terminal_status": "CAMPAIGN_FAILED_CLOSED_TERMINAL",
    "prior_v2r5_terminal_body_digest": (
        "sha256:df82eca0ad58a7b39687792a7cdb56bd0f8c8cc75b68a03c5294d112844e6f90"
    ),
    "prior_v2r5_terminal_ledger_record_digest": (
        "sha256:e3b5ab05808234a5d56acd9c09240dc84b4b4ba29d4b293e43286ab6d7732797"
    ),
    "prior_v2r5_terminal_error_code_or_null": None,
    "prior_v2r5_completion_body_digest": (
        "sha256:776fa05886220708165fb1c81f649c82f8e123035ab4896f7245eab7f19363aa"
    ),
    "prior_v2r5_completion_ledger_record_digest": (
        "sha256:a2ddd0e6493ed8023143052df672fa7425c39f2d5a4f0ba59df7d59bc9fd1f42"
    ),
    "prior_v2r5_completion_is_authoritative_final_commit": True,
    "prior_v2r5_no_later_campaign_write_authorized": True,
    "prior_v2r5_terminal_is_historical_nonresumable_evidence_only": True,
    "prior_v2r5_campaign_root_open_allowed": False,
    "prior_v2r5_campaign_root_access_allowed": False,
    "prior_v2r5_capsule_authorization_or_ledger_reuse_allowed": False,
}
_PRIOR_V2R6_STATE_PREFIX: Final = (
    f"{_STATE_ROOT_V2}/q05b-shadow-retry-campaign-v2r6"
)
_PRIOR_V2R6_WORKTREE_ROOT: Final = "/tmp/hegel-q05b-shadow-retry-v2r6-worktree"
_PRIOR_V2R6_RUNTIME_ROOTS: Final = (
    f"{_PRIOR_V2R6_STATE_PREFIX}-authorization",
    f"{_PRIOR_V2R6_STATE_PREFIX}-ledger",
    *tuple(
        f"{_PRIOR_V2R6_STATE_PREFIX}-slot-{ordinal:02d}-{suffix}"
        for ordinal in range(1, 4)
        for suffix in ("clone", "cargo-cache", "artifact", "evidence", "trace")
    ),
)
_PRIOR_V2R6_PROTECTED_POLICY_FACTS: Final = {
    "prior_v2r6_campaign_contract_id": (
        "hegel-phase3a-q05b-shadow-retry-campaign-v2r6"
    ),
    "prior_v2r6_controller_commit": (
        "d0701ad308fca2d27818cbd6cab861dd47c8cdcd"
    ),
    "prior_v2r6_controller_tree": (
        "2bef012c37fc86c88e674697b2fb6e6a0fd4d97a"
    ),
    "prior_v2r6_controller_parent_commit": (
        "5d85bf858e80a22633f96de41259344c70e25519"
    ),
    "prior_v2r6_target_commit": "5d85bf858e80a22633f96de41259344c70e25519",
    "prior_v2r6_target_tree": "aca3fc65be392c51c829544bfda8ae9149fc566b",
    "prior_v2r6_bound_target_source_binding_digest": (
        "sha256:5dde4a55053e2e6ffc26f3c6b485bd1c92b001b68046431f878793d90e3c27c6"
    ),
    "prior_v2r6_observed_target_source_binding_digest": (
        "sha256:8b24967b9fca37cb0f7f741b6ebc356dbeded597b567cca1ddd11c3ce61fe61a"
    ),
    "prior_v2r6_target_contract_digest": (
        "sha256:bb89ee62447ca72b5e45419cb0a63f3ca45659789dc4abbac2ce21d500e91e5f"
    ),
    "prior_v2r6_campaign_source_binding_digest": (
        "sha256:883ae7ec454833bc3e6c1112422e88f0ccf428159309681b6779d14c00290ab8"
    ),
    "prior_v2r6_contract_digest": (
        "sha256:47d7a676d836bc1c2c44ffb79f7b3de4c0ad658d95df3ca800ae750fff0be130"
    ),
    "prior_v2r6_preflight_cli_invocation_count": 1,
    "prior_v2r6_preflight_cli_status": "CAMPAIGN_COMMAND_FAILED_CLOSED",
    "prior_v2r6_preflight_error_code": "FAIL_CAMPAIGN_SOURCE",
    "prior_v2r6_preflight_error_detail": "target derived binding differs",
    "prior_v2r6_preflight_exit_code": 1,
    "prior_v2r6_preflight_stdout_canonical_single_lf": True,
    "prior_v2r6_preflight_stderr_empty": True,
    "prior_v2r6_preflight_resumable": False,
    "prior_v2r6_preflight_error_receipt_or_null": None,
    "prior_v2r6_preflight_cli_supervisor_invocation_count_or_null": None,
    "prior_v2r6_preflight_error_raw_byte_count": 290,
    "prior_v2r6_preflight_error_raw_sha256": (
        "8119c23e0df62403f01b3eddb5508664f0d49410e41f72b3196300c4c21fa1bb"
    ),
    "prior_v2r6_preflight_stderr_raw_sha256": (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ),
    "prior_v2r6_target_source_preimage_canonical_without_lf_byte_count": 2_029,
    "prior_v2r6_preflight_last_completed_phase": (
        "TARGET_SOURCE_PREIMAGE_DERIVED"
    ),
    "prior_v2r6_preflight_failure_phase": (
        "TARGET_SOURCE_BINDING_DIGEST_COMPARE"
    ),
    "prior_v2r6_preflight_first_unreached_phase": (
        "TARGET_DOCKER_LABEL_BINDING"
    ),
    "prior_v2r6_preflight_freshness_target_count": 16,
    "prior_v2r6_preflight_freshness_targets_absent": True,
    "prior_v2r6_preflight_failure_before_capsule_derivation": True,
    "prior_v2r6_dependency_preflight_invocation_count": 0,
    "prior_v2r6_capsule_digest_or_null": None,
    "prior_v2r6_owner_confirmation_digest_or_null": None,
    "prior_v2r6_authorization_digest_or_null": None,
    "prior_v2r6_authorization_lookup_or_publication_count": 0,
    "prior_v2r6_state_write_or_consumption_count": 0,
    "prior_v2r6_supervisor_invocation_count": 0,
    "prior_v2r6_campaign_execution_started": False,
    "prior_v2r6_runtime_roots_in_exact_order": list(
        _PRIOR_V2R6_RUNTIME_ROOTS
    ),
    "prior_v2r6_runtime_root_count": 17,
    "prior_v2r6_all_runtime_roots_absent": True,
    "prior_v2r6_external_before_runtime_roots_absent": True,
    "prior_v2r6_external_after_runtime_roots_absent": True,
    "prior_v2r6_worktree_clean": True,
    "prior_v2r6_failure_is_historical_nonresumable_evidence_only": True,
    "prior_v2r6_campaign_root_open_allowed": False,
    "prior_v2r6_campaign_root_access_allowed": False,
    "prior_v2r6_capsule_authorization_or_ledger_reuse_allowed": False,
}
_PRIOR_V2R7_STATE_PREFIX: Final = (
    f"{_STATE_ROOT_V2}/q05b-shadow-retry-campaign-v2r7"
)
_PRIOR_V2R7_WORKTREE_ROOT: Final = "/tmp/hegel-q05b-shadow-retry-v2r7-worktree"
_PRIOR_V2R7_LEDGER_RECORD_NAMES_IN_SEQUENCE_ORDER: Final = (
    "campaign.intent.json",
    "campaign.authorization-consumed.json",
    "slot-01.intent.json",
    "campaign.terminal.json",
    "campaign.completion.json",
)
_PRIOR_V2R7_LEDGER_RECORD_DIGESTS_IN_SEQUENCE_ORDER: Final = (
    "sha256:9cac7e1a6ec7e222d54c658282790e44f1fcc08b8e4f6be67463a3aee13b4f8e",
    "sha256:9f738ff92376f8fb71e4d66f80c31f0fc5f4e92d7a2db8047e5cffaac286204d",
    "sha256:ae2d395311e5f5a7f31d9c127985dfaff6eb62a23bed04eb8ee2e00fead2be8b",
    "sha256:27f492d3682edefd592688f9cf2b06f4d9d42cb3503ca364e215c7c80a05d2fa",
    "sha256:423a52720a358320db155d9496def9be3dc9a6008a7b1be63090956532bd190a",
)
_PRIOR_V2R7_PROTECTED_POLICY_FACTS: Final = {
    "prior_v2r7_campaign_contract_id": (
        "hegel-phase3a-q05b-shadow-retry-campaign-v2r7"
    ),
    "prior_v2r7_controller_commit": (
        "18bbcc62b7ed74cc4c479494cda0a3323321f628"
    ),
    "prior_v2r7_controller_tree": (
        "dc01db925c911c29f8d64f6bdd82edb0c6863638"
    ),
    "prior_v2r7_controller_parent_commit": (
        "5d85bf858e80a22633f96de41259344c70e25519"
    ),
    "prior_v2r7_target_commit": "5d85bf858e80a22633f96de41259344c70e25519",
    "prior_v2r7_target_tree": "aca3fc65be392c51c829544bfda8ae9149fc566b",
    "prior_v2r7_target_source_binding_digest": (
        "sha256:8b24967b9fca37cb0f7f741b6ebc356dbeded597b567cca1ddd11c3ce61fe61a"
    ),
    "prior_v2r7_target_contract_digest": (
        "sha256:bb89ee62447ca72b5e45419cb0a63f3ca45659789dc4abbac2ce21d500e91e5f"
    ),
    "prior_v2r7_campaign_source_binding_digest": (
        "sha256:78b2a44f0c511446d128a8d0bc2dbd8d7531afda081ac498a6a00cc7d02b7549"
    ),
    "prior_v2r7_contract_digest": (
        "sha256:7999c4a70812f45552c638ca01ad6b542858f5501bc42b717c71bbc47aebdbef"
    ),
    "prior_v2r7_capsule_digest": (
        "sha256:568a76b249fc9f37c16e296e8b1e777bf80b83f7d7feccbdf1579837f7437b22"
    ),
    "prior_v2r7_owner_confirmation_digest": (
        "sha256:8719cf0a11c2da930f1da7d345ba99b261304f89ae8b5aeba8deb488183d3e58"
    ),
    "prior_v2r7_python310_dependency_preflight_receipt_digest": (
        "sha256:d94e836471d6b4da87324197e9958a075e3ed79b74d0dbaf79d375402c34d221"
    ),
    "prior_v2r7_authorization_digest": (
        "sha256:7f4b09c4e3f7f3f441b4e1572f3e08cf9bbe5451906adc069f8d0b66c4e44663"
    ),
    "prior_v2r7_ledger_record_names_in_sequence_order": list(
        _PRIOR_V2R7_LEDGER_RECORD_NAMES_IN_SEQUENCE_ORDER
    ),
    "prior_v2r7_ledger_record_digests_in_sequence_order": list(
        _PRIOR_V2R7_LEDGER_RECORD_DIGESTS_IN_SEQUENCE_ORDER
    ),
    "prior_v2r7_ledger_record_count": 5,
    "prior_v2r7_authorization_consumed_once": True,
    "prior_v2r7_supervisor_invocation_count": 0,
    "prior_v2r7_last_slot_id_or_null": None,
    "prior_v2r7_last_slot_outcome_digest_or_null": None,
    "prior_v2r7_slot_01_intent_staged": True,
    "prior_v2r7_slot_01_consumed": False,
    "prior_v2r7_slot_01_supervisor_invoked": False,
    "prior_v2r7_slot_01_trace_created": False,
    "prior_v2r7_slot_01_artifact_created": False,
    "prior_v2r7_slot_02_invoked": False,
    "prior_v2r7_slot_03_invoked": False,
    "prior_v2r7_scheduling_admission_window_count": 3,
    "prior_v2r7_scheduling_admission_selected_window_ordinal_or_null": None,
    "prior_v2r7_scheduling_admission_burst_counts_by_window": [16, 16, 16],
    "prior_v2r7_scheduling_admission_maximum_anchor_ns_by_window": [
        1_097_945, 898_337, 863_776,
    ],
    "prior_v2r7_scheduling_admission_maximum_projected_gap_ns_by_window": [
        126_097_945, 125_898_337, 125_863_776,
    ],
    "prior_v2r7_scheduling_admission_maximum_burst_drain_ns_by_window": [
        208_786_870, 423_538_280, 217_788_690,
    ],
    "prior_v2r7_scheduling_admission_maximum_burst_queue_by_window": [6, 7, 6],
    "prior_v2r7_sustained_timed_records_enqueued_by_window": [
        3_010, 4_800, 2_455,
    ],
    "prior_v2r7_sustained_timed_records_durable_by_deadline_by_window": [
        3_010, 4_776, 2_455,
    ],
    "prior_v2r7_sustained_timed_records_durable_after_deadline_by_window": [
        0, 24, 0,
    ],
    "prior_v2r7_sustained_retained_records_by_window": [
        8_421, 10_211, 7_866,
    ],
    "prior_v2r7_sustained_queue_high_watermark_by_window": [64, 34, 64],
    "prior_v2r7_sustained_dropped_records_by_window": [1_790, 0, 2_345],
    "prior_v2r7_sustained_duplicate_records_by_window": [0, 0, 0],
    "prior_v2r7_sustained_reordered_records_by_window": [0, 0, 0],
    "prior_v2r7_sustained_terminal_drain_ns_by_window": [
        0, 776_134_344, 0,
    ],
    "prior_v2r7_sustained_cleanup_removed_records_by_window": [
        8_421, 10_211, 7_866,
    ],
    "prior_v2r7_sustained_cleanup_absence_by_window": [True, True, True],
    "prior_v2r7_scheduling_admission_passed_by_window": [False, False, False],
    "prior_v2r7_terminal_status": "CAMPAIGN_FAILED_CLOSED_TERMINAL",
    "prior_v2r7_terminal_error_code": "FAIL_CAMPAIGN_SCHEDULING_ADMISSION",
    "prior_v2r7_terminal_body_digest": (
        "sha256:f5bd4b123af1cf3d4bb522cb5b644b3b2e81608d4fd53f5237b262dfc803fce4"
    ),
    "prior_v2r7_terminal_ledger_record_digest": (
        "sha256:27f492d3682edefd592688f9cf2b06f4d9d42cb3503ca364e215c7c80a05d2fa"
    ),
    "prior_v2r7_completion_body_digest": (
        "sha256:df1fba1e98e5d2ddaf8fc247e5e1c584bfdddbf68a293c09d4ecff332e59ac6f"
    ),
    "prior_v2r7_completion_ledger_record_digest": (
        "sha256:423a52720a358320db155d9496def9be3dc9a6008a7b1be63090956532bd190a"
    ),
    "prior_v2r7_completion_is_authoritative_final_commit": True,
    "prior_v2r7_no_later_campaign_write_authorized": True,
    "prior_v2r7_terminal_is_historical_nonresumable_evidence_only": True,
    "prior_v2r7_campaign_root_open_allowed": False,
    "prior_v2r7_campaign_root_access_allowed": False,
    "prior_v2r7_capsule_authorization_or_ledger_reuse_allowed": False,
}
_CLONE_TARGET_REF_V2R8: Final = (
    "refs/hegel/q05b-shadow-retry-v2r8-target"
)
_PRIOR_V2_PROTECTED_ROOTS: Final = (
    f"{_PRIOR_V2_STATE_PREFIX}-authorization",
    f"{_PRIOR_V2_STATE_PREFIX}-ledger",
    *tuple(
        f"{_PRIOR_V2_STATE_PREFIX}-slot-{ordinal:02d}-{suffix}"
        for ordinal in range(1, 4)
        for suffix in ("clone", "cargo-cache", "artifact", "evidence", "trace")
    ),
)
_PRIOR_V2R1_PROTECTED_ROOTS: Final = (
    _PRIOR_V2R1_WORKTREE_ROOT,
    f"{_PRIOR_V2R1_STATE_PREFIX}-authorization",
    f"{_PRIOR_V2R1_STATE_PREFIX}-ledger",
    *tuple(
        f"{_PRIOR_V2R1_STATE_PREFIX}-slot-{ordinal:02d}-{suffix}"
        for ordinal in range(1, 4)
        for suffix in ("clone", "cargo-cache", "artifact", "evidence", "trace")
    ),
)
_PRIOR_V2R2_PROTECTED_ROOTS: Final = (
    _PRIOR_V2R2_WORKTREE_ROOT,
    f"{_PRIOR_V2R2_STATE_PREFIX}-authorization",
    f"{_PRIOR_V2R2_STATE_PREFIX}-ledger",
    *tuple(
        f"{_PRIOR_V2R2_STATE_PREFIX}-slot-{ordinal:02d}-{suffix}"
        for ordinal in range(1, 4)
        for suffix in ("clone", "cargo-cache", "artifact", "evidence", "trace")
    ),
)
_PRIOR_V2R3_PROTECTED_ROOTS: Final = (
    _PRIOR_V2R3_WORKTREE_ROOT,
    f"{_PRIOR_V2R3_STATE_PREFIX}-authorization",
    f"{_PRIOR_V2R3_STATE_PREFIX}-ledger",
    *tuple(
        f"{_PRIOR_V2R3_STATE_PREFIX}-slot-{ordinal:02d}-{suffix}"
        for ordinal in range(1, 4)
        for suffix in ("clone", "cargo-cache", "artifact", "evidence", "trace")
    ),
)
_PRIOR_V2R4_PROTECTED_ROOTS: Final = (
    _PRIOR_V2R4_WORKTREE_ROOT,
    f"{_PRIOR_V2R4_STATE_PREFIX}-authorization",
    f"{_PRIOR_V2R4_STATE_PREFIX}-ledger",
    *tuple(
        f"{_PRIOR_V2R4_STATE_PREFIX}-slot-{ordinal:02d}-{suffix}"
        for ordinal in range(1, 4)
        for suffix in ("clone", "cargo-cache", "artifact", "evidence", "trace")
    ),
)
_PRIOR_V2R5_PROTECTED_ROOTS: Final = (
    _PRIOR_V2R5_WORKTREE_ROOT,
    f"{_PRIOR_V2R5_STATE_PREFIX}-authorization",
    f"{_PRIOR_V2R5_STATE_PREFIX}-ledger",
    *tuple(
        f"{_PRIOR_V2R5_STATE_PREFIX}-slot-{ordinal:02d}-{suffix}"
        for ordinal in range(1, 4)
        for suffix in ("clone", "cargo-cache", "artifact", "evidence", "trace")
    ),
)
_PRIOR_V2R6_PROTECTED_ROOTS: Final = (
    _PRIOR_V2R6_WORKTREE_ROOT,
    *_PRIOR_V2R6_RUNTIME_ROOTS,
)
_PRIOR_V2R7_PROTECTED_ROOTS: Final = (
    _PRIOR_V2R7_WORKTREE_ROOT,
    f"{_PRIOR_V2R7_STATE_PREFIX}-authorization",
    f"{_PRIOR_V2R7_STATE_PREFIX}-ledger",
    *tuple(
        f"{_PRIOR_V2R7_STATE_PREFIX}-slot-{ordinal:02d}-{suffix}"
        for ordinal in range(1, 4)
        for suffix in ("clone", "cargo-cache", "artifact", "evidence", "trace")
    ),
)
_EXPECTED_PROTECTED_ROOTS_V2R8: Final = (
    "/home/erzhu419/mine_code/Asumption Agent",
    os.fspath(WORKTREE_ROOT),
    f"{_STATE_ROOT_V2}/q05b-shadow-trajectory-v2-dev-worktree",
    *_PRIOR_V2_PROTECTED_ROOTS,
    *_PRIOR_V2R1_PROTECTED_ROOTS,
    *_PRIOR_V2R2_PROTECTED_ROOTS,
    *_PRIOR_V2R3_PROTECTED_ROOTS,
    *_PRIOR_V2R4_PROTECTED_ROOTS,
    *_PRIOR_V2R5_PROTECTED_ROOTS,
    *_PRIOR_V2R6_PROTECTED_ROOTS,
    *_PRIOR_V2R7_PROTECTED_ROOTS,
    f"{_STATE_ROOT_V2}/m3-start-runtime-worktree",
    f"{_STATE_ROOT_V2}/q05b-attempt2-ledger-v1",
    f"{_STATE_ROOT_V2}/q05b-attempt-2-authorization",
    f"{_STATE_ROOT_V2}/rust-cargo-cache",
    "/home/erzhu419/.local/state/kg-op/meta-prior-theory-operations-worktree",
    "/home/erzhu419/.local/state/kg-op/meta-prior-production-campaigns",
    "/home/erzhu419/.local/state/kg-op/meta-prior-production-execution-preregs",
    "/home/erzhu419/.local/state/kg-op/meta-prior-production-executions",
)


_RAW_SHA256: Final = re.compile(r"[0-9a-f]{64}\Z")
_TAGGED_SHA256: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_COMMIT: Final = re.compile(r"[0-9a-f]{40}\Z")
_SAFE_NAME: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_TRACE_ID: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@/+-]{0,255}\Z")
_MAX_JSON_BYTES: Final = 64 * 1024 * 1024
_MAX_LEDGER_BYTES: Final = 4 * 1024 * 1024
_GAP_CODE: Final = "HELD_ACTOR_GAP_EXCEEDED"
_LEDGER_RECORD_NAMES_V2: Final = (
    "campaign.intent.json",
    "campaign.authorization-consumed.json",
    "slot-01.intent.json",
    "slot-01.consumed.json",
    "slot-01.trajectory-binding.json",
    "slot-01.outcome.json",
    "slot-02.intent.json",
    "slot-02.consumed.json",
    "slot-02.trajectory-binding.json",
    "slot-02.outcome.json",
    "slot-03.intent.json",
    "slot-03.consumed.json",
    "slot-03.trajectory-binding.json",
    "slot-03.outcome.json",
    "campaign.terminal.json",
    "campaign.completion.json",
)
_TRAJECTORY_MINIMAL_EVENT_FIELDS: Final = (
    "sequence",
    "timestamp_ns",
    "event_type",
    "span_id",
    "operation",
    "stage_or_null",
    "substep_or_null",
    "actor_or_null",
    "detail_or_null",
    "error_code_or_null",
    "error_detail_or_null",
    "record_digest",
)
_TRAJECTORY_FAILURE_ROW_FIELDS: Final = (
    "sequence",
    "timestamp_ns",
    "span_id",
    "operation",
    "stage_or_null",
    "substep_or_null",
    "actor_or_null",
    "detail_or_null",
    "error_code",
    "error_detail",
    "role_id_or_null",
    "sample_ordinal_or_null",
    "previous_finish_ns_or_null",
    "scheduled_start_ns_or_null",
    "actual_start_ns_or_null",
    "finish_ns_or_null",
    "collection_duration_ns_or_null",
    "scheduler_lateness_ns_or_null",
    "target_wait_ns_or_null",
    "gap_ns_or_null",
    "maximum_gap_ns_or_null",
    "gap_exceeded_or_null",
    "record_digest",
)
_PARTITION_CERTIFICATE_SCHEMA_V2: Final = (
    "hegel-phase3a-q05b-shadow-retry-post-origin-partition-certificate/2"
)
_MAX_RETAINED_FAILURE_ROWS: Final = 7
_MAX_RETAINED_CLEANUP_ROWS: Final = 12
_MAX_PARTITION_CERTIFICATE_BYTES: Final = 256 * 1024
_MAX_ERROR_RECEIPT_BYTES: Final = 128 * 1024
_SCHEDULING_ADMISSION_POLICY_SCHEMA_V6: Final = (
    "hegel-phase3a-q05b-shadow-retry-scheduling-admission-policy/6"
)
_SCHEDULING_ADMISSION_EVIDENCE_SCHEMA_V6: Final = (
    "hegel-phase3a-q05b-shadow-retry-scheduling-admission-evidence/6"
)
_SCHEDULING_ADMISSION_RECORD_SCHEMA_V6: Final = (
    "hegel-phase3a-q05b-shadow-retry-scheduling-admission-record/6"
)
_SCHEDULING_SUSTAINED_EVIDENCE_SCHEMA_V6: Final = (
    "hegel-phase3a-q05b-shadow-retry-sustained-publication-evidence/6"
)
_SCHEDULING_ADMISSION_FAILURE_V2: Final = "FAIL_CAMPAIGN_SCHEDULING_ADMISSION"
_SCHEDULING_CONTROLLER_EVENTS_V6: Final = (
    "SUPERVISOR_STAGE_04_OK",
    "SUPERVISOR_STAGE_05_BEGIN",
    "HELD_ACTOR_CLOSE_BEGIN",
)
_SCHEDULING_SAMPLER_ACTORS_V6: Final = (
    "PYTHON_ENDPOINT",
    "RUST_ENDPOINT",
)
_SPAWN_REGISTRATION_SCHEMA_V2: Final = (
    "hegel-phase3a-q05b-shadow-retry-spawn-registration/2"
)
_PRESPAWN_DOCKER_RECEIPT_SCHEMA_V2: Final = (
    "hegel-phase3a-q05b-shadow-retry-pre-spawn-docker-receipt/2"
)
_PUBLICATION_RECEIPT_SCHEMA_V2: Final = (
    "hegel-phase3a-q05b-shadow-retry-publication-receipt/2"
)
_BOUNDED_COMMAND_DIAGNOSTIC_SCHEMA_V2: Final = (
    "hegel-phase3a-q05b-shadow-retry-bounded-command-diagnostic/2"
)
_BOUNDED_COMMAND_EXCEPTION_DIAGNOSTIC_SCHEMA_V2: Final = (
    "hegel-phase3a-q05b-shadow-retry-bounded-command-exception-diagnostic/2"
)
_BOUNDED_RECEIPT_OMISSION_SCHEMA_V2: Final = (
    "hegel-phase3a-q05b-shadow-retry-bounded-receipt-omission/2"
)
_AUTHORITATIVE_ERROR_RECEIPT_SCHEMAS_V2: Final = (
    _BOUNDED_COMMAND_DIAGNOSTIC_SCHEMA_V2,
    _BOUNDED_COMMAND_EXCEPTION_DIAGNOSTIC_SCHEMA_V2,
    _SPAWN_REGISTRATION_SCHEMA_V2,
    _PRESPAWN_DOCKER_RECEIPT_SCHEMA_V2,
    _PUBLICATION_RECEIPT_SCHEMA_V2,
    _SCHEDULING_ADMISSION_EVIDENCE_SCHEMA_V6,
    _BOUNDED_RECEIPT_OMISSION_SCHEMA_V2,
)
_SPAWN_RECEIPT_MAX_BYTES: Final = 64
_SPAWN_RECEIPT_WAIT_SECONDS: Final = 2.0
_SPAWN_LAUNCHER_INTERPRETER_V2: Final = "/usr/bin/python3.10"
_SPAWN_LAUNCHER_SOURCE_V2: Final = (
    "import os,signal,sys\n"
    "fd=int(sys.argv[1])\n"
    "argv=sys.argv[2:]\n"
    "registered=False\n"
    "try:\n"
    " signal.pthread_sigmask(signal.SIG_SETMASK,set())\n"
    " signal.signal(signal.SIGINT,signal.SIG_DFL)\n"
    " signal.signal(signal.SIGCHLD,signal.SIG_DFL)\n"
    " payload=(str(os.getpid())+'\\n').encode('ascii')\n"
    " if os.write(fd,payload)!=len(payload): os._exit(125)\n"
    " registered=True\n"
    " os.close(fd)\n"
    " os.execve(argv[0],argv,dict(os.environ))\n"
    "except BaseException:\n"
    " try: os.close(fd)\n"
    " except BaseException: pass\n"
    " if registered:\n"
    "  try: os.write(2,b'HEGEL_Q05B_LAUNCHER_ACTUAL_EXEC_FAILED\\n')\n"
    "  except BaseException: pass\n"
    " os._exit(127 if registered else 126)\n"
)
_TARGET_STATIC_POLICY_DOMAIN: Final = (
    b"HEGEL/Q05B/ACTUAL/COMMIT_A_STATIC_POLICY/V1\x00"
)
_TARGET_SOURCE_BINDING_DOMAIN: Final = (
    "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN/TARGET_SOURCE_BINDING/V2"
)
_PYTHON310_DEPENDENCY_PREFLIGHT_POLICY_SCHEMA_V1: Final = (
    "hegel-phase3a-q05b-shadow-retry-python310-dependency-preflight-policy/1"
)
_PYTHON310_DEPENDENCY_PREFLIGHT_CHILD_SCHEMA_V1: Final = (
    "hegel-phase3a-q05b-shadow-retry-python310-dependency-preflight-child/1"
)
_PYTHON310_DEPENDENCY_PREFLIGHT_RECEIPT_SCHEMA_V1: Final = (
    "hegel-phase3a-q05b-shadow-retry-python310-dependency-preflight-receipt/1"
)
_PYTHON310_DEPENDENCY_PREFLIGHT_RECEIPT_DOMAIN_V1: Final = (
    "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R8/"
    "PYTHON310_VENDORED_TOMLI_CARGO_LOCK_PREFLIGHT_RECEIPT/V1"
)
_VENDORED_TOMLI_CLOSURE_DOMAIN_V1: Final = (
    "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R8/"
    "VENDORED_TOMLI_CLOSURE/V1"
)
_CARGO_LOCK_REGISTRY_PROJECTION_DOMAIN_V1: Final = (
    "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R8/"
    "CARGO_LOCK_REGISTRY_PACKAGE_PROJECTION/V1"
)
_DEPENDENCY_RUNTIME_ENVIRONMENT_DOMAIN_V1: Final = (
    "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R8/"
    "PYTHON310_DEPENDENCY_RUNTIME_ENVIRONMENT/V1"
)
_CONTROLLER_CHANGED_PATHS_FROM_TARGET_IN_EXACT_ORDER: Final = (
    "config/phase3_q05b_shadow_retry_campaign_v2.json",
    "docs/Hegel_Machine_Q05b_Shadow_Retry_Campaign_v2.md",
    "src/hegel_machine/phase3_q05b_shadow_retry_campaign_v2.py",
    "tests/test_phase3_q05b_shadow_retry_campaign_v2_core.py",
    "tests/test_phase3_q05b_shadow_retry_campaign_v2_tool.py",
    "tools/phase3_q05b_shadow_retry_campaign_v2.py",
)
_TARGET_REQUIRED_PATHS_V2R8: Final = (
    "config/phase3_q05b_dual_isolation_v2.json",
    "src/hegel_machine/_vendor/__init__.py",
    "src/hegel_machine/_vendor/tomli/LICENSE",
    "src/hegel_machine/_vendor/tomli/__init__.py",
    "src/hegel_machine/_vendor/tomli/_parser.py",
    "src/hegel_machine/_vendor/tomli/_re.py",
    "src/hegel_machine/_vendor/tomli/_types.py",
    "src/hegel_machine/phase3_q05b_actual_admission_v2.py",
    "src/hegel_machine/phase3_q05b_actual_artifact_v2.py",
    "src/hegel_machine/phase3_q05b_trajectory_v2.py",
    "tests/test_phase3_q05b_dual_qualification_v2.py",
    "tests/test_phase3_q05b_trajectory_v2.py",
    "tools/phase3_q05b_dual_qualification_v2.py",
)
_VENDORED_TOMLI_CLOSURE_ROWS_V1: Final = (
    (
        "src/hegel_machine/_vendor/__init__.py",
        57,
        "ee9748e550c7a30a129da0ca1fb2a93116f599541a57039ebd687a679dd86673",
    ),
    (
        "src/hegel_machine/_vendor/tomli/LICENSE",
        1_072,
        "b80816b0d530b8accb4c2211783790984a6e3b61922c2b5ee92f3372ab2742fe",
    ),
    (
        "src/hegel_machine/_vendor/tomli/__init__.py",
        314,
        "9eb042d7c0db5d14c2168ec4946e410de5a91c9cce86892f5e4db5e4633c6762",
    ),
    (
        "src/hegel_machine/_vendor/tomli/_parser.py",
        26_440,
        "a412234c86bf710b361e0943276961f0e25fa6d7c36ba7a0e7eec87a3e018c7b",
    ),
    (
        "src/hegel_machine/_vendor/tomli/_re.py",
        3_396,
        "a12359fe294523a72112e434d58452a14c9d050affa2417f9927474e4166bfdd",
    ),
    (
        "src/hegel_machine/_vendor/tomli/_types.py",
        254,
        "f864c6d9552a929c7032ace654ee05ef26ca75d21b027b801d77e65907138b74",
    ),
)
_VENDORED_TOMLI_CLOSURE_DIGEST_V1: Final = (
    "sha256:6b00d41d6f281d523a52fe4738c8260aa5680fecb38e605ebb9d19c2c0f73c54"
)
_HOST_SOURCE_ALLOWLIST_COUNT_V1: Final = 55
_HOST_SOURCE_ALLOWLIST_CANONICAL_LF_SHA256_V1: Final = (
    "97c6c00d1c981c5343a12054e09edd846d6931f964ad394e92efc560ccaa1fce"
)
_CARGO_LOCK_RELATIVE_PATH_V1: Final = (
    "rust/q1_archive_projection_oracle/Cargo.lock"
)
_CARGO_LOCK_BYTE_COUNT_V1: Final = 6_493
_CARGO_LOCK_RAW_SHA256_V1: Final = (
    "c4b9d1eff522017c85b151696d082a0ff9f88bd2890c93d932f7467323a13b49"
)
_CARGO_LOCK_FORMAT_V1: Final = 4
_CARGO_LOCK_PACKAGE_COUNT_V1: Final = 29
_CARGO_LOCK_REGISTRY_PACKAGE_COUNT_V1: Final = 21
_CARGO_LOCK_REGISTRY_PROJECTION_DIGEST_V1: Final = (
    "sha256:ea29b23b1e7d5e4f50677786b9188a420cf6f79b1f5a3cf19d9a6f45447489ea"
)
_DEPENDENCY_PREFLIGHT_PYCACHE_PREFIX_V1: Final = (
    "/nonexistent/hegel-q05b-v2r8-preflight-pyc"
)
_DEPENDENCY_PREFLIGHT_ARGV_PREFIX_V1: Final = (
    "/usr/bin/python3.10",
    "-I",
    "-S",
    "-B",
    "-X",
    "utf8",
    "-X",
    f"pycache_prefix={_DEPENDENCY_PREFLIGHT_PYCACHE_PREFIX_V1}",
    "-c",
)
_DEPENDENCY_PREFLIGHT_LOADED_VENDORED_MODULES_V1: Final = (
    "hegel_machine._vendor",
    "hegel_machine._vendor.tomli",
    "hegel_machine._vendor.tomli._parser",
    "hegel_machine._vendor.tomli._re",
)
_PYTHON310_DEPENDENCY_PREFLIGHT_PROGRAM_V1: Final = (
    "import hashlib,importlib,importlib.util,json,os,runpy,sys\n"
    "from pathlib import Path\n"
    "root=Path(sys.argv[1])\n"
    "cargo_path=Path(sys.argv[2])\n"
    "assert sys.flags.isolated==1\n"
    "assert sys.flags.no_site==1\n"
    "assert sys.flags.ignore_environment==1\n"
    "assert sys.dont_write_bytecode is True\n"
    "assert sys.flags.utf8_mode==1\n"
    "assert sys.pycache_prefix=='/nonexistent/hegel-q05b-v2r8-preflight-pyc'\n"
    "pycache_absent_before=not os.path.lexists(sys.pycache_prefix)\n"
    "assert pycache_absent_before\n"
    "assert importlib.util.find_spec('tomli') is None\n"
    "artifact_name='hegel_machine.phase3_q05b_actual_artifact_v2'\n"
    "assert artifact_name not in sys.modules\n"
    "supervisor_path=root/'tools/phase3_q05b_dual_qualification_v2.py'\n"
    "namespace=runpy.run_path(str(supervisor_path))\n"
    "host=namespace['_load_host_replay_module_v1'](root)\n"
    "assert artifact_name not in sys.modules\n"
    "artifact=importlib.import_module(artifact_name)\n"
    "assert artifact.tomllib.__name__=='hegel_machine._vendor.tomli'\n"
    "assert 'tomli' not in sys.modules\n"
    "pycache_absent_after=not os.path.lexists(sys.pycache_prefix)\n"
    "assert pycache_absent_after\n"
    "assert artifact.tomllib.loads('answer = 42\\n')=={'answer':42}\n"
    "raw=cargo_path.read_bytes()\n"
    "parsed=artifact.tomllib.loads(raw.decode('utf-8'))\n"
    "packages=parsed['package']\n"
    "registry_rows=[[row['name'],row['version'],row['source'],row['checksum']] for row in packages if row.get('source','').startswith('registry+')]\n"
    "canonical=lambda value:(json.dumps(value,ensure_ascii=True,allow_nan=False,sort_keys=True,separators=(',',':'))+'\\n').encode('ascii')\n"
    "domain_digest=lambda domain,value:'sha256:'+hashlib.sha256(domain.encode('ascii')+b'\\x00'+canonical(value)[:-1]).hexdigest()\n"
    "vendor_paths=['src/hegel_machine/_vendor/__init__.py','src/hegel_machine/_vendor/tomli/LICENSE','src/hegel_machine/_vendor/tomli/__init__.py','src/hegel_machine/_vendor/tomli/_parser.py','src/hegel_machine/_vendor/tomli/_re.py','src/hegel_machine/_vendor/tomli/_types.py']\n"
    "vendor_rows=[]\n"
    "for relative in vendor_paths:\n"
    " payload=(root/relative).read_bytes()\n"
    " vendor_rows.append([relative,len(payload),hashlib.sha256(payload).hexdigest()])\n"
    "allowlist=list(namespace['HOST_SOURCE_ALLOWLIST'])\n"
    "result={'schema_version':'hegel-phase3a-q05b-shadow-retry-python310-dependency-preflight-child/1','status':'PASS','python_flags':{'isolated':sys.flags.isolated,'no_site':sys.flags.no_site,'ignore_environment':sys.flags.ignore_environment,'dont_write_bytecode':sys.dont_write_bytecode,'utf8_mode':sys.flags.utf8_mode,'pycache_prefix':sys.pycache_prefix,'pycache_prefix_absent_before_import':pycache_absent_before,'pycache_prefix_absent_after_import':pycache_absent_after},'tomli_top_level_spec_or_null':None,'artifact_module_name':artifact.__name__,'vendored_tomli_module_name':artifact.tomllib.__name__,'top_level_tomli_in_sys_modules':False,'host_replay_module_name':host.__name__,'host_source_allowlist_count':len(allowlist),'host_source_allowlist_canonical_lf_sha256':hashlib.sha256(canonical(allowlist)).hexdigest(),'vendor_closure_rows':vendor_rows,'vendor_closure_digest':domain_digest('HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R8/VENDORED_TOMLI_CLOSURE/V1',vendor_rows),'loaded_vendored_modules':sorted(name for name in sys.modules if name.startswith('hegel_machine._vendor')),'cargo_lock':{'relative_path':'rust/q1_archive_projection_oracle/Cargo.lock','byte_count':len(raw),'raw_sha256':hashlib.sha256(raw).hexdigest(),'format':parsed.get('version'),'package_count':len(packages),'registry_package_count':len(registry_rows),'registry_package_rows':registry_rows,'registry_projection_digest':domain_digest('HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R8/CARGO_LOCK_REGISTRY_PACKAGE_PROJECTION/V1',registry_rows)}}\n"
    "sys.stdout.buffer.write(canonical(result))\n"
)
_EXPECTED_HASH_DOMAINS_V2R8: Final = {
    "contract": "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R8/CONTRACT/V2",
    "campaign_source_binding": "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R8/CAMPAIGN_SOURCE_BINDING/V2",
    "capsule": "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R8/CAPSULE/V2",
    "authorization": "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R8/AUTHORIZATION/V2",
    "cargo_manifest": "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R8/CARGO_MANIFEST/V2",
    "ledger_record": "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R8/LEDGER_RECORD/V2",
    "trajectory_binding": "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R8/TRAJECTORY_BINDING/V2",
    "trajectory_event_payloads": "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R8/EVENT_PAYLOADS/V2",
    "trajectory_suffix": "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R8/TRAJECTORY_SUFFIX/V2",
    "trajectory_other_nonfailure": "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R8/TRAJECTORY_OTHER_NONFAILURE/V2",
    "trajectory_failure_rows": "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R8/TRAJECTORY_FAILURE_ROWS/V2",
    "trajectory_cleanup_rows": "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R8/TRAJECTORY_CLEANUP_ROWS/V2",
    "trajectory_partition_certificate": "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R8/POST_ORIGIN_PARTITION_CERTIFICATE/V2",
    "slot_outcome": "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R8/SLOT_OUTCOME/V2",
    "campaign_terminal": "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R8/CAMPAIGN_TERMINAL/V2",
    "campaign_completion": "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R8/CAMPAIGN_COMPLETION/V2",
}
_PREDICATE_DIGEST_DOMAINS: Final = {
    "retry_policy": "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R8/RETRY_PREDICATE/V2",
    "success_policy": "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R8/SUCCESS_PREDICATE/V2",
    "ledger_policy": "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R8/LEDGER_STATE_REGISTRY/V2",
    "forbidden_scope": "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R8/FORBIDDEN_SCOPE/V2",
}
_EXPECTED_FIXED_ENVIRONMENT: Final = {
    "DOCKER_CONFIG": "/nonexistent",
    "GIT_ALLOW_PROTOCOL": "file",
    "GIT_ASKPASS": "/bin/false",
    "GIT_ATTR_NOSYSTEM": "1",
    "GIT_CONFIG_COUNT": "3",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_KEY_0": "core.fsmonitor",
    "GIT_CONFIG_KEY_1": "core.hooksPath",
    "GIT_CONFIG_KEY_2": "core.worktree",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_VALUE_0": "false",
    "GIT_CONFIG_VALUE_1": "/dev/null",
    "GIT_CONFIG_VALUE_2": "SLOT.clone_root",
    "GIT_NO_LAZY_FETCH": "1",
    "GIT_NO_REPLACE_OBJECTS": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_PROTOCOL_FROM_USER": "0",
    "GIT_SSH_COMMAND": "/bin/false",
    "GIT_TERMINAL_PROMPT": "0",
    "HOME": "/nonexistent",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "SSH_ASKPASS": "/bin/false",
    "TMPDIR": "/tmp",
    "TZ": "UTC0",
    "XDG_CONFIG_HOME": "/nonexistent",
}


class ShadowRetryCampaignErrorV2(RuntimeError):
    """Stable fail-closed campaign error with optional durable receipt."""

    def __init__(
        self,
        code: str,
        detail: str,
        *,
        terminal: bool = False,
        receipt: Mapping[str, object] | None = None,
    ) -> None:
        self.code = code
        self.detail = detail
        self.terminal = terminal
        self.receipt = deepcopy(dict(receipt)) if receipt is not None else None
        super().__init__(f"{code}: {detail}")


def _fail(
    code: str,
    detail: str,
    *,
    terminal: bool = False,
    receipt: Mapping[str, object] | None = None,
) -> NoReturn:
    raise ShadowRetryCampaignErrorV2(
        code,
        detail,
        terminal=terminal,
        receipt=receipt,
    )


def _pairs_no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail("FAIL_CAMPAIGN_CANONICAL", f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _validate_json_tree(value: object, label: str = "value") -> None:
    value_type = type(value)
    if value is None or value_type in (bool, int, str):
        return
    if value_type is float:
        if not math.isfinite(value):
            _fail("FAIL_CAMPAIGN_CANONICAL", f"{label} contains non-finite float")
        return
    if value_type is list:
        for index, item in enumerate(value):
            _validate_json_tree(item, f"{label}[{index}]")
        return
    if value_type is dict:
        for key, item in value.items():
            if type(key) is not str:
                _fail("FAIL_CAMPAIGN_CANONICAL", f"{label} has non-string key")
            _validate_json_tree(item, f"{label}.{key}")
        return
    _fail(
        "FAIL_CAMPAIGN_CANONICAL",
        f"{label} has unsupported exact type {value_type.__name__}",
    )


def _canonical_json_bytes(value: object) -> bytes:
    _validate_json_tree(value)
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _strict_json(raw: bytes, label: str, maximum: int = _MAX_JSON_BYTES) -> object:
    if type(raw) is not bytes or len(raw) > maximum:
        _fail("FAIL_CAMPAIGN_CANONICAL", f"{label} byte bound differs")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=lambda token: _fail(
                "FAIL_CAMPAIGN_CANONICAL", f"{label} constant differs: {token}"
            ),
        )
    except ShadowRetryCampaignErrorV2:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail("FAIL_CAMPAIGN_CANONICAL", f"{label} decode failed: {error}")
    _validate_json_tree(value, label)
    if _canonical_json_bytes(value) != raw:
        _fail("FAIL_CAMPAIGN_CANONICAL", f"{label} is not canonical single-LF JSON")
    return value


def _pretty_json(raw: bytes, label: str) -> object:
    if type(raw) is not bytes or len(raw) > _MAX_JSON_BYTES:
        _fail("FAIL_CAMPAIGN_CANONICAL", f"{label} byte bound differs")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=lambda token: _fail(
                "FAIL_CAMPAIGN_CANONICAL", f"{label} constant differs: {token}"
            ),
        )
    except ShadowRetryCampaignErrorV2:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail("FAIL_CAMPAIGN_CANONICAL", f"{label} decode failed: {error}")
    _validate_json_tree(value, label)
    expected = (
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8")
        + b"\n"
    )
    if expected != raw:
        _fail("FAIL_CAMPAIGN_CANONICAL", f"{label} pretty-byte format differs")
    return value


def _decoded_json_bytes(raw: bytes, label: str) -> object:
    if type(raw) is not bytes or len(raw) > _MAX_JSON_BYTES:
        _fail("FAIL_CAMPAIGN_CANONICAL", f"{label} byte bound differs")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=lambda token: _fail(
                "FAIL_CAMPAIGN_CANONICAL",
                f"{label} constant differs: {token}",
            ),
        )
    except ShadowRetryCampaignErrorV2:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail("FAIL_CAMPAIGN_CANONICAL", f"{label} decode failed: {error}")
    _validate_json_tree(value, label)
    return value


def _domain_digest(domain: str, value: object) -> str:
    if type(domain) is not str or not domain or "\x00" in domain:
        _fail("FAIL_CAMPAIGN_CONTRACT", "hash domain differs")
    return "sha256:" + sha256(
        domain.encode("ascii") + b"\x00" + _canonical_json_bytes(value)[:-1]
    ).hexdigest()


def _json_value_type_exact_v1(observed: object, expected: object) -> bool:
    """Compare JSON trees without bool/int or int/float equivalence."""

    if type(observed) is not type(expected):
        return False
    if type(expected) is dict:
        assert type(observed) is dict
        return set(observed) == set(expected) and all(
            _json_value_type_exact_v1(observed[key], expected[key])
            for key in expected
        )
    if type(expected) is list:
        assert type(observed) is list
        return len(observed) == len(expected) and all(
            _json_value_type_exact_v1(left, right)
            for left, right in zip(observed, expected, strict=True)
        )
    return observed == expected


def _exact_dict(
    value: object,
    fields: Sequence[str],
    label: str,
) -> dict[str, object]:
    if type(value) is not dict or set(value) != set(fields) or len(value) != len(fields):
        _fail("FAIL_CAMPAIGN_CONTRACT", f"{label} fields differ")
    return value


def _normalized_absolute_parts(path: Path, label: str) -> tuple[str, ...]:
    text = os.fspath(path)
    if (
        not path.is_absolute()
        or os.path.normpath(text) != text
        or path.name in (".", "..")
        or any(part in ("", ".", "..") for part in path.parts[1:])
    ):
        _fail("FAIL_CAMPAIGN_FILESYSTEM", f"{label} path differs")
    return tuple(path.parts[1:])


def _open_absolute_directory_chain_nofollow(
    path: Path,
    label: str,
) -> tuple[int, os.stat_result]:
    parts = _normalized_absolute_parts(path, label)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    current = os.open("/", flags)
    try:
        for part in parts:
            before = os.stat(part, dir_fd=current, follow_symlinks=False)
            if not stat.S_ISDIR(before.st_mode) or stat.S_ISLNK(before.st_mode):
                _fail(
                    "FAIL_CAMPAIGN_FILESYSTEM",
                    f"{label} has a non-directory or symlink component",
                )
            following = os.open(part, flags, dir_fd=current)
            opened = os.fstat(following)
            after = os.stat(part, dir_fd=current, follow_symlinks=False)
            if (
                (before.st_dev, before.st_ino, before.st_mode)
                != (opened.st_dev, opened.st_ino, opened.st_mode)
                or (before.st_dev, before.st_ino, before.st_mode)
                != (after.st_dev, after.st_ino, after.st_mode)
            ):
                os.close(following)
                _fail(
                    "FAIL_CAMPAIGN_FILESYSTEM",
                    f"{label} component identity changed",
                )
            os.close(current)
            current = following
        result = os.fstat(current)
        descriptor = current
        current = -1
        return descriptor, result
    finally:
        if current >= 0:
            os.close(current)


def _fd_mount_id(descriptor: int, label: str) -> int:
    path = Path(f"/proc/self/fdinfo/{descriptor}")
    try:
        raw = path.read_bytes()
    except OSError as error:
        _fail("FAIL_CAMPAIGN_FILESYSTEM", f"{label} mount-id read failed: {error}")
    rows = [line for line in raw.splitlines() if line.startswith(b"mnt_id:\t")]
    if len(rows) != 1:
        _fail("FAIL_CAMPAIGN_FILESYSTEM", f"{label} mount-id row differs")
    try:
        value = int(rows[0].split(b"\t", 1)[1])
    except (ValueError, IndexError):
        _fail("FAIL_CAMPAIGN_FILESYSTEM", f"{label} mount-id value differs")
    if value <= 0:
        _fail("FAIL_CAMPAIGN_FILESYSTEM", f"{label} mount-id is not positive")
    return value


def _verify_held_directory_path(
    path: Path,
    descriptor: int,
    label: str,
) -> dict[str, int]:
    held = os.fstat(descriptor)
    held_mount = _fd_mount_id(descriptor, f"{label} held descriptor")
    replay_fd, replay = _open_directory_nofollow(path, f"{label} pathname")
    try:
        replay_mount = _fd_mount_id(replay_fd, f"{label} pathname")
    finally:
        os.close(replay_fd)
    if (
        (held.st_dev, held.st_ino, held.st_mode, held.st_uid, held.st_nlink)
        != (replay.st_dev, replay.st_ino, replay.st_mode, replay.st_uid, replay.st_nlink)
        or held_mount != replay_mount
    ):
        _fail(
            "FAIL_CAMPAIGN_FILESYSTEM",
            f"{label} held/pathname identity differs",
            terminal=True,
        )
    return {
        "dev": held.st_dev,
        "ino": held.st_ino,
        "mount_id": held_mount,
        "mode": held.st_mode,
        "uid": held.st_uid,
        "nlink": held.st_nlink,
    }


def _read_regular_nofollow(
    path: Path,
    label: str,
    maximum: int = _MAX_JSON_BYTES,
    *,
    required_mode: int | None = None,
    required_uid: int | None = None,
) -> tuple[bytes, os.stat_result]:
    _normalized_absolute_parts(path, label)
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    parent_fd, _parent_status = _open_absolute_directory_chain_nofollow(
        path.parent,
        f"{label} parent",
    )
    try:
        descriptor = os.open(path.name, flags, dir_fd=parent_fd)
    except OSError as error:
        os.close(parent_fd)
        _fail("FAIL_CAMPAIGN_FILESYSTEM", f"{label} secure open failed: {error}")
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid
            != (os.geteuid() if required_uid is None else required_uid)
            or before.st_size > maximum
            or (
                required_mode is not None
                and stat.S_IMODE(before.st_mode) != required_mode
            )
        ):
            _fail("FAIL_CAMPAIGN_FILESYSTEM", f"{label} identity differs")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1 << 20, maximum + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum:
                _fail("FAIL_CAMPAIGN_FILESYSTEM", f"{label} exceeds byte bound")
        after = os.fstat(descriptor)
        pathname = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        identity = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_uid,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        if identity != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_uid,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or identity != (
            pathname.st_dev,
            pathname.st_ino,
            pathname.st_mode,
            pathname.st_uid,
            pathname.st_nlink,
            pathname.st_size,
            pathname.st_mtime_ns,
            pathname.st_ctime_ns,
        ):
            _fail("FAIL_CAMPAIGN_FILESYSTEM", f"{label} changed during read")
        return b"".join(chunks), before
    finally:
        os.close(descriptor)
        os.close(parent_fd)


def _open_directory_nofollow(path: Path, label: str) -> tuple[int, os.stat_result]:
    return _open_absolute_directory_chain_nofollow(path, label)


def _read_regular_at_nofollow(
    directory_fd: int,
    basename: str,
    label: str,
    maximum: int = _MAX_JSON_BYTES,
    *,
    required_mode: int | None = None,
) -> tuple[bytes, os.stat_result]:
    """Read one regular single-link file relative to an already held parent."""

    if type(basename) is not str or _SAFE_NAME.fullmatch(basename) is None:
        _fail("FAIL_CAMPAIGN_FILESYSTEM", f"{label} basename differs")
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(basename, flags, dir_fd=directory_fd)
    except OSError as error:
        _fail(
            "FAIL_CAMPAIGN_FILESYSTEM",
            f"{label} held-parent open failed: {error}",
        )
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.geteuid()
            or before.st_size > maximum
            or (
                required_mode is not None
                and stat.S_IMODE(before.st_mode) != required_mode
            )
        ):
            _fail("FAIL_CAMPAIGN_FILESYSTEM", f"{label} identity differs")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1 << 20, maximum + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum:
                _fail("FAIL_CAMPAIGN_FILESYSTEM", f"{label} exceeds byte bound")
        after = os.fstat(descriptor)
        pathname = os.stat(
            basename,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        identity = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_uid,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        if identity != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_uid,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or identity != (
            pathname.st_dev,
            pathname.st_ino,
            pathname.st_mode,
            pathname.st_uid,
            pathname.st_nlink,
            pathname.st_size,
            pathname.st_mtime_ns,
            pathname.st_ctime_ns,
        ):
            _fail("FAIL_CAMPAIGN_FILESYSTEM", f"{label} changed during read")
        return b"".join(chunks), before
    finally:
        os.close(descriptor)


def _create_private_root(path: Path, label: str) -> tuple[int, int]:
    if os.path.lexists(path):
        _fail("FAIL_CAMPAIGN_FRESHNESS", f"{label} already exists", terminal=True)
    parent_fd, parent_stat = _open_directory_nofollow(path.parent, f"{label} parent")
    try:
        os.mkdir(path.name, 0o700, dir_fd=parent_fd)
        os.fsync(parent_fd)
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        root_fd = os.open(path.name, flags, dir_fd=parent_fd)
        root_stat = os.fstat(root_fd)
        entry_stat = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            stat.S_IMODE(root_stat.st_mode) != 0o700
            or root_stat.st_uid != os.geteuid()
            or root_stat.st_nlink < 2
            or (root_stat.st_dev, root_stat.st_ino)
            != (entry_stat.st_dev, entry_stat.st_ino)
            or (parent_stat.st_dev, parent_stat.st_ino)
            != (os.fstat(parent_fd).st_dev, os.fstat(parent_fd).st_ino)
        ):
            os.close(root_fd)
            _fail("FAIL_CAMPAIGN_FILESYSTEM", f"{label} creation identity differs", terminal=True)
        return parent_fd, root_fd
    except BaseException:
        os.close(parent_fd)
        raise


def _publish_immutable_at(
    directory_fd: int,
    basename: str,
    value_or_bytes: Mapping[str, object] | bytes,
    *,
    publication_label: str | None = None,
    maximum_bytes: int = _MAX_LEDGER_BYTES,
    known_failure_code: str = "FAIL_CAMPAIGN_LEDGER",
    commit_callback: Callable[[], None] | None = None,
) -> bytes:
    if (
        type(known_failure_code) is not str
        or re.fullmatch(r"FAIL_CAMPAIGN_[A-Z_]+", known_failure_code) is None
    ):
        _fail(
            "FAIL_CAMPAIGN_CONTRACT",
            "publication known-failure code differs",
            terminal=True,
        )
    if _SAFE_NAME.fullmatch(basename) is None:
        _fail(
            known_failure_code,
            "publication basename differs before O_CREAT",
            terminal=True,
            receipt={
                "schema_version": _PUBLICATION_RECEIPT_SCHEMA_V2,
                "publication_outcome": "KNOWN_NO_PUBLICATION",
                "publication_basename": str(basename)[:128],
            },
        )
    if publication_label is None:
        publication_label = basename
    if type(publication_label) is not str or _SAFE_NAME.fullmatch(
        publication_label
    ) is None:
        _fail(
            known_failure_code,
            "publication label differs before O_CREAT",
            terminal=True,
            receipt={
                "schema_version": _PUBLICATION_RECEIPT_SCHEMA_V2,
                "publication_outcome": "KNOWN_NO_PUBLICATION",
                "publication_basename": basename,
            },
        )
    publication_context = (
        f"publication_label={publication_label}; basename={basename}"
    )
    publication_receipt = {
        "schema_version": _PUBLICATION_RECEIPT_SCHEMA_V2,
        "publication_outcome": "UNKNOWN_AFTER_CREATE_ATTEMPT",
        "publication_label": publication_label,
        "publication_basename": basename,
    }
    known_no_publication_receipt = {
        **publication_receipt,
        "publication_outcome": "KNOWN_NO_PUBLICATION",
    }
    payload = (
        value_or_bytes
        if type(value_or_bytes) is bytes
        else _canonical_json_bytes(dict(value_or_bytes))
    )
    if type(maximum_bytes) is not int or maximum_bytes < 0 or len(payload) > maximum_bytes:
        _fail(
            known_failure_code,
            f"{publication_context}; publication exceeds byte bound before O_CREAT",
            terminal=True,
            receipt=known_no_publication_receipt,
        )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    replay_fd = -1
    create_attempted = False
    known_no_publication = False
    try:
        try:
            # This guard intentionally moves before the O_CREAT call.  An
            # asynchronous exception after the kernel installs the entry but
            # before Python stores the returned descriptor must never be
            # misclassified as a known pre-publication failure.
            create_attempted = True
            descriptor = os.open(
                basename,
                flags,
                0o444,
                dir_fd=directory_fd,
            )
        except OSError as error:
            # No descriptor was returned, so this is a known no-publication
            # failure rather than an ambiguous partially durable record.
            known_no_publication = True
            _fail(
                known_failure_code,
                f"{publication_context}; exclusive publication failed: {error}",
                terminal=True,
                receipt=known_no_publication_receipt,
            )
        except BaseException as error:
            # An asynchronous exception around O_CREAT cannot prove whether
            # the directory entry was installed.
            _fail(
                "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN",
                f"{publication_context}; exclusive publication outcome uncertain: "
                f"{type(error).__name__}:{error}",
                terminal=True,
                receipt=publication_receipt,
            )
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                _fail(
                    "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN",
                    f"{publication_context}; publication made no progress after O_CREAT",
                    terminal=True,
                    receipt=publication_receipt,
                )
            offset += written
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        created_stat = os.fstat(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.fsync(directory_fd)
        replay_fd = os.open(
            basename,
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
        replay_stat = os.fstat(replay_fd)
        replay_chunks: list[bytes] = []
        while True:
            chunk = os.read(replay_fd, 1 << 20)
            if not chunk:
                break
            replay_chunks.append(chunk)
        os.close(replay_fd)
        replay_fd = -1
        replay = b"".join(replay_chunks)
        pathname_stat = os.stat(
            basename,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        created_identity = (
            created_stat.st_dev,
            created_stat.st_ino,
            created_stat.st_mode,
            created_stat.st_uid,
            created_stat.st_nlink,
            created_stat.st_size,
        )
        if (
            replay != payload
            or created_identity
            != (
                replay_stat.st_dev,
                replay_stat.st_ino,
                replay_stat.st_mode,
                replay_stat.st_uid,
                replay_stat.st_nlink,
                replay_stat.st_size,
            )
            or created_identity
            != (
                pathname_stat.st_dev,
                pathname_stat.st_ino,
                pathname_stat.st_mode,
                pathname_stat.st_uid,
                pathname_stat.st_nlink,
                pathname_stat.st_size,
            )
            or stat.S_IMODE(replay_stat.st_mode) != 0o444
            or replay_stat.st_nlink != 1
            or replay_stat.st_uid != os.geteuid()
        ):
            _fail(
                "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN",
                f"{publication_context}; publication replay differs after O_CREAT",
                terminal=True,
                receipt=publication_receipt,
            )
        if commit_callback is not None:
            commit_callback()
        return payload
    except ShadowRetryCampaignErrorV2 as error:
        if (
            create_attempted
            and not known_no_publication
            and error.code != "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN"
        ):
            _fail(
                "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN",
                f"{publication_context}; publication failed after O_CREAT: "
                f"{error.code}:{error.detail}",
                terminal=True,
                receipt=(
                    error.receipt
                    if type(error.receipt) is dict
                    else publication_receipt
                ),
            )
        raise
    except BaseException as error:
        code = (
            "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN"
            if create_attempted and not known_no_publication
            else known_failure_code
        )
        _fail(
            code,
            f"{publication_context}; publication system operation failed: "
            f"{type(error).__name__}:{error}",
            terminal=True,
            receipt=publication_receipt,
        )
    finally:
        for open_fd in (replay_fd, descriptor):
            if open_fd >= 0:
                try:
                    os.close(open_fd)
                except BaseException:
                    # A close failure after O_CREAT is already covered by the
                    # enclosing uncertain failure path when it is observable.
                    pass


def _scheduling_admission_policy_v2(
    contract: Mapping[str, object],
) -> dict[str, object]:
    execution = contract.get("execution_policy")
    policy = execution.get("scheduling_admission_gate") if type(execution) is dict else None
    expected_fields = {
        "schema_version", "clock_id", "maximum_windows", "quiet_period_ns",
        "bursts_per_window", "producer_lanes_in_exact_order",
        "records_per_burst", "controller_records_per_burst",
        "sampler_lane_count", "sampler_records_per_lane_per_burst",
        "sampling_target_wait_ns", "maximum_unsampled_gap_ns",
        "maximum_sampler_deadline_to_post_enqueue_anchor_ns",
        "maximum_projected_gap_ns", "reserved_runtime_margin_ns",
        "sampler_event_records_per_sample",
        "maximum_sampler_event_production_rate_records_per_second",
        "minimum_sustained_publication_rate_records_per_second",
        "maximum_utilization_numerator", "maximum_utilization_denominator",
        "sustained_offered_rate_records_per_second",
        "sustained_records_per_cycle", "sustained_cycle_count",
        "sustained_measurement_duration_ns", "sustained_epoch_duration_ns",
        "sustained_prefill_retained_records", "sustained_timed_records",
        "prior_failed_trajectory_event_count", "minimum_final_retained_records",
        "sustained_queue_target_records",
        "maximum_sustained_queue_high_watermark_records",
        "maximum_sustained_terminal_drain_ns",
        "maximum_burst_drain_ns", "terminal_drain_policy_ns",
        "queue_capacity_records", "normal_sampler_capacity_records",
        "failure_closure_reserve_records",
        "maximum_queue_high_watermark_records",
        "publication_payload_byte_count", "publication_protocol",
        "retained_probe_success_file_fsync_count",
        "retained_probe_success_directory_fsync_count",
        "retained_probe_success_total_fsync_count",
        "retained_probe_success_cleanup_directory_fsync_count",
        "retained_probe_duration_scope",
        "error_cleanup_directory_fsync_required_if_namespace_mutated",
        "queue_policy",
        "cleanup_policy", "claim_ceiling",
    }
    exact_ints = {
        "maximum_windows": 3,
        "quiet_period_ns": 30_000_000_000,
        "bursts_per_window": 16,
        "records_per_burst": 7,
        "controller_records_per_burst": 3,
        "sampler_lane_count": 2,
        "sampler_records_per_lane_per_burst": 2,
        "sampling_target_wait_ns": 125_000_000,
        "maximum_unsampled_gap_ns": 250_000_000,
        "maximum_sampler_deadline_to_post_enqueue_anchor_ns": 25_000_000,
        "maximum_projected_gap_ns": 150_000_000,
        "reserved_runtime_margin_ns": 100_000_000,
        "sampler_event_records_per_sample": 2,
        "maximum_sampler_event_production_rate_records_per_second": 32,
        "minimum_sustained_publication_rate_records_per_second": 40,
        "maximum_utilization_numerator": 4,
        "maximum_utilization_denominator": 5,
        "sustained_offered_rate_records_per_second": 40,
        "sustained_records_per_cycle": 5,
        "sustained_cycle_count": 960,
        "sustained_measurement_duration_ns": 120_000_000_000,
        "sustained_epoch_duration_ns": 10_000_000_000,
        "sustained_prefill_retained_records": 5_411,
        "sustained_timed_records": 4_800,
        "prior_failed_trajectory_event_count": 5_410,
        "minimum_final_retained_records": 10_211,
        "sustained_queue_target_records": 64,
        "maximum_sustained_queue_high_watermark_records": 64,
        "maximum_sustained_terminal_drain_ns": 2_000_000_000,
        "maximum_burst_drain_ns": 5_000_000_000,
        "terminal_drain_policy_ns": 120_000_000_000,
        "queue_capacity_records": 512,
        "normal_sampler_capacity_records": 384,
        "failure_closure_reserve_records": 128,
        "maximum_queue_high_watermark_records": 7,
        "publication_payload_byte_count": 4096,
        "retained_probe_success_file_fsync_count": 1,
        "retained_probe_success_directory_fsync_count": 3,
        "retained_probe_success_total_fsync_count": 4,
        "retained_probe_success_cleanup_directory_fsync_count": 0,
    }
    if (
        type(policy) is not dict
        or set(policy) != expected_fields
        or policy.get("schema_version") != _SCHEDULING_ADMISSION_POLICY_SCHEMA_V6
        or policy.get("clock_id") != "CLOCK_MONOTONIC"
        or any(type(policy.get(name)) is not int or policy.get(name) != expected
               for name, expected in exact_ints.items())
        or policy.get("producer_lanes_in_exact_order")
        != ["CONTROLLER", *_SCHEDULING_SAMPLER_ACTORS_V6]
        or int(policy.get("controller_records_per_burst", -1))
        + int(policy.get("sampler_lane_count", -1))
        * int(policy.get("sampler_records_per_lane_per_burst", -1))
        != int(policy.get("records_per_burst", -1))
        or int(policy.get("sampling_target_wait_ns", -1))
        + int(policy.get(
            "maximum_sampler_deadline_to_post_enqueue_anchor_ns", -1
        ))
        != int(policy.get("maximum_projected_gap_ns", -1))
        or int(policy.get("maximum_projected_gap_ns", -1))
        + int(policy.get("reserved_runtime_margin_ns", -1))
        != int(policy.get("maximum_unsampled_gap_ns", -1))
        or 1_000_000_000 % int(policy.get("sampling_target_wait_ns", -1)) != 0
        or int(policy.get("sampler_lane_count", -1))
        * int(policy.get("sampler_event_records_per_sample", -1))
        * 1_000_000_000
        // int(policy.get("sampling_target_wait_ns", -1))
        != int(policy.get(
            "maximum_sampler_event_production_rate_records_per_second", -1
        ))
        or int(policy.get("sustained_offered_rate_records_per_second", -1))
        != int(policy.get(
            "minimum_sustained_publication_rate_records_per_second", -1
        ))
        or int(policy.get("sustained_records_per_cycle", -1))
        * 1_000_000_000
        // int(policy.get("sampling_target_wait_ns", -1))
        != int(policy.get("sustained_offered_rate_records_per_second", -1))
        or int(policy.get("sustained_records_per_cycle", -1))
        * int(policy.get("sustained_cycle_count", -1))
        != int(policy.get("sustained_timed_records", -1))
        or int(policy.get("sustained_cycle_count", -1))
        * int(policy.get("sampling_target_wait_ns", -1))
        != int(policy.get("sustained_measurement_duration_ns", -1))
        or 1_000_000_000 % int(policy.get(
            "sustained_offered_rate_records_per_second", -1
        )) != 0
        or int(policy.get("sustained_measurement_duration_ns", -1))
        % int(policy.get("sustained_epoch_duration_ns", -1)) != 0
        or int(policy.get("sustained_offered_rate_records_per_second", -1))
        * int(policy.get("sustained_measurement_duration_ns", -1))
        // 1_000_000_000
        != int(policy.get("sustained_timed_records", -1))
        or int(policy.get("sustained_prefill_retained_records", -1))
        + int(policy.get("sustained_timed_records", -1))
        != int(policy.get("minimum_final_retained_records", -1))
        or int(policy.get("minimum_final_retained_records", -1))
        <= int(policy.get("prior_failed_trajectory_event_count", -1))
        or int(policy.get("normal_sampler_capacity_records", -1))
        + int(policy.get("failure_closure_reserve_records", -1))
        != int(policy.get("queue_capacity_records", -1))
        or int(policy.get("sustained_queue_target_records", -1))
        != int(policy.get(
            "maximum_sustained_queue_high_watermark_records", -1
        ))
        or int(policy.get("sustained_queue_target_records", -1))
        > int(policy.get("normal_sampler_capacity_records", -1))
        or int(policy.get(
            "maximum_sampler_event_production_rate_records_per_second", -1
        )) * int(policy.get("maximum_utilization_denominator", -1))
        > int(policy.get(
            "minimum_sustained_publication_rate_records_per_second", -1
        )) * int(policy.get("maximum_utilization_numerator", -1))
        or policy.get("publication_protocol")
        != "STAGING_EXCL_NOFOLLOW_WRITE_FCHMOD_FILE_FSYNC_DIR_FSYNC_LINK_FINAL_NOFOLLOW_DIR_FSYNC_UNLINK_STAGING_DIR_FSYNC_READBACK"
        or policy.get("retained_probe_duration_scope")
        != "OPEN_WRITE_FCHMOD_FILE_FSYNC_DIR_FSYNC_LINK_DIR_FSYNC_UNLINK_DIR_FSYNC_READBACK_REPLAY_FD_CLOSE_STAGING_ABSENCE_PROOF"
        or policy.get(
            "error_cleanup_directory_fsync_required_if_namespace_mutated"
        ) is not True
        or policy.get("queue_policy")
        != "THREE_PRODUCER_LANES_PUT_NOWAIT_SINGLE_FIFO_DURABILITY_WORKER_NORMAL_SAMPLER_CAP_384_FAILURE_CLOSURE_RESERVE_128_QUEUE_LOCK_RELEASED_BEFORE_PUBLICATION_ZERO_DROP_DUPLICATE_OR_REORDER"
        or policy.get("cleanup_policy")
        != "CRITICAL_BURSTS_UNLINK_AFTER_EACH_PROBE_SUSTAINED_PHASE_RETAINS_ALL_FINALS_IN_ONE_DIRECTORY_THROUGH_MEASUREMENT_THEN_UNLINKS_FINAL_AND_STAGING_DIR_FSYNC_AND_PROVES_ABSENCE"
        or policy.get("claim_ceiling")
        != "OBSERVED_PRECONSUME_BOUND_STAGE5_ENQUEUE_AND_120_SECOND_SAME_DIRECTORY_RETAINED_FILE_PUBLICATION_RATE_HEALTH_ONLY_NOT_A_RUNTIME_SCHEDULING_DURABILITY_OR_250MS_GAP_GUARANTEE"
    ):
        _fail("FAIL_CAMPAIGN_CONTRACT", "scheduling admission policy differs", terminal=True)
    return deepcopy(policy)


def _python310_dependency_preflight_policy_v1(
    contract: Mapping[str, object],
) -> dict[str, object]:
    """Return the exact pre-capsule Python 3.10 dependency policy."""

    execution = contract.get("execution_policy")
    if type(execution) is not dict:
        _fail("FAIL_CAMPAIGN_CONTRACT", "execution policy differs")
    policy = execution.get("python310_dependency_preflight_gate")
    expected = {
        "schema_version": _PYTHON310_DEPENDENCY_PREFLIGHT_POLICY_SCHEMA_V1,
        "status": (
            "REQUIRED_BEFORE_CAPSULE_AUTHORIZATION_STATE_OR_SUPERVISOR_SPAWN"
        ),
        "interpreter_path": "/usr/bin/python3.10",
        "argv_prefix_before_program": list(
            _DEPENDENCY_PREFLIGHT_ARGV_PREFIX_V1
        ),
        "cwd": "/",
        "program_raw_sha256": sha256(
            _PYTHON310_DEPENDENCY_PREFLIGHT_PROGRAM_V1.encode("utf-8")
        ).hexdigest(),
        "pycache_prefix": _DEPENDENCY_PREFLIGHT_PYCACHE_PREFIX_V1,
        "pycache_prefix_must_remain_absent": True,
        "timeout_seconds": 30,
        "stdout_max_bytes": 1_048_576,
        "stderr_max_bytes": 1_048_576,
        "supervisor_relative_path": (
            "tools/phase3_q05b_dual_qualification_v2.py"
        ),
        "artifact_relative_path": (
            "src/hegel_machine/phase3_q05b_actual_artifact_v2.py"
        ),
        "vendored_tomli_module_name": "hegel_machine._vendor.tomli",
        "top_level_tomli_spec_expected_null": True,
        "top_level_tomli_in_sys_modules_allowed": False,
        "vendor_closure_rows_in_exact_order": [
            list(row) for row in _VENDORED_TOMLI_CLOSURE_ROWS_V1
        ],
        "vendor_closure_digest_domain": _VENDORED_TOMLI_CLOSURE_DOMAIN_V1,
        "vendor_closure_digest": _VENDORED_TOMLI_CLOSURE_DIGEST_V1,
        "loaded_vendored_modules_in_exact_order": list(
            _DEPENDENCY_PREFLIGHT_LOADED_VENDORED_MODULES_V1
        ),
        "host_source_allowlist_count": _HOST_SOURCE_ALLOWLIST_COUNT_V1,
        "host_source_allowlist_canonical_lf_sha256": (
            _HOST_SOURCE_ALLOWLIST_CANONICAL_LF_SHA256_V1
        ),
        "cargo_lock_relative_path": _CARGO_LOCK_RELATIVE_PATH_V1,
        "cargo_lock_git_object_must_match_target_commit_a": True,
        "cargo_lock_byte_count": _CARGO_LOCK_BYTE_COUNT_V1,
        "cargo_lock_raw_sha256": _CARGO_LOCK_RAW_SHA256_V1,
        "cargo_lock_format": _CARGO_LOCK_FORMAT_V1,
        "cargo_lock_package_count": _CARGO_LOCK_PACKAGE_COUNT_V1,
        "cargo_lock_registry_package_count": (
            _CARGO_LOCK_REGISTRY_PACKAGE_COUNT_V1
        ),
        "cargo_lock_registry_projection_domain": (
            _CARGO_LOCK_REGISTRY_PROJECTION_DOMAIN_V1
        ),
        "cargo_lock_registry_projection_digest": (
            _CARGO_LOCK_REGISTRY_PROJECTION_DIGEST_V1
        ),
        "runtime_environment_digest_domain": (
            _DEPENDENCY_RUNTIME_ENVIRONMENT_DOMAIN_V1
        ),
        "receipt_digest_domain": (
            _PYTHON310_DEPENDENCY_PREFLIGHT_RECEIPT_DOMAIN_V1
        ),
        "replay_on_preflight_record_authorization_and_run": True,
        "cache_allowed": False,
        "stdout_policy": "EXACT_CANONICAL_SINGLE_LF_CHILD_RECEIPT",
        "stderr_policy": "EMPTY",
    }
    if not _json_value_type_exact_v1(policy, expected):
        _fail(
            "FAIL_CAMPAIGN_CONTRACT",
            "Python 3.10 dependency preflight policy differs",
        )
    return deepcopy(policy)


def _scheduling_probe_payload_v6(
    slot_id: str,
    window: int,
    burst: int,
    enqueue_ordinal: int,
    lane_id: str,
    lane_event_ordinal: int,
    operation: str,
) -> bytes:
    record: dict[str, object] = {
        "schema_version": _SCHEDULING_ADMISSION_RECORD_SCHEMA_V6,
        "slot_id": slot_id,
        "window_ordinal": window,
        "burst_ordinal": burst,
        "enqueue_ordinal": enqueue_ordinal,
        "lane_id": lane_id,
        "lane_event_ordinal": lane_event_ordinal,
        "operation": operation,
        "padding": "",
    }
    base_size = len(_canonical_json_bytes(record))
    padding_size = 4096 - base_size
    if (
        type(slot_id) is not str
        or _SAFE_NAME.fullmatch(slot_id) is None
        or any(type(item) is not int or item <= 0 for item in (
            window, burst, enqueue_ordinal, lane_event_ordinal
        ))
        or type(lane_id) is not str
        or type(operation) is not str
        or padding_size < 0
    ):
        _fail(_SCHEDULING_ADMISSION_FAILURE_V2, "record inputs differ", terminal=True)
    record["padding"] = "0" * padding_size
    payload = _canonical_json_bytes(record)
    if len(payload) != 4096:
        _fail(_SCHEDULING_ADMISSION_FAILURE_V2, "record byte count differs", terminal=True)
    return payload


_SCHEDULING_SUSTAINED_OPERATIONS_V6: Final = (
    ("PYTHON_ENDPOINT", "LIVE_RESOURCE_SAMPLE_BEGIN"),
    ("RUST_ENDPOINT", "LIVE_RESOURCE_SAMPLE_BEGIN"),
    ("PYTHON_ENDPOINT", "LIVE_RESOURCE_SAMPLE_OK"),
    ("RUST_ENDPOINT", "LIVE_RESOURCE_SAMPLE_OK"),
    ("CONTROLLER", "CONTROLLER_SERVICE_MARGIN"),
)
_SCHEDULING_SUSTAINED_CHAIN_GENESIS_V6: Final = sha256(
    b"hegel-phase3a-q05b-scheduling-sustained-chain-genesis/6"
).hexdigest()
_SCHEDULING_SUSTAINED_CHAIN_DOMAIN_V6: Final = (
    b"hegel-phase3a-q05b-scheduling-sustained-chain-step/6\x00"
)


def _scheduling_sustained_record_identity_v6(
    slot_id: str,
    window: int,
    global_ordinal: int,
    prefill_records: int,
) -> dict[str, object]:
    if (
        type(slot_id) is not str
        or _SAFE_NAME.fullmatch(slot_id) is None
        or type(window) is not int
        or not 1 <= window <= 3
        or type(global_ordinal) is not int
        or global_ordinal <= 0
        or type(prefill_records) is not int
        or prefill_records <= 0
    ):
        _fail(
            _SCHEDULING_ADMISSION_FAILURE_V2,
            "sustained record identity inputs differ",
            terminal=True,
        )
    if global_ordinal <= prefill_records:
        phase = "PREFILL_RETAINED"
        phase_ordinal = global_ordinal
    else:
        phase = "TIMED_SUSTAINED"
        phase_ordinal = global_ordinal - prefill_records
    cycle_event_ordinal = (
        (phase_ordinal - 1) % len(_SCHEDULING_SUSTAINED_OPERATIONS_V6)
    ) + 1
    cycle_ordinal = (
        (phase_ordinal - 1) // len(_SCHEDULING_SUSTAINED_OPERATIONS_V6)
    ) + 1
    lane_id, operation = _SCHEDULING_SUSTAINED_OPERATIONS_V6[
        cycle_event_ordinal - 1
    ]
    return {
        "schema_version": _SCHEDULING_ADMISSION_RECORD_SCHEMA_V6,
        "phase": phase,
        "phase_ordinal": phase_ordinal,
        "slot_id": slot_id,
        "window_ordinal": window,
        "global_ordinal": global_ordinal,
        "lane_id": lane_id,
        "cycle_ordinal": cycle_ordinal,
        "cycle_event_ordinal": cycle_event_ordinal,
        "lane_event_ordinal": cycle_ordinal,
        "operation": operation,
    }


def _scheduling_sustained_record_digest_v6(
    slot_id: str,
    window: int,
    global_ordinal: int,
    prefill_records: int,
) -> str:
    return sha256(_canonical_json_bytes(
        _scheduling_sustained_record_identity_v6(
            slot_id, window, global_ordinal, prefill_records
        )
    )).hexdigest()


def _scheduling_sustained_probe_payload_v6(
    slot_id: str,
    window: int,
    global_ordinal: int,
    prefill_records: int,
) -> bytes:
    record = _scheduling_sustained_record_identity_v6(
        slot_id, window, global_ordinal, prefill_records
    )
    record["record_digest"] = sha256(_canonical_json_bytes(record)).hexdigest()
    record["padding"] = ""
    padding_size = 4096 - len(_canonical_json_bytes(record))
    if padding_size < 0:
        _fail(
            _SCHEDULING_ADMISSION_FAILURE_V2,
            "sustained record exceeds publication payload bound",
            terminal=True,
        )
    record["padding"] = "0" * padding_size
    payload = _canonical_json_bytes(record)
    if len(payload) != 4096:
        _fail(
            _SCHEDULING_ADMISSION_FAILURE_V2,
            "sustained record byte count differs",
            terminal=True,
        )
    return payload


def _scheduling_sustained_chain_step_v6(
    previous_digest: str,
    global_ordinal: int,
    record_digest: str,
) -> str:
    if (
        type(previous_digest) is not str
        or _RAW_SHA256.fullmatch(previous_digest) is None
        or type(global_ordinal) is not int
        or not 1 <= global_ordinal < 2**64
        or type(record_digest) is not str
        or _RAW_SHA256.fullmatch(record_digest) is None
    ):
        _fail(
            _SCHEDULING_ADMISSION_FAILURE_V2,
            "sustained chain inputs differ",
            terminal=True,
        )
    return sha256(
        _SCHEDULING_SUSTAINED_CHAIN_DOMAIN_V6
        + bytes.fromhex(previous_digest)
        + global_ordinal.to_bytes(8, "big")
        + bytes.fromhex(record_digest)
    ).hexdigest()


def _scheduling_sustained_chain_checkpoints_v6(
    slot_id: str,
    window: int,
    maximum_ordinal: int,
    prefill_records: int,
    checkpoints: set[int],
) -> dict[int, str]:
    if (
        type(maximum_ordinal) is not int
        or maximum_ordinal < 0
        or type(prefill_records) is not int
        or prefill_records <= 0
        or type(checkpoints) is not set
        or any(type(item) is not int or not 0 <= item <= maximum_ordinal
               for item in checkpoints)
    ):
        _fail(
            _SCHEDULING_ADMISSION_FAILURE_V2,
            "sustained checkpoint inputs differ",
            terminal=True,
        )
    chain = _SCHEDULING_SUSTAINED_CHAIN_GENESIS_V6
    observed = {0: chain} if 0 in checkpoints else {}
    for ordinal in range(1, maximum_ordinal + 1):
        chain = _scheduling_sustained_chain_step_v6(
            chain,
            ordinal,
            _scheduling_sustained_record_digest_v6(
                slot_id, window, ordinal, prefill_records
            ),
        )
        if ordinal in checkpoints:
            observed[ordinal] = chain
    return observed


def _scheduling_sustained_probe_token_v6(
    window: int, global_ordinal: int
) -> str:
    if (
        type(window) is not int
        or not 1 <= window <= 3
        or type(global_ordinal) is not int
        or not 1 <= global_ordinal <= 999_999
    ):
        _fail(
            _SCHEDULING_ADMISSION_FAILURE_V2,
            "sustained probe token inputs differ",
            terminal=True,
        )
    return f"w{window:02d}-r{global_ordinal:06d}"


def _scheduling_sustained_final_name_v6(token: str) -> str:
    if type(token) is not str or re.fullmatch(r"w[0-9]{2}-r[0-9]{6}", token) is None:
        _fail(
            _SCHEDULING_ADMISSION_FAILURE_V2,
            "sustained probe final token differs",
            terminal=True,
        )
    return f"q05b-admission-{token}.probe-v6"


def _four_fsync_publication_probe_v6(
    directory_fd: int,
    unique_token: str,
    payload: bytes,
    clock_ns: Callable[[], int] = time.monotonic_ns,
) -> int:
    """Measure the target's four-fsync immutable publication, then remove it."""
    if (
        type(directory_fd) is not int or directory_fd < 0
        or type(unique_token) is not str
        or re.fullmatch(r"w[0-9]{2}-p[0-9]{3}", unique_token) is None
        or type(payload) is not bytes or len(payload) != 4096
    ):
        _fail(_SCHEDULING_ADMISSION_FAILURE_V2, "probe inputs differ", terminal=True)
    staging = f".q05b-admission-{unique_token}.staging-v2"
    final = f"q05b-admission-{unique_token}.probe-v2"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    replay_fd = -1
    duration: int | None = None
    primary_error: BaseException | None = None
    try:
        started = clock_ns()
        if type(started) is not int or started < 0:
            raise RuntimeError("probe start clock differs")
        descriptor = os.open(staging, flags, 0o600, dir_fd=directory_fd)
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise OSError("probe write made no progress")
            offset += written
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.fsync(directory_fd)
        os.link(staging, final, src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd, follow_symlinks=False)
        os.fsync(directory_fd)
        os.unlink(staging, dir_fd=directory_fd)
        os.fsync(directory_fd)
        replay_fd = os.open(final, os.O_RDONLY | os.O_CLOEXEC |
                            getattr(os, "O_NOFOLLOW", 0), dir_fd=directory_fd)
        replay_stat = os.fstat(replay_fd)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(replay_fd, 4096)
            if not chunk:
                break
            chunks.append(chunk)
        if (
            b"".join(chunks) != payload
            or stat.S_IMODE(replay_stat.st_mode) != 0o444
            or replay_stat.st_uid != os.geteuid()
            or replay_stat.st_nlink != 1
        ):
            raise RuntimeError("probe readback identity differs")
        finished = clock_ns()
        if type(finished) is not int or finished < started:
            raise RuntimeError("probe finish clock differs")
        duration = finished - started
    except BaseException as error:
        primary_error = error
    finally:
        cleanup_errors: list[str] = []
        for open_fd in (replay_fd, descriptor):
            if open_fd >= 0:
                try:
                    os.close(open_fd)
                except BaseException as error:
                    cleanup_errors.append(f"close:{type(error).__name__}:{error}")
        for basename in (final, staging):
            try:
                os.unlink(basename, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
            except BaseException as error:
                cleanup_errors.append(f"unlink:{basename}:{type(error).__name__}:{error}")
        try:
            os.fsync(directory_fd)
        except BaseException as error:
            cleanup_errors.append(f"dir-fsync:{type(error).__name__}:{error}")
        for basename in (final, staging):
            try:
                os.stat(basename, dir_fd=directory_fd, follow_symlinks=False)
                cleanup_errors.append(f"residual:{basename}")
            except FileNotFoundError:
                pass
            except BaseException as error:
                cleanup_errors.append(f"absence:{basename}:{type(error).__name__}:{error}")
        if primary_error is not None or cleanup_errors:
            detail = (
                f"probe failed: {type(primary_error).__name__}:{primary_error}"
                if primary_error is not None else "probe cleanup failed"
            )
            if cleanup_errors:
                detail += "; cleanup=" + "|".join(cleanup_errors)
            _fail(_SCHEDULING_ADMISSION_FAILURE_V2, detail, terminal=True)
    assert duration is not None
    return duration


def _retained_four_fsync_publication_probe_v6(
    directory_fd: int,
    unique_token: str,
    payload: bytes,
    clock_ns: Callable[[], int] = time.monotonic_ns,
) -> int:
    """Publish one immutable probe and retain its final name for the window."""

    if (
        type(directory_fd) is not int
        or directory_fd < 0
        or type(unique_token) is not str
        or re.fullmatch(r"w[0-9]{2}-r[0-9]{6}", unique_token) is None
        or type(payload) is not bytes
        or len(payload) != 4096
    ):
        _fail(
            _SCHEDULING_ADMISSION_FAILURE_V2,
            "retained probe inputs differ",
            terminal=True,
        )
    staging = f".q05b-admission-{unique_token}.staging-v6"
    final = _scheduling_sustained_final_name_v6(unique_token)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    replay_fd = -1
    started: int | None = None
    duration: int | None = None
    primary_error: BaseException | None = None
    cleanup_errors: list[str] = []
    try:
        started = clock_ns()
        if type(started) is not int or started < 0:
            raise RuntimeError("retained probe start clock differs")
        descriptor = os.open(staging, flags, 0o600, dir_fd=directory_fd)
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise OSError("retained probe write made no progress")
            offset += written
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.fsync(directory_fd)
        os.link(
            staging,
            final,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
        os.fsync(directory_fd)
        os.unlink(staging, dir_fd=directory_fd)
        os.fsync(directory_fd)
        replay_fd = os.open(
            final,
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
        replay_stat = os.fstat(replay_fd)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(replay_fd, 4096)
            if not chunk:
                break
            chunks.append(chunk)
        if (
            b"".join(chunks) != payload
            or not stat.S_ISREG(replay_stat.st_mode)
            or stat.S_IMODE(replay_stat.st_mode) != 0o444
            or replay_stat.st_uid != os.geteuid()
            or replay_stat.st_nlink != 1
            or replay_stat.st_size != 4096
        ):
            raise RuntimeError("retained probe readback identity differs")
        os.close(replay_fd)
        replay_fd = -1
        try:
            os.stat(staging, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise RuntimeError("retained probe staging absence differs")
        finished = clock_ns()
        if type(finished) is not int or finished < started:
            raise RuntimeError("retained probe finish clock differs")
        duration = finished - started
    except BaseException as error:
        primary_error = error
    finally:
        for open_fd in (replay_fd, descriptor):
            if open_fd >= 0:
                try:
                    os.close(open_fd)
                except BaseException as error:
                    cleanup_errors.append(
                        f"close:{type(error).__name__}:{error}"
                    )
        if primary_error is not None or cleanup_errors:
            namespace_mutated = False
            for basename in (staging, final):
                try:
                    os.unlink(basename, dir_fd=directory_fd)
                    namespace_mutated = True
                except FileNotFoundError:
                    pass
                except BaseException as error:
                    cleanup_errors.append(
                        f"unlink:{basename}:{type(error).__name__}:{error}"
                    )
            if namespace_mutated:
                try:
                    os.fsync(directory_fd)
                except BaseException as error:
                    cleanup_errors.append(
                        f"cleanup-dir-fsync:{type(error).__name__}:{error}"
                    )
            for basename in (staging, final):
                try:
                    os.stat(
                        basename,
                        dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                    cleanup_errors.append(f"residual:{basename}")
                except FileNotFoundError:
                    pass
                except BaseException as error:
                    cleanup_errors.append(
                        f"absence:{basename}:{type(error).__name__}:{error}"
                    )
            detail = (
                f"retained probe failed: {type(primary_error).__name__}:"
                f"{primary_error}"
                if primary_error is not None
                else "retained probe cleanup failed"
            )
            if cleanup_errors:
                detail += "; cleanup=" + "|".join(cleanup_errors)
            _fail(_SCHEDULING_ADMISSION_FAILURE_V2, detail, terminal=True)
    assert started is not None and duration is not None
    return duration


def _uint64_exact(value: object) -> bool:
    return type(value) is int and 0 <= value < 2**64


def _scheduling_sustained_evidence_exact_v6(
    policy: Mapping[str, object],
    slot_id: str,
    window: int,
    value: object,
) -> bool:
    fields = {
        "schema_version", "slot_id", "window_ordinal", "clock_id",
        "measurement_start_ns", "measurement_deadline_ns",
        "measurement_finish_ns", "measurement_duration_ns",
        "offered_rate_records_per_second", "cycle_interval_ns",
        "records_per_cycle", "cycle_count",
        "prefill_records_durable", "prefill_publication_duration_ns_total",
        "prefill_completed_before_measurement",
        "timed_records_enqueued", "timed_records_durable_by_deadline",
        "timed_records_durable_after_deadline", "records_durable_total",
        "retained_records_at_measurement_end", "queue_start_depth_records",
        "queue_end_depth_records", "queue_high_watermark_records",
        "dropped_record_count", "duplicate_record_count",
        "reordered_record_count", "last_timed_durable_finish_ns_or_null",
        "terminal_drain_ns", "sampler_anchor_count",
        "maximum_sampler_release_to_post_enqueue_anchor_ns",
        "epoch_aggregates_in_strict_order",
        "prefill_chain_digest", "final_chain_digest",
        "cleanup_removed_records", "cleanup_absence_passed", "passed",
    }
    if type(value) is not dict or set(value) != fields:
        return False
    exact_policy_names = (
        "sustained_offered_rate_records_per_second",
        "sustained_measurement_duration_ns",
        "sustained_epoch_duration_ns",
        "sustained_prefill_retained_records",
        "sustained_timed_records",
        "prior_failed_trajectory_event_count",
        "minimum_final_retained_records",
        "sustained_queue_target_records",
        "maximum_sustained_queue_high_watermark_records",
        "maximum_sustained_terminal_drain_ns",
        "minimum_sustained_publication_rate_records_per_second",
        "maximum_sampler_deadline_to_post_enqueue_anchor_ns",
        "sampling_target_wait_ns", "sustained_records_per_cycle",
        "sustained_cycle_count",
    )
    if any(type(policy.get(name)) is not int or int(policy[name]) <= 0
           for name in exact_policy_names):
        return False
    offered = int(policy["sustained_offered_rate_records_per_second"])
    duration_policy = int(policy["sustained_measurement_duration_ns"])
    epoch_duration = int(policy["sustained_epoch_duration_ns"])
    prefill_policy = int(policy["sustained_prefill_retained_records"])
    timed_policy = int(policy["sustained_timed_records"])
    queue_target = int(policy["sustained_queue_target_records"])
    records_per_cycle = int(policy["sustained_records_per_cycle"])
    cycle_count = int(policy["sustained_cycle_count"])
    cycle_interval = int(policy["sampling_target_wait_ns"])
    if (
        value.get("schema_version") != _SCHEDULING_SUSTAINED_EVIDENCE_SCHEMA_V6
        or value.get("slot_id") != slot_id
        or type(window) is not int
        or value.get("window_ordinal") != window
        or value.get("clock_id") != "CLOCK_MONOTONIC"
        or 1_000_000_000 % offered != 0
        or duration_policy % epoch_duration != 0
        or offered * duration_policy // 1_000_000_000 != timed_policy
        or records_per_cycle * cycle_count != timed_policy
        or cycle_count * cycle_interval != duration_policy
        or records_per_cycle * 1_000_000_000 // cycle_interval != offered
        or prefill_policy + timed_policy
        != int(policy["minimum_final_retained_records"])
    ):
        return False
    uint_names = (
        "measurement_start_ns", "measurement_deadline_ns",
        "measurement_finish_ns", "measurement_duration_ns",
        "offered_rate_records_per_second", "cycle_interval_ns",
        "records_per_cycle", "cycle_count",
        "prefill_records_durable", "prefill_publication_duration_ns_total",
        "timed_records_enqueued", "timed_records_durable_by_deadline",
        "timed_records_durable_after_deadline", "records_durable_total",
        "retained_records_at_measurement_end", "queue_start_depth_records",
        "queue_end_depth_records", "queue_high_watermark_records",
        "dropped_record_count", "duplicate_record_count",
        "reordered_record_count", "terminal_drain_ns", "sampler_anchor_count",
        "maximum_sampler_release_to_post_enqueue_anchor_ns",
        "cleanup_removed_records",
    )
    if any(not _uint64_exact(value.get(name)) for name in uint_names):
        return False
    start = int(value["measurement_start_ns"])
    deadline = int(value["measurement_deadline_ns"])
    finish = int(value["measurement_finish_ns"])
    timed_enqueued = int(value["timed_records_enqueued"])
    timed_by_deadline = int(value["timed_records_durable_by_deadline"])
    timed_after_deadline = int(value["timed_records_durable_after_deadline"])
    durable_total = int(value["records_durable_total"])
    last_finish = value.get("last_timed_durable_finish_ns_or_null")
    if (
        value["measurement_duration_ns"] != duration_policy
        or deadline != start + duration_policy
        or finish < deadline
        or value["offered_rate_records_per_second"] != offered
        or value["cycle_interval_ns"] != cycle_interval
        or value["records_per_cycle"] != records_per_cycle
        or value["cycle_count"] != cycle_count
        or value["prefill_records_durable"] != prefill_policy
        or value.get("prefill_completed_before_measurement") is not True
        or not 0 <= timed_enqueued <= timed_policy
        or timed_by_deadline + timed_after_deadline != timed_enqueued
        or durable_total != prefill_policy + timed_enqueued
        or value["retained_records_at_measurement_end"] != durable_total
        or value["queue_start_depth_records"] != records_per_cycle
        or value["queue_end_depth_records"] != 0
        or not records_per_cycle
        <= value["queue_high_watermark_records"]
        <= queue_target
        or timed_enqueued % records_per_cycle != 0
        or value["dropped_record_count"] != timed_policy - timed_enqueued
        or value["duplicate_record_count"] != 0
        or value["reordered_record_count"] != 0
        or value["cleanup_removed_records"] != durable_total
        or value.get("cleanup_absence_passed") is not True
        or type(value.get("passed")) is not bool
    ):
        return False
    if timed_enqueued == 0:
        if last_finish is not None or value["terminal_drain_ns"] != 0:
            return False
    elif (
        not _uint64_exact(last_finish)
        or int(last_finish) < start
        or int(last_finish) > finish
        or value["terminal_drain_ns"] != max(0, int(last_finish) - deadline)
    ):
        return False
    epochs = value.get("epoch_aggregates_in_strict_order")
    epoch_count = duration_policy // epoch_duration
    if type(epochs) is not list or len(epochs) != epoch_count:
        return False
    cumulative = 0
    checkpoints = {prefill_policy, durable_total}
    checkpoint_for_epoch: list[int] = []
    for ordinal, row in enumerate(epochs, start=1):
        if type(row) is not dict or set(row) != {
            "epoch_ordinal", "start_offset_ns", "finish_offset_ns",
            "records_durable", "first_global_ordinal_or_null",
            "last_global_ordinal_or_null", "total_publication_duration_ns",
            "minimum_publication_duration_ns_or_null",
            "maximum_publication_duration_ns_or_null",
            "cumulative_timed_records_durable", "sampler_anchor_count",
            "maximum_sampler_release_to_post_enqueue_anchor_ns_or_null",
            "chain_digest",
        }:
            return False
        count = row.get("records_durable")
        if (
            type(row.get("epoch_ordinal")) is not int
            or row["epoch_ordinal"] != ordinal
            or row.get("start_offset_ns") != (ordinal - 1) * epoch_duration
            or row.get("finish_offset_ns") != ordinal * epoch_duration
            or not _uint64_exact(count)
            or not _uint64_exact(row.get("total_publication_duration_ns"))
            or not _uint64_exact(row.get("sampler_anchor_count"))
            or type(row.get("cumulative_timed_records_durable")) is not int
        ):
            return False
        count = int(count)
        previous_cumulative = cumulative
        cumulative += count
        if row["cumulative_timed_records_durable"] != cumulative:
            return False
        first = row.get("first_global_ordinal_or_null")
        last = row.get("last_global_ordinal_or_null")
        minimum = row.get("minimum_publication_duration_ns_or_null")
        maximum = row.get("maximum_publication_duration_ns_or_null")
        if count == 0:
            if any(item is not None for item in (first, last, minimum, maximum)):
                return False
        elif (
            first != prefill_policy + previous_cumulative + 1
            or last != prefill_policy + cumulative
            or not _uint64_exact(minimum)
            or not _uint64_exact(maximum)
            or int(minimum) > int(maximum)
            or not int(minimum) * count
            <= int(row["total_publication_duration_ns"])
            <= int(maximum) * count
        ):
            return False
        sampler_count = int(row["sampler_anchor_count"])
        sampler_maximum = row.get(
            "maximum_sampler_release_to_post_enqueue_anchor_ns_or_null"
        )
        if (
            (sampler_count == 0 and sampler_maximum is not None)
            or (sampler_count > 0 and not _uint64_exact(sampler_maximum))
        ):
            return False
        checkpoint = prefill_policy + cumulative
        checkpoints.add(checkpoint)
        checkpoint_for_epoch.append(checkpoint)
        if (
            type(row.get("chain_digest")) is not str
            or _RAW_SHA256.fullmatch(str(row["chain_digest"])) is None
        ):
            return False
    if cumulative != timed_by_deadline:
        return False
    expected_sampler_anchors = sum(
        _scheduling_sustained_record_identity_v6(
            slot_id, window, prefill_policy + ordinal, prefill_policy
        )["operation"] == "LIVE_RESOURCE_SAMPLE_BEGIN"
        for ordinal in range(1, timed_enqueued + 1)
    )
    expected_sampler_anchors_by_epoch = [0] * epoch_count
    for cycle_index in range(timed_enqueued // records_per_cycle):
        epoch_index = min(
            (cycle_index * cycle_interval) // epoch_duration,
            epoch_count - 1,
        )
        expected_sampler_anchors_by_epoch[epoch_index] += 2
    sampler_anchor_total = sum(
        int(row["sampler_anchor_count"]) for row in epochs
    )
    sampler_epoch_maxima = [
        int(row[
            "maximum_sampler_release_to_post_enqueue_anchor_ns_or_null"
        ])
        for row in epochs
        if row[
            "maximum_sampler_release_to_post_enqueue_anchor_ns_or_null"
        ] is not None
    ]
    if (
        value["sampler_anchor_count"] != expected_sampler_anchors
        or sampler_anchor_total != expected_sampler_anchors
        or [int(row["sampler_anchor_count"]) for row in epochs]
        != expected_sampler_anchors_by_epoch
        or value["maximum_sampler_release_to_post_enqueue_anchor_ns"]
        != (max(sampler_epoch_maxima) if sampler_epoch_maxima else 0)
    ):
        return False
    try:
        expected_chains = _scheduling_sustained_chain_checkpoints_v6(
            slot_id, window, durable_total, prefill_policy, checkpoints
        )
    except BaseException:
        return False
    if (
        value.get("prefill_chain_digest") != expected_chains[prefill_policy]
        or value.get("final_chain_digest") != expected_chains[durable_total]
        or any(
            row["chain_digest"] != expected_chains[checkpoint]
            for row, checkpoint in zip(epochs, checkpoint_for_epoch)
        )
    ):
        return False
    passed = (
        timed_enqueued == timed_policy
        and timed_by_deadline == timed_policy
        and timed_after_deadline == 0
        and value["dropped_record_count"] == 0
        and value["duplicate_record_count"] == 0
        and value["reordered_record_count"] == 0
        and durable_total >= int(policy["minimum_final_retained_records"])
        and durable_total > int(policy["prior_failed_trajectory_event_count"])
        and timed_by_deadline * 1_000_000_000
        >= int(policy[
            "minimum_sustained_publication_rate_records_per_second"
        ]) * duration_policy
        and value["queue_high_watermark_records"]
        <= int(policy["maximum_sustained_queue_high_watermark_records"])
        and value["terminal_drain_ns"]
        <= int(policy["maximum_sustained_terminal_drain_ns"])
        and value["maximum_sampler_release_to_post_enqueue_anchor_ns"]
        <= int(policy[
            "maximum_sampler_deadline_to_post_enqueue_anchor_ns"
        ])
        and value["cleanup_absence_passed"] is True
    )
    return value["passed"] is passed


def _scheduling_admission_burst_exact_v6(
    policy: Mapping[str, object], slot_id: str, window: int, value: object,
) -> bool:
    if type(value) is not dict or set(value) != {
        "burst_ordinal", "release_deadline_ns",
        "enqueue_rows_in_global_order", "durable_rows_in_global_order",
        "sampler_rows_in_actor_order", "queue_start_depth", "queue_end_depth",
        "queue_high_watermark_records", "dropped_record_count",
        "duplicate_record_count", "reordered_record_count", "records_enqueued",
        "records_durable", "burst_start_ns", "burst_finish_ns", "burst_drain_ns",
        "passed",
    }:
        return False
    burst = value.get("burst_ordinal")
    release = value.get("release_deadline_ns")
    start = value.get("burst_start_ns")
    finish = value.get("burst_finish_ns")
    drain = value.get("burst_drain_ns")
    enqueued = value.get("enqueue_rows_in_global_order")
    durable = value.get("durable_rows_in_global_order")
    samplers = value.get("sampler_rows_in_actor_order")
    uint_fields = (
        "queue_start_depth", "queue_end_depth", "queue_high_watermark_records",
        "dropped_record_count", "duplicate_record_count",
        "reordered_record_count", "records_enqueued", "records_durable",
    )
    if (
        type(burst) is not int or not 1 <= burst <= int(policy["bursts_per_window"])
        or not all(_uint64_exact(item) for item in (release, start, finish, drain))
        or start != release or finish < start or drain != finish - start
        or any(not _uint64_exact(value.get(name)) for name in uint_fields)
        or type(enqueued) is not list or type(durable) is not list
        or type(samplers) is not list
    ):
        return False
    expected_events = {
        "CONTROLLER": dict(enumerate(_SCHEDULING_CONTROLLER_EVENTS_V6, start=1)),
        "PYTHON_ENDPOINT": {
            1: "LIVE_RESOURCE_SAMPLE_BEGIN",
            2: "LIVE_RESOURCE_SAMPLE_OK",
        },
        "RUST_ENDPOINT": {
            1: "LIVE_RESOURCE_SAMPLE_BEGIN",
            2: "LIVE_RESOURCE_SAMPLE_OK",
        },
    }
    seen_lane_events: set[tuple[str, int]] = set()
    hashes: list[str] = []
    if len(enqueued) != int(value["records_enqueued"]):
        return False
    for ordinal, row in enumerate(enqueued, start=1):
        if type(row) is not dict or set(row) != {
            "enqueue_ordinal", "lane_id", "lane_event_ordinal", "operation",
            "raw_sha256",
        }:
            return False
        lane = row.get("lane_id")
        lane_ordinal = row.get("lane_event_ordinal")
        operation = row.get("operation")
        if (
            type(row.get("enqueue_ordinal")) is not int
            or row.get("enqueue_ordinal") != ordinal
            or type(lane) is not str or lane not in expected_events
            or type(lane_ordinal) is not int
            or expected_events[lane].get(lane_ordinal) != operation
            or (lane, lane_ordinal) in seen_lane_events
            or type(row.get("raw_sha256")) is not str
            or _RAW_SHA256.fullmatch(str(row.get("raw_sha256"))) is None
        ):
            return False
        payload = _scheduling_probe_payload_v6(
            slot_id, window, burst, ordinal, lane, lane_ordinal, str(operation)
        )
        expected_hash = sha256(payload).hexdigest()
        if row["raw_sha256"] != expected_hash:
            return False
        seen_lane_events.add((lane, lane_ordinal))
        hashes.append(expected_hash)
    expected_lane_events = {
        (lane, lane_ordinal)
        for lane, events in expected_events.items()
        for lane_ordinal in events
    }
    if seen_lane_events != expected_lane_events:
        return False
    if len(durable) != int(value["records_durable"]):
        return False
    for ordinal, row in enumerate(durable, start=1):
        if type(row) is not dict or set(row) != {
            "durable_ordinal", "enqueue_ordinal", "duration_ns",
            "raw_sha256",
        }:
            return False
        if (
            type(row.get("durable_ordinal")) is not int
            or row.get("durable_ordinal") != ordinal
            or type(row.get("enqueue_ordinal")) is not int
            or row.get("enqueue_ordinal") != ordinal
            or not _uint64_exact(row.get("duration_ns"))
            or ordinal > len(hashes)
            or row.get("raw_sha256") != hashes[ordinal - 1]
        ):
            return False
    sampler_by_actor: dict[str, Mapping[str, object]] = {}
    if len(samplers) != 2:
        return False
    for actor, row in zip(_SCHEDULING_SAMPLER_ACTORS_V6, samplers):
        if type(row) is not dict or set(row) != {
            "actor_id", "enqueue_ordinal", "enqueue_start_ns",
            "post_enqueue_anchor_ns",
            "deadline_to_post_enqueue_anchor_ns", "projected_gap_ns", "passed",
        }:
            return False
        enqueue_ordinal = row.get("enqueue_ordinal")
        enqueue_start = row.get("enqueue_start_ns")
        anchor = row.get("post_enqueue_anchor_ns")
        critical = row.get("deadline_to_post_enqueue_anchor_ns")
        projected = row.get("projected_gap_ns")
        if (
            row.get("actor_id") != actor
            or type(enqueue_ordinal) is not int
            or not 1 <= enqueue_ordinal <= len(enqueued)
            or enqueued[enqueue_ordinal - 1].get("lane_id") != actor
            or not all(_uint64_exact(item) for item in (
                enqueue_start, anchor, critical, projected
            ))
            or enqueue_start < release or anchor < enqueue_start
            or critical != anchor - release
            or projected != int(policy["sampling_target_wait_ns"]) + critical
            or row.get("passed") is not (
                critical <= int(policy[
                    "maximum_sampler_deadline_to_post_enqueue_anchor_ns"
                ])
                and projected <= int(policy["maximum_projected_gap_ns"])
            )
        ):
            return False
        sampler_by_actor[actor] = row
    duplicates = len(enqueued) - len({
        (row["lane_id"], row["lane_event_ordinal"]) for row in enqueued
    })
    reordered = sum(
        row.get("enqueue_ordinal") != ordinal
        for ordinal, row in enumerate(durable, start=1)
    )
    passed = (
        value["queue_start_depth"] == 0
        and value["queue_end_depth"] == 0
        and value["queue_high_watermark_records"]
        <= int(policy["maximum_queue_high_watermark_records"])
        and value["dropped_record_count"] == 0
        and value["duplicate_record_count"] == duplicates == 0
        and value["reordered_record_count"] == reordered == 0
        and value["records_enqueued"] == int(policy["records_per_burst"])
        and value["records_durable"] == int(policy["records_per_burst"])
        and drain <= int(policy["maximum_burst_drain_ns"])
        and all(row["passed"] is True for row in sampler_by_actor.values())
    )
    return type(value.get("passed")) is bool and value["passed"] is passed


_SCHEDULING_BURST_EVENT_BY_CODE_V6: Final = {
    "C1": ("CONTROLLER", 1, "SUPERVISOR_STAGE_04_OK"),
    "C2": ("CONTROLLER", 2, "SUPERVISOR_STAGE_05_BEGIN"),
    "C3": ("CONTROLLER", 3, "HELD_ACTOR_CLOSE_BEGIN"),
    "P1": ("PYTHON_ENDPOINT", 1, "LIVE_RESOURCE_SAMPLE_BEGIN"),
    "P2": ("PYTHON_ENDPOINT", 2, "LIVE_RESOURCE_SAMPLE_OK"),
    "R1": ("RUST_ENDPOINT", 1, "LIVE_RESOURCE_SAMPLE_BEGIN"),
    "R2": ("RUST_ENDPOINT", 2, "LIVE_RESOURCE_SAMPLE_OK"),
}
_SCHEDULING_BURST_CODE_BY_EVENT_V6: Final = {
    value: key for key, value in _SCHEDULING_BURST_EVENT_BY_CODE_V6.items()
}
_SCHEDULING_BURST_CHAIN_GENESIS_V6: Final = sha256(
    b"hegel-phase3a-q05b-scheduling-critical-burst-chain-genesis/6"
).hexdigest()
_SCHEDULING_BURST_CHAIN_DOMAIN_V6: Final = (
    b"hegel-phase3a-q05b-scheduling-critical-burst-chain-step/6\x00"
)


def _scheduling_burst_chain_step_v6(
    previous_digest: str, enqueue_ordinal: int, raw_sha256: str
) -> str:
    if (
        type(previous_digest) is not str
        or _RAW_SHA256.fullmatch(previous_digest) is None
        or type(enqueue_ordinal) is not int
        or not 1 <= enqueue_ordinal < 2**64
        or type(raw_sha256) is not str
        or _RAW_SHA256.fullmatch(raw_sha256) is None
    ):
        _fail(
            _SCHEDULING_ADMISSION_FAILURE_V2,
            "critical burst chain inputs differ",
            terminal=True,
        )
    return sha256(
        _SCHEDULING_BURST_CHAIN_DOMAIN_V6
        + bytes.fromhex(previous_digest)
        + enqueue_ordinal.to_bytes(8, "big")
        + bytes.fromhex(raw_sha256)
    ).hexdigest()


def _compact_scheduling_admission_burst_v6(
    policy: Mapping[str, object],
    slot_id: str,
    window: int,
    value: Mapping[str, object],
) -> dict[str, object]:
    if not _scheduling_admission_burst_exact_v6(
        policy, slot_id, window, value
    ):
        _fail(
            _SCHEDULING_ADMISSION_FAILURE_V2,
            "critical burst cannot be compacted",
            terminal=True,
        )
    codes: list[str] = []
    chain = _SCHEDULING_BURST_CHAIN_GENESIS_V6
    for ordinal, row in enumerate(
        value["enqueue_rows_in_global_order"], start=1
    ):
        event = (
            str(row["lane_id"]),
            int(row["lane_event_ordinal"]),
            str(row["operation"]),
        )
        code = _SCHEDULING_BURST_CODE_BY_EVENT_V6.get(event)
        if code is None:
            _fail(
                _SCHEDULING_ADMISSION_FAILURE_V2,
                "critical burst event code differs",
                terminal=True,
            )
        codes.append(code)
        chain = _scheduling_burst_chain_step_v6(
            chain, ordinal, str(row["raw_sha256"])
        )
    durations = [
        int(row["duration_ns"])
        for row in value["durable_rows_in_global_order"]
    ]
    compact: dict[str, object] = {
        "burst_ordinal": value["burst_ordinal"],
        "release_deadline_ns": value["release_deadline_ns"],
        "enqueue_order_codes_in_global_order": codes,
        "payload_chain_digest": chain,
        "sampler_rows_in_actor_order": deepcopy(
            value["sampler_rows_in_actor_order"]
        ),
        "queue_start_depth": value["queue_start_depth"],
        "queue_end_depth": value["queue_end_depth"],
        "queue_high_watermark_records": value[
            "queue_high_watermark_records"
        ],
        "dropped_record_count": value["dropped_record_count"],
        "duplicate_record_count": value["duplicate_record_count"],
        "reordered_record_count": value["reordered_record_count"],
        "records_enqueued": value["records_enqueued"],
        "records_durable": value["records_durable"],
        "durable_duration_ns_total": sum(durations),
        "durable_duration_ns_minimum": min(durations),
        "durable_duration_ns_maximum": max(durations),
        "burst_start_ns": value["burst_start_ns"],
        "burst_finish_ns": value["burst_finish_ns"],
        "burst_drain_ns": value["burst_drain_ns"],
        "passed": value["passed"],
    }
    if not _scheduling_admission_compact_burst_exact_v6(
        policy, slot_id, window, compact
    ):
        _fail(
            _SCHEDULING_ADMISSION_FAILURE_V2,
            "compacted critical burst evidence differs",
            terminal=True,
        )
    return compact


def _scheduling_admission_compact_burst_exact_v6(
    policy: Mapping[str, object],
    slot_id: str,
    window: int,
    value: object,
) -> bool:
    fields = {
        "burst_ordinal", "release_deadline_ns",
        "enqueue_order_codes_in_global_order", "payload_chain_digest",
        "sampler_rows_in_actor_order", "queue_start_depth", "queue_end_depth",
        "queue_high_watermark_records", "dropped_record_count",
        "duplicate_record_count", "reordered_record_count", "records_enqueued",
        "records_durable", "durable_duration_ns_total",
        "durable_duration_ns_minimum", "durable_duration_ns_maximum",
        "burst_start_ns", "burst_finish_ns", "burst_drain_ns", "passed",
    }
    if type(value) is not dict or set(value) != fields:
        return False
    burst = value.get("burst_ordinal")
    release = value.get("release_deadline_ns")
    start = value.get("burst_start_ns")
    finish = value.get("burst_finish_ns")
    drain = value.get("burst_drain_ns")
    codes = value.get("enqueue_order_codes_in_global_order")
    samplers = value.get("sampler_rows_in_actor_order")
    uint_names = (
        "queue_start_depth", "queue_end_depth",
        "queue_high_watermark_records", "dropped_record_count",
        "duplicate_record_count", "reordered_record_count", "records_enqueued",
        "records_durable", "durable_duration_ns_total",
        "durable_duration_ns_minimum", "durable_duration_ns_maximum",
    )
    if (
        type(burst) is not int
        or not 1 <= burst <= int(policy["bursts_per_window"])
        or not all(_uint64_exact(item) for item in (release, start, finish, drain))
        or start != release
        or finish < start
        or drain != finish - start
        or any(not _uint64_exact(value.get(name)) for name in uint_names)
        or type(codes) is not list
        or len(codes) != int(policy["records_per_burst"])
        or any(type(code) is not str for code in codes)
        or set(codes) != set(_SCHEDULING_BURST_EVENT_BY_CODE_V6)
        or type(samplers) is not list
        or len(samplers) != 2
    ):
        return False
    positions = {code: index for index, code in enumerate(codes)}
    if not (
        positions["C1"] < positions["C2"] < positions["C3"]
        and positions["P1"] < positions["P2"]
        and positions["R1"] < positions["R2"]
    ):
        return False
    chain = _SCHEDULING_BURST_CHAIN_GENESIS_V6
    for ordinal, code in enumerate(codes, start=1):
        lane, lane_ordinal, operation = _SCHEDULING_BURST_EVENT_BY_CODE_V6[code]
        payload = _scheduling_probe_payload_v6(
            slot_id,
            window,
            int(burst),
            ordinal,
            lane,
            lane_ordinal,
            operation,
        )
        chain = _scheduling_burst_chain_step_v6(
            chain, ordinal, sha256(payload).hexdigest()
        )
    if value.get("payload_chain_digest") != chain:
        return False
    for actor, row in zip(_SCHEDULING_SAMPLER_ACTORS_V6, samplers):
        if type(row) is not dict or set(row) != {
            "actor_id", "enqueue_ordinal", "enqueue_start_ns",
            "post_enqueue_anchor_ns",
            "deadline_to_post_enqueue_anchor_ns", "projected_gap_ns", "passed",
        }:
            return False
        enqueue_ordinal = row.get("enqueue_ordinal")
        enqueue_start = row.get("enqueue_start_ns")
        anchor = row.get("post_enqueue_anchor_ns")
        critical = row.get("deadline_to_post_enqueue_anchor_ns")
        projected = row.get("projected_gap_ns")
        actor_code = "P1" if actor == "PYTHON_ENDPOINT" else "R1"
        if (
            row.get("actor_id") != actor
            or enqueue_ordinal != positions[actor_code] + 1
            or not all(_uint64_exact(item) for item in (
                enqueue_start, anchor, critical, projected
            ))
            or enqueue_start < release
            or anchor < enqueue_start
            or critical != anchor - release
            or projected != int(policy["sampling_target_wait_ns"]) + critical
            or row.get("passed") is not (
                critical <= int(policy[
                    "maximum_sampler_deadline_to_post_enqueue_anchor_ns"
                ])
                and projected <= int(policy["maximum_projected_gap_ns"])
            )
        ):
            return False
    minimum = int(value["durable_duration_ns_minimum"])
    maximum = int(value["durable_duration_ns_maximum"])
    total = int(value["durable_duration_ns_total"])
    records = int(policy["records_per_burst"])
    if minimum > maximum or not minimum * records <= total <= maximum * records:
        return False
    passed = (
        value["queue_start_depth"] == 0
        and value["queue_end_depth"] == 0
        and value["queue_high_watermark_records"]
        <= int(policy["maximum_queue_high_watermark_records"])
        and value["dropped_record_count"] == 0
        and value["duplicate_record_count"] == 0
        and value["reordered_record_count"] == 0
        and value["records_enqueued"] == records
        and value["records_durable"] == records
        and drain <= int(policy["maximum_burst_drain_ns"])
        and all(row["passed"] is True for row in samplers)
    )
    return type(value.get("passed")) is bool and value["passed"] is passed


def _scheduling_admission_evidence_semantics_v6(
    contract: Mapping[str, object], value: object, *, require_passed: bool | None,
) -> bool:
    try:
        policy = _scheduling_admission_policy_v2(contract)
    except BaseException:
        return False
    if type(value) is not dict or set(value) != {
        "schema_version", "slot_id", "policy", "windows_in_strict_order",
        "selected_window_ordinal", "passed",
    }:
        return False
    slots = contract.get("slot_registry")
    slot_id = value.get("slot_id")
    valid_slots = {
        row.get("slot_id") for row in slots if type(row) is dict
    } if type(slots) is list else set()
    rows = value.get("windows_in_strict_order")
    selected = value.get("selected_window_ordinal")
    passed = value.get("passed")
    if (
        value.get("schema_version") != _SCHEDULING_ADMISSION_EVIDENCE_SCHEMA_V6
        or type(slot_id) is not str or slot_id not in valid_slots
        or value.get("policy") != policy
        or type(passed) is not bool
        or (require_passed is not None and passed is not require_passed)
        or type(rows) is not list or not 1 <= len(rows) <= 3
        or (
            passed is True
            and (type(selected) is not int or selected != len(rows))
        )
        or (passed is False and (selected is not None or len(rows) != 3))
    ):
        return False
    previous_finish = -1
    for index, row in enumerate(rows, start=1):
        if type(row) is not dict or set(row) != {
            "window_ordinal", "quiet_start_ns", "quiet_finish_ns",
            "quiet_duration_ns", "bursts_in_strict_order",
            "sustained_publication_or_null",
            "maximum_sampler_deadline_to_post_enqueue_anchor_ns",
            "maximum_projected_gap_ns", "maximum_burst_drain_ns",
            "maximum_queue_high_watermark_records", "passed",
        }:
            return False
        quiet_start = row.get("quiet_start_ns")
        quiet_finish = row.get("quiet_finish_ns")
        quiet_duration = row.get("quiet_duration_ns")
        bursts = row.get("bursts_in_strict_order")
        maxima = (
            row.get("maximum_sampler_deadline_to_post_enqueue_anchor_ns"),
            row.get("maximum_projected_gap_ns"),
            row.get("maximum_burst_drain_ns"),
            row.get("maximum_queue_high_watermark_records"),
        )
        if (
            type(row.get("window_ordinal")) is not int
            or row.get("window_ordinal") != index
            or not all(_uint64_exact(item) for item in (
                quiet_start, quiet_finish, quiet_duration, *maxima
            ))
            or quiet_start < previous_finish or quiet_finish < quiet_start
            or quiet_duration != quiet_finish - quiet_start
            or quiet_duration < int(policy["quiet_period_ns"])
            or type(bursts) is not list
            or len(bursts) != int(policy["bursts_per_window"])
        ):
            return False
        previous_burst_finish = quiet_finish
        for burst_index, burst_row in enumerate(bursts, start=1):
            if (
                not _scheduling_admission_compact_burst_exact_v6(
                    policy, slot_id, index, burst_row
                )
                or burst_row["burst_ordinal"] != burst_index
                or burst_row["burst_start_ns"] < previous_burst_finish
            ):
                return False
            previous_burst_finish = burst_row["burst_finish_ns"]
        sampler_rows = [
            sample
            for burst_row in bursts
            for sample in burst_row["sampler_rows_in_actor_order"]
        ]
        expected_maxima = (
            max(sample["deadline_to_post_enqueue_anchor_ns"] for sample in sampler_rows),
            max(sample["projected_gap_ns"] for sample in sampler_rows),
            max(burst_row["burst_drain_ns"] for burst_row in bursts),
            max(burst_row["queue_high_watermark_records"] for burst_row in bursts),
        )
        critical_passed = all(
            burst_row["passed"] is True for burst_row in bursts
        )
        sustained = row.get("sustained_publication_or_null")
        sustained_exact = (
            type(sustained) is dict
            and _scheduling_sustained_evidence_exact_v6(
                policy, str(slot_id), index, sustained
            )
            and sustained["measurement_start_ns"] >= previous_burst_finish
        )
        window_passed = (
            critical_passed
            and sustained_exact
            and sustained["passed"] is True
        )
        if (
            maxima != expected_maxima
            or (critical_passed is False and sustained is not None)
            or (critical_passed is True and not sustained_exact)
            or type(row.get("passed")) is not bool
            or row["passed"] is not window_passed
            or (index < len(rows) and row["passed"] is not False)
            or (passed is True and index == len(rows) and row["passed"] is not True)
            or (passed is False and row["passed"] is not False)
        ):
            return False
        previous_finish = (
            int(sustained["measurement_finish_ns"])
            if sustained_exact
            else previous_burst_finish
        )
    return True


def _scheduling_admission_evidence_exact_v2(
    contract: Mapping[str, object], value: object
) -> bool:
    return _scheduling_admission_evidence_semantics_v6(
        contract, value, require_passed=True
    )


def _scheduling_admission_failure_detail_v6(value: object) -> str | None:
    if type(value) is not dict:
        return None
    slot_id = value.get("slot_id")
    rows = value.get("windows_in_strict_order")
    if (
        type(slot_id) is not str
        or _SAFE_NAME.fullmatch(slot_id) is None
        or type(rows) is not list
        or not rows
    ):
        return None
    maxima: list[dict[str, int]] = []
    for row in rows:
        if type(row) is not dict:
            return None
        values = {
            "critical_ns": row.get(
                "maximum_sampler_deadline_to_post_enqueue_anchor_ns"
            ),
            "projected_ns": row.get("maximum_projected_gap_ns"),
            "drain_ns": row.get("maximum_burst_drain_ns"),
            "high_watermark": row.get("maximum_queue_high_watermark_records"),
        }
        if any(not _uint64_exact(item) for item in values.values()):
            return None
        summary = {key: int(item) for key, item in values.items()}
        sustained = row.get("sustained_publication_or_null")
        if sustained is not None:
            if type(sustained) is not dict:
                return None
            for source, target in (
                ("timed_records_durable_by_deadline", "sustained_by_deadline"),
                ("timed_records_durable_after_deadline", "sustained_after_deadline"),
                ("retained_records_at_measurement_end", "retained_cardinality"),
                ("queue_high_watermark_records", "sustained_high_watermark"),
                ("terminal_drain_ns", "sustained_terminal_drain_ns"),
            ):
                item = sustained.get(source)
                if not _uint64_exact(item):
                    return None
                summary[target] = int(item)
        maxima.append(summary)
    return (
        f"{slot_id} admission exhausted before consume/Popen; "
        f"window_maxima={maxima}; observed admission is not a runtime "
        "scheduling, durability, or 250ms gap guarantee"
    )


def _run_stage5_scheduling_admission_burst_v6(
    directory_fd: int,
    slot_id: str,
    window: int,
    burst: int,
    policy: Mapping[str, object],
    *,
    clock_ns: Callable[[], int] = time.monotonic_ns,
    probe_publisher: Callable[[int, str, bytes, Callable[[], int]], int] =
        _four_fsync_publication_probe_v6,
) -> dict[str, object]:
    queue: deque[dict[str, object]] = deque()
    condition = threading.Condition(threading.Lock())
    release = threading.Event()
    ready_count = 0
    producers_done = 0
    dropped = 0
    queue_high_watermark = 0
    enqueue_rows: list[dict[str, object]] = []
    durable_rows: list[dict[str, object]] = []
    sampler_rows_by_actor: dict[str, dict[str, object]] = {}
    producer_errors: list[BaseException] = []
    publisher_errors: list[BaseException] = []
    release_deadline_ns = -1
    accepting = True
    shutdown_requested = False

    def producer(lane_id: str, operations: Sequence[str]) -> None:
        nonlocal ready_count, producers_done, dropped, queue_high_watermark
        try:
            with condition:
                ready_count += 1
                condition.notify_all()
            release.wait()
            with condition:
                if not accepting:
                    return
            for lane_event_ordinal, operation in enumerate(operations, start=1):
                enqueue_start = clock_ns()
                with condition:
                    if len(queue) >= int(policy["queue_capacity_records"]):
                        dropped += 1
                        continue
                    enqueue_ordinal = len(enqueue_rows) + 1
                    payload = _scheduling_probe_payload_v6(
                        slot_id, window, burst, enqueue_ordinal, lane_id,
                        lane_event_ordinal, operation,
                    )
                    payload_hash = sha256(payload).hexdigest()
                    item = {
                        "enqueue_ordinal": enqueue_ordinal,
                        "lane_id": lane_id,
                        "lane_event_ordinal": lane_event_ordinal,
                        "operation": operation,
                        "raw_sha256": payload_hash,
                        "payload": payload,
                    }
                    enqueue_rows.append({
                        key: value for key, value in item.items() if key != "payload"
                    })
                    queue.append(item)
                    queue_high_watermark = max(queue_high_watermark, len(queue))
                    condition.notify_all()
                post_enqueue_anchor = clock_ns()
                if (
                    lane_id in _SCHEDULING_SAMPLER_ACTORS_V6
                    and operation == "LIVE_RESOURCE_SAMPLE_BEGIN"
                ):
                    critical = post_enqueue_anchor - release_deadline_ns
                    projected = int(policy["sampling_target_wait_ns"]) + critical
                    sampler_rows_by_actor[lane_id] = {
                        "actor_id": lane_id,
                        "enqueue_ordinal": enqueue_ordinal,
                        "enqueue_start_ns": enqueue_start,
                        "post_enqueue_anchor_ns": post_enqueue_anchor,
                        "deadline_to_post_enqueue_anchor_ns": critical,
                        "projected_gap_ns": projected,
                        "passed": (
                            critical <= int(policy[
                                "maximum_sampler_deadline_to_post_enqueue_anchor_ns"
                            ])
                            and projected <= int(policy["maximum_projected_gap_ns"])
                        ),
                    }
        except BaseException as error:
            with condition:
                producer_errors.append(error)
        finally:
            with condition:
                producers_done += 1
                condition.notify_all()

    def publisher() -> None:
        while True:
            with condition:
                condition.wait_for(lambda: bool(queue) or shutdown_requested)
                if not queue and shutdown_requested:
                    return
                item = queue.popleft()
            if publisher_errors:
                continue
            try:
                ordinal = int(item["enqueue_ordinal"])
                duration = probe_publisher(
                    directory_fd,
                    f"w{window:02d}-p{((burst - 1) * 7 + ordinal):03d}",
                    item["payload"],
                    clock_ns,
                )
                if not _uint64_exact(duration):
                    raise RuntimeError("publication duration differs")
                durable_rows.append({
                    "durable_ordinal": len(durable_rows) + 1,
                    "enqueue_ordinal": ordinal,
                    "duration_ns": duration,
                    "raw_sha256": item["raw_sha256"],
                })
            except BaseException as error:
                publisher_errors.append(error)

    threads = [
        threading.Thread(
            target=producer,
            args=("CONTROLLER", _SCHEDULING_CONTROLLER_EVENTS_V6),
            name=f"q05b-admission-controller-w{window}-b{burst}",
            daemon=True,
        ),
        *[
            threading.Thread(
                target=producer,
                args=(actor, (
                    "LIVE_RESOURCE_SAMPLE_BEGIN",
                    "LIVE_RESOURCE_SAMPLE_OK",
                )),
                name=f"q05b-admission-{actor.lower()}-w{window}-b{burst}",
                daemon=True,
            )
            for actor in _SCHEDULING_SAMPLER_ACTORS_V6
        ],
    ]
    worker = threading.Thread(
        target=publisher,
        name=f"q05b-admission-durability-w{window}-b{burst}",
        daemon=True,
    )
    started: list[threading.Thread] = []
    primary_error: BaseException | None = None
    cleanup_errors: list[str] = []
    terminal_drain_exceeded = False
    burst_start_ns = -1
    previous_signal_mask: set[signal.Signals] | None = None
    terminal_seconds = int(policy["terminal_drain_policy_ns"]) / 1_000_000_000
    drain_deadline_monotonic: float | None = None
    try:
        previous_signal_mask = signal.pthread_sigmask(
            signal.SIG_BLOCK,
            {signal.SIGINT},
        )
        started.append(worker)
        worker.start()
        for thread in threads:
            started.append(thread)
            thread.start()
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_signal_mask)
        previous_signal_mask = None
        with condition:
            if not condition.wait_for(lambda: ready_count == 3, timeout=terminal_seconds):
                raise RuntimeError("producer readiness exceeded terminal drain policy")
        release_deadline_ns = clock_ns()
        if not _uint64_exact(release_deadline_ns):
            raise RuntimeError("burst release clock differs")
        burst_start_ns = release_deadline_ns
        release.set()
        terminal_deadline = time.monotonic() + terminal_seconds
        drain_deadline_monotonic = terminal_deadline
        for thread in threads:
            thread.join(max(0.0, terminal_deadline - time.monotonic()))
        if any(thread.is_alive() for thread in threads):
            terminal_drain_exceeded = True
    except BaseException as error:
        primary_error = error
    finally:
        with condition:
            if primary_error is not None:
                accepting = False
            release.set()
        terminal_deadline = (
            drain_deadline_monotonic
            if drain_deadline_monotonic is not None
            else time.monotonic() + terminal_seconds
        )
        joinable = [thread for thread in started if thread.ident is not None]
        for thread in joinable:
            if thread is not worker:
                thread.join(max(0.0, terminal_deadline - time.monotonic()))
        with condition:
            shutdown_requested = True
            condition.notify_all()
        if worker in joinable:
            worker.join(max(0.0, terminal_deadline - time.monotonic()))
        live_threads = [thread for thread in joinable if thread.is_alive()]
        if live_threads:
            # No returning admission path may leave a thread behind.  If a
            # kernel durability syscall never returns this remains a fail-stop
            # boundary before consumption/Popen rather than inventing cleanup.
            terminal_drain_exceeded = True
            for thread in live_threads:
                thread.join()
        if previous_signal_mask is not None:
            try:
                signal.pthread_sigmask(
                    signal.SIG_SETMASK,
                    previous_signal_mask,
                )
            except BaseException as error:
                cleanup_errors.append(
                    f"signal-mask-restore:{type(error).__name__}:{error}"
                )
            previous_signal_mask = None
        try:
            _require_single_threaded_spawn_controller_v2()
        except BaseException as error:
            cleanup_errors.append(
                f"thread-proof:{type(error).__name__}:{error}"
            )
    errors = ([primary_error] if primary_error is not None else [])
    errors.extend(producer_errors)
    errors.extend(publisher_errors)
    if errors or cleanup_errors or terminal_drain_exceeded:
        error = errors[0] if errors else RuntimeError(
            "terminal drain policy was exceeded"
        )
        detail = (
            f"burst {window}:{burst} failed after zero-thread drain: "
            f"{type(error).__name__}:{error}"
        )
        if terminal_drain_exceeded:
            detail += "; terminal-drain-policy-exceeded"
        if cleanup_errors:
            detail += "; cleanup=" + "|".join(cleanup_errors)
        _fail(
            _SCHEDULING_ADMISSION_FAILURE_V2,
            detail,
            terminal=True,
        )
    burst_finish_ns = clock_ns()
    if not _uint64_exact(burst_finish_ns) or burst_finish_ns < burst_start_ns:
        _fail(_SCHEDULING_ADMISSION_FAILURE_V2, "burst finish clock differs", terminal=True)
    sampler_rows = [sampler_rows_by_actor[actor]
                    for actor in _SCHEDULING_SAMPLER_ACTORS_V6]
    reordered = sum(
        row["enqueue_ordinal"] != ordinal
        for ordinal, row in enumerate(durable_rows, start=1)
    )
    duplicates = len(enqueue_rows) - len({
        (row["lane_id"], row["lane_event_ordinal"]) for row in enqueue_rows
    })
    result: dict[str, object] = {
        "burst_ordinal": burst,
        "release_deadline_ns": release_deadline_ns,
        "enqueue_rows_in_global_order": enqueue_rows,
        "durable_rows_in_global_order": durable_rows,
        "sampler_rows_in_actor_order": sampler_rows,
        "queue_start_depth": 0,
        "queue_end_depth": len(queue),
        "queue_high_watermark_records": queue_high_watermark,
        "dropped_record_count": dropped,
        "duplicate_record_count": duplicates,
        "reordered_record_count": reordered,
        "records_enqueued": len(enqueue_rows),
        "records_durable": len(durable_rows),
        "burst_start_ns": burst_start_ns,
        "burst_finish_ns": burst_finish_ns,
        "burst_drain_ns": burst_finish_ns - burst_start_ns,
        "passed": False,
    }
    result["passed"] = (
        result["queue_end_depth"] == 0
        and queue_high_watermark
        <= int(policy["maximum_queue_high_watermark_records"])
        and dropped == duplicates == reordered == 0
        and len(enqueue_rows) == len(durable_rows)
        == int(policy["records_per_burst"])
        and result["burst_drain_ns"] <= int(policy["maximum_burst_drain_ns"])
        and all(row["passed"] is True for row in sampler_rows)
    )
    if not _scheduling_admission_burst_exact_v6(
        policy, slot_id, window, result
    ):
        _fail(_SCHEDULING_ADMISSION_FAILURE_V2, "burst evidence differs", terminal=True)
    return result


def _run_stage5_sustained_publication_admission_v6(
    directory_fd: int,
    slot_id: str,
    window: int,
    policy: Mapping[str, object],
    *,
    clock_ns: Callable[[], int] = time.monotonic_ns,
    sleeper: Callable[[float], None] = time.sleep,
    probe_publisher: Callable[[int, str, bytes, Callable[[], int]], int] =
        _retained_four_fsync_publication_probe_v6,
) -> dict[str, object]:
    """Offer 40 records/s for 120 s while retaining every immutable file."""

    if (
        type(directory_fd) is not int
        or directory_fd < 0
        or type(slot_id) is not str
        or _SAFE_NAME.fullmatch(slot_id) is None
        or type(window) is not int
        or not 1 <= window <= 3
        or not callable(clock_ns)
        or not callable(sleeper)
        or not callable(probe_publisher)
    ):
        _fail(
            _SCHEDULING_ADMISSION_FAILURE_V2,
            "sustained admission inputs differ",
            terminal=True,
        )
    required = (
        "sustained_offered_rate_records_per_second",
        "sustained_measurement_duration_ns",
        "sustained_epoch_duration_ns",
        "sustained_prefill_retained_records",
        "sustained_timed_records",
        "sustained_records_per_cycle",
        "sustained_cycle_count",
        "sampling_target_wait_ns",
        "prior_failed_trajectory_event_count",
        "minimum_final_retained_records",
        "minimum_sustained_publication_rate_records_per_second",
        "maximum_sustained_queue_high_watermark_records",
        "maximum_sampler_deadline_to_post_enqueue_anchor_ns",
        "sustained_queue_target_records",
        "maximum_sustained_terminal_drain_ns",
    )
    if any(type(policy.get(name)) is not int or int(policy[name]) <= 0
           for name in required):
        _fail(
            _SCHEDULING_ADMISSION_FAILURE_V2,
            "sustained admission policy inputs differ",
            terminal=True,
        )
    offered = int(policy["sustained_offered_rate_records_per_second"])
    measurement_duration_ns = int(policy["sustained_measurement_duration_ns"])
    epoch_duration_ns = int(policy["sustained_epoch_duration_ns"])
    prefill_records = int(policy["sustained_prefill_retained_records"])
    timed_records = int(policy["sustained_timed_records"])
    records_per_cycle = int(policy["sustained_records_per_cycle"])
    cycle_count = int(policy["sustained_cycle_count"])
    cycle_interval_ns = int(policy["sampling_target_wait_ns"])
    queue_target = int(policy["sustained_queue_target_records"])
    maximum_terminal_drain_ns = int(
        policy["maximum_sustained_terminal_drain_ns"]
    )
    if (
        1_000_000_000 % offered != 0
        or measurement_duration_ns % epoch_duration_ns != 0
        or offered * measurement_duration_ns // 1_000_000_000
        != timed_records
        or records_per_cycle != len(_SCHEDULING_SUSTAINED_OPERATIONS_V6)
        or records_per_cycle * cycle_count != timed_records
        or cycle_count * cycle_interval_ns != measurement_duration_ns
        or records_per_cycle * 1_000_000_000 // cycle_interval_ns != offered
        or not records_per_cycle <= queue_target <= timed_records
        or prefill_records + timed_records
        != int(policy["minimum_final_retained_records"])
        or int(policy["minimum_final_retained_records"])
        <= int(policy["prior_failed_trajectory_event_count"])
        or offered < int(policy[
            "minimum_sustained_publication_rate_records_per_second"
        ])
        or queue_target > int(policy[
            "maximum_sustained_queue_high_watermark_records"
        ])
        or prefill_records + timed_records > 999_999
    ):
        _fail(
            _SCHEDULING_ADMISSION_FAILURE_V2,
            "sustained admission rate topology differs",
            terminal=True,
        )
    epoch_count = measurement_duration_ns // epoch_duration_ns
    pending: deque[tuple[int, bytes, str]] = deque()
    condition = threading.Condition(threading.Lock())
    release = threading.Event()
    worker_ready = False
    producer_done = False
    publisher_done = False
    publisher_errors: list[BaseException] = []
    owned_tokens: list[str] = []
    chain_digest = _SCHEDULING_SUSTAINED_CHAIN_GENESIS_V6
    prefill_chain_digest = chain_digest
    prefill_duration_total = 0
    timed_enqueued = 0
    timed_durable = 0
    timed_durable_by_deadline = 0
    timed_durable_after_deadline = 0
    queue_high_watermark = 0
    duplicate_count = 0
    reordered_count = 0
    sampler_anchor_count = 0
    maximum_sampler_anchor_ns = 0
    seen_ordinals: set[int] = set()
    last_timed_finish_ns: int | None = None
    measurement_start_ns = -1
    measurement_deadline_ns = -1
    measurement_finish_ns = -1
    epoch_bins: list[dict[str, object]] = [
        {
            "records": 0,
            "first": None,
            "last": None,
            "total_duration": 0,
            "minimum_duration": None,
            "maximum_duration": None,
            "chain_digest": None,
        }
        for _ in range(epoch_count)
    ]
    sampler_anchor_bins: list[dict[str, object]] = [
        {"count": 0, "maximum": None} for _ in range(epoch_count)
    ]

    def item_for_timed_ordinal(
        timed_ordinal: int,
    ) -> tuple[int, bytes, str]:
        global_ordinal = prefill_records + timed_ordinal
        return (
            global_ordinal,
            _scheduling_sustained_probe_payload_v6(
                slot_id, window, global_ordinal, prefill_records
            ),
            _scheduling_sustained_record_digest_v6(
                slot_id, window, global_ordinal, prefill_records
            ),
        )

    def enqueue_timed_cycle(cycle_ordinal: int, due_ns: int) -> bool:
        nonlocal timed_enqueued, queue_high_watermark
        nonlocal sampler_anchor_count, maximum_sampler_anchor_ns
        first_timed_ordinal = (cycle_ordinal - 1) * records_per_cycle + 1
        items = [
            item_for_timed_ordinal(first_timed_ordinal + offset)
            for offset in range(records_per_cycle)
        ]
        with condition:
            if (
                publisher_errors
                or len(pending) + records_per_cycle > queue_target
            ):
                return False
            for item in items:
                pending.append(item)
                timed_enqueued += 1
                global_ordinal = item[0]
                identity = _scheduling_sustained_record_identity_v6(
                    slot_id, window, global_ordinal, prefill_records
                )
                if identity["operation"] == "LIVE_RESOURCE_SAMPLE_BEGIN":
                    post_enqueue_anchor_ns = clock_ns()
                    if (
                        not _uint64_exact(post_enqueue_anchor_ns)
                        or post_enqueue_anchor_ns < due_ns
                    ):
                        raise RuntimeError(
                            "sustained sampler enqueue anchor differs"
                        )
                    critical_ns = int(post_enqueue_anchor_ns) - due_ns
                    sampler_anchor_count += 1
                    maximum_sampler_anchor_ns = max(
                        maximum_sampler_anchor_ns, critical_ns
                    )
                    elapsed = min(
                        max(0, due_ns - measurement_start_ns),
                        measurement_duration_ns - 1,
                    )
                    epoch_index = min(
                        elapsed // epoch_duration_ns, epoch_count - 1
                    )
                    anchor_row = sampler_anchor_bins[epoch_index]
                    anchor_row["count"] = int(anchor_row["count"]) + 1
                    anchor_row["maximum"] = (
                        critical_ns
                        if anchor_row["maximum"] is None
                        else max(
                            int(anchor_row["maximum"]), critical_ns
                        )
                    )
            queue_high_watermark = max(queue_high_watermark, len(pending))
            condition.notify_all()
        return True

    def publisher() -> None:
        nonlocal worker_ready, publisher_done, chain_digest
        nonlocal timed_durable, timed_durable_by_deadline
        nonlocal timed_durable_after_deadline, duplicate_count
        nonlocal reordered_count, last_timed_finish_ns
        try:
            with condition:
                worker_ready = True
                condition.notify_all()
            release.wait()
            while True:
                with condition:
                    condition.wait_for(lambda: bool(pending) or producer_done)
                    if not pending and producer_done:
                        return
                    global_ordinal, payload, record_digest = pending.popleft()
                    condition.notify_all()
                token = _scheduling_sustained_probe_token_v6(
                    window, global_ordinal
                )
                duration = probe_publisher(
                    directory_fd, token, payload, clock_ns
                )
                finished = clock_ns()
                if not _uint64_exact(duration) or not _uint64_exact(finished):
                    raise RuntimeError(
                        "sustained publication duration or finish differs"
                    )
                owned_tokens.append(token)
                expected_ordinal = prefill_records + timed_durable + 1
                if global_ordinal in seen_ordinals:
                    duplicate_count += 1
                seen_ordinals.add(global_ordinal)
                if global_ordinal != expected_ordinal:
                    reordered_count += 1
                chain_digest = _scheduling_sustained_chain_step_v6(
                    chain_digest, global_ordinal, record_digest
                )
                timed_durable += 1
                last_timed_finish_ns = int(finished)
                if finished <= measurement_deadline_ns:
                    timed_durable_by_deadline += 1
                    elapsed = max(0, int(finished) - measurement_start_ns)
                    epoch_index = (
                        0 if elapsed == 0 else
                        min((elapsed - 1) // epoch_duration_ns, epoch_count - 1)
                    )
                    row = epoch_bins[epoch_index]
                    if row["records"] == 0:
                        row["first"] = global_ordinal
                        row["minimum_duration"] = int(duration)
                        row["maximum_duration"] = int(duration)
                    row["records"] = int(row["records"]) + 1
                    row["last"] = global_ordinal
                    row["total_duration"] = (
                        int(row["total_duration"]) + int(duration)
                    )
                    row["minimum_duration"] = min(
                        int(row["minimum_duration"]), int(duration)
                    )
                    row["maximum_duration"] = max(
                        int(row["maximum_duration"]), int(duration)
                    )
                    row["chain_digest"] = chain_digest
                else:
                    timed_durable_after_deadline += 1
        except BaseException as error:
            with condition:
                publisher_errors.append(error)
                pending.clear()
                condition.notify_all()
        finally:
            with condition:
                publisher_done = True
                condition.notify_all()

    queue_start_depth = records_per_cycle
    worker = threading.Thread(
        target=publisher,
        name=f"q05b-admission-sustained-w{window}",
        daemon=True,
    )
    primary_error: BaseException | None = None
    cleanup_errors: list[str] = []
    previous_signal_mask: set[signal.Signals] | None = None
    worker_started = False
    try:
        for global_ordinal in range(1, prefill_records + 1):
            token = _scheduling_sustained_probe_token_v6(
                window, global_ordinal
            )
            duration = probe_publisher(
                directory_fd,
                token,
                _scheduling_sustained_probe_payload_v6(
                    slot_id, window, global_ordinal, prefill_records
                ),
                clock_ns,
            )
            if not _uint64_exact(duration):
                raise RuntimeError("sustained prefill duration differs")
            owned_tokens.append(token)
            prefill_duration_total += int(duration)
            chain_digest = _scheduling_sustained_chain_step_v6(
                chain_digest,
                global_ordinal,
                _scheduling_sustained_record_digest_v6(
                    slot_id, window, global_ordinal, prefill_records
                ),
            )
        prefill_chain_digest = chain_digest
        previous_signal_mask = signal.pthread_sigmask(
            signal.SIG_BLOCK, {signal.SIGINT}
        )
        worker.start()
        worker_started = True
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_signal_mask)
        previous_signal_mask = None
        with condition:
            if not condition.wait_for(lambda: worker_ready, timeout=2.0):
                raise RuntimeError("sustained publisher readiness timed out")
        measurement_start_ns = clock_ns()
        if not _uint64_exact(measurement_start_ns):
            raise RuntimeError("sustained measurement start clock differs")
        measurement_deadline_ns = (
            measurement_start_ns + measurement_duration_ns
        )
        if measurement_deadline_ns >= 2**64:
            raise RuntimeError("sustained measurement deadline overflowed")
        if not enqueue_timed_cycle(1, measurement_start_ns):
            raise RuntimeError("sustained initial cycle was rejected")
        release.set()
        for cycle_ordinal in range(2, cycle_count + 1):
            due_ns = measurement_start_ns + (
                cycle_ordinal - 1
            ) * cycle_interval_ns
            now_ns = clock_ns()
            if not _uint64_exact(now_ns):
                raise RuntimeError("sustained producer clock differs")
            if now_ns < due_ns:
                sleeper((due_ns - now_ns) / 1_000_000_000)
            if not enqueue_timed_cycle(cycle_ordinal, due_ns):
                break
        with condition:
            producer_done = True
            condition.notify_all()
        now_ns = clock_ns()
        if not _uint64_exact(now_ns):
            raise RuntimeError("sustained completion clock differs")
        if now_ns < measurement_deadline_ns and not publisher_errors:
            sleeper(
                (measurement_deadline_ns - now_ns) / 1_000_000_000
            )
        terminal_deadline = time.monotonic() + (
            maximum_terminal_drain_ns / 1_000_000_000
        )
        worker.join(max(0.0, terminal_deadline - time.monotonic()))
        if worker.is_alive():
            raise RuntimeError("sustained terminal drain policy was exceeded")
        if publisher_errors:
            raise publisher_errors[0]
        measurement_finish_ns = clock_ns()
        if (
            not _uint64_exact(measurement_finish_ns)
            or measurement_finish_ns < measurement_deadline_ns
        ):
            raise RuntimeError("sustained measurement finish clock differs")
    except BaseException as error:
        primary_error = error
    finally:
        with condition:
            producer_done = True
            condition.notify_all()
        release.set()
        if worker.ident is not None and worker.is_alive():
            worker.join()
        if previous_signal_mask is not None:
            try:
                signal.pthread_sigmask(
                    signal.SIG_SETMASK, previous_signal_mask
                )
            except BaseException as error:
                cleanup_errors.append(
                    f"signal-mask-restore:{type(error).__name__}:{error}"
                )
        retained_count = 0
        for token in owned_tokens:
            final = _scheduling_sustained_final_name_v6(token)
            try:
                final_stat = os.stat(
                    final, dir_fd=directory_fd, follow_symlinks=False
                )
                if (
                    not stat.S_ISREG(final_stat.st_mode)
                    or stat.S_IMODE(final_stat.st_mode) != 0o444
                    or final_stat.st_uid != os.geteuid()
                    or final_stat.st_nlink != 1
                    or final_stat.st_size != 4096
                ):
                    cleanup_errors.append(f"retained-identity:{final}")
                else:
                    retained_count += 1
            except BaseException as error:
                cleanup_errors.append(
                    f"retained-stat:{final}:{type(error).__name__}:{error}"
                )
        removed_count = 0
        for token in reversed(owned_tokens):
            final = _scheduling_sustained_final_name_v6(token)
            staging = f".q05b-admission-{token}.staging-v6"
            try:
                os.unlink(final, dir_fd=directory_fd)
                removed_count += 1
            except BaseException as error:
                cleanup_errors.append(
                    f"unlink:{final}:{type(error).__name__}:{error}"
                )
            try:
                os.unlink(staging, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
            except BaseException as error:
                cleanup_errors.append(
                    f"unlink:{staging}:{type(error).__name__}:{error}"
                )
        try:
            os.fsync(directory_fd)
        except BaseException as error:
            cleanup_errors.append(f"dir-fsync:{type(error).__name__}:{error}")
        for token in owned_tokens:
            for basename in (
                _scheduling_sustained_final_name_v6(token),
                f".q05b-admission-{token}.staging-v6",
            ):
                try:
                    os.stat(
                        basename,
                        dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                    cleanup_errors.append(f"residual:{basename}")
                except FileNotFoundError:
                    pass
                except BaseException as error:
                    cleanup_errors.append(
                        f"absence:{basename}:{type(error).__name__}:{error}"
                    )
        try:
            _require_single_threaded_spawn_controller_v2()
        except BaseException as error:
            cleanup_errors.append(
                f"thread-proof:{type(error).__name__}:{error}"
            )
    if primary_error is not None or cleanup_errors:
        error = primary_error or RuntimeError("sustained cleanup failed")
        detail = (
            "sustained publication failed after zero-thread drain: "
            f"{type(error).__name__}:{error}"
        )
        if cleanup_errors:
            detail += "; cleanup=" + "|".join(cleanup_errors)
        _fail(
            _SCHEDULING_ADMISSION_FAILURE_V2,
            detail,
            terminal=True,
        )
    assert measurement_start_ns >= 0 and measurement_deadline_ns >= 0
    assert measurement_finish_ns >= measurement_deadline_ns
    cumulative = 0
    carried_chain = prefill_chain_digest
    epoch_rows: list[dict[str, object]] = []
    for epoch_ordinal, (source, anchor_source) in enumerate(
        zip(epoch_bins, sampler_anchor_bins), start=1
    ):
        count = int(source["records"])
        cumulative += count
        if source["chain_digest"] is not None:
            carried_chain = str(source["chain_digest"])
        epoch_rows.append({
            "epoch_ordinal": epoch_ordinal,
            "start_offset_ns": (epoch_ordinal - 1) * epoch_duration_ns,
            "finish_offset_ns": epoch_ordinal * epoch_duration_ns,
            "records_durable": count,
            "first_global_ordinal_or_null": source["first"],
            "last_global_ordinal_or_null": source["last"],
            "total_publication_duration_ns": int(source["total_duration"]),
            "minimum_publication_duration_ns_or_null": source[
                "minimum_duration"
            ],
            "maximum_publication_duration_ns_or_null": source[
                "maximum_duration"
            ],
            "cumulative_timed_records_durable": cumulative,
            "sampler_anchor_count": int(anchor_source["count"]),
            "maximum_sampler_release_to_post_enqueue_anchor_ns_or_null": (
                anchor_source["maximum"]
            ),
            "chain_digest": carried_chain,
        })
    dropped = timed_records - timed_enqueued
    terminal_drain_ns = (
        max(0, last_timed_finish_ns - measurement_deadline_ns)
        if last_timed_finish_ns is not None
        else 0
    )
    result: dict[str, object] = {
        "schema_version": _SCHEDULING_SUSTAINED_EVIDENCE_SCHEMA_V6,
        "slot_id": slot_id,
        "window_ordinal": window,
        "clock_id": "CLOCK_MONOTONIC",
        "measurement_start_ns": measurement_start_ns,
        "measurement_deadline_ns": measurement_deadline_ns,
        "measurement_finish_ns": measurement_finish_ns,
        "measurement_duration_ns": measurement_duration_ns,
        "offered_rate_records_per_second": offered,
        "cycle_interval_ns": cycle_interval_ns,
        "records_per_cycle": records_per_cycle,
        "cycle_count": cycle_count,
        "prefill_records_durable": prefill_records,
        "prefill_publication_duration_ns_total": prefill_duration_total,
        "prefill_completed_before_measurement": True,
        "timed_records_enqueued": timed_enqueued,
        "timed_records_durable_by_deadline": timed_durable_by_deadline,
        "timed_records_durable_after_deadline": timed_durable_after_deadline,
        "records_durable_total": prefill_records + timed_durable,
        "retained_records_at_measurement_end": retained_count,
        "queue_start_depth_records": queue_start_depth,
        "queue_end_depth_records": 0,
        "queue_high_watermark_records": queue_high_watermark,
        "dropped_record_count": dropped,
        "duplicate_record_count": duplicate_count,
        "reordered_record_count": reordered_count,
        "last_timed_durable_finish_ns_or_null": last_timed_finish_ns,
        "terminal_drain_ns": terminal_drain_ns,
        "sampler_anchor_count": sampler_anchor_count,
        "maximum_sampler_release_to_post_enqueue_anchor_ns": (
            maximum_sampler_anchor_ns
        ),
        "epoch_aggregates_in_strict_order": epoch_rows,
        "prefill_chain_digest": prefill_chain_digest,
        "final_chain_digest": chain_digest,
        "cleanup_removed_records": removed_count,
        "cleanup_absence_passed": True,
        "passed": False,
    }
    result["passed"] = (
        timed_enqueued == timed_records
        and timed_durable_by_deadline == timed_records
        and timed_durable_after_deadline == 0
        and dropped == duplicate_count == reordered_count == 0
        and result["records_durable_total"]
        >= int(policy["minimum_final_retained_records"])
        and result["records_durable_total"]
        > int(policy["prior_failed_trajectory_event_count"])
        and timed_durable_by_deadline * 1_000_000_000
        >= int(policy[
            "minimum_sustained_publication_rate_records_per_second"
        ]) * measurement_duration_ns
        and queue_high_watermark
        <= int(policy["maximum_sustained_queue_high_watermark_records"])
        and terminal_drain_ns <= maximum_terminal_drain_ns
        and maximum_sampler_anchor_ns
        <= int(policy[
            "maximum_sampler_deadline_to_post_enqueue_anchor_ns"
        ])
        and retained_count == removed_count
        == result["records_durable_total"]
    )
    if not _scheduling_sustained_evidence_exact_v6(
        policy, slot_id, window, result
    ):
        _fail(
            _SCHEDULING_ADMISSION_FAILURE_V2,
            "sustained publication evidence differs",
            terminal=True,
        )
    return result


def _run_scheduling_admission_gate_v2(
    contract: Mapping[str, object], evidence_fd: int, slot_id: str, *,
    clock_ns: Callable[[], int] = time.monotonic_ns,
    sleeper: Callable[[float], None] = time.sleep,
    burst_runner: Callable[..., dict[str, object]] =
        _run_stage5_scheduling_admission_burst_v6,
    sustained_runner: Callable[..., dict[str, object]] =
        _run_stage5_sustained_publication_admission_v6,
) -> dict[str, object]:
    policy = _scheduling_admission_policy_v2(contract)
    rows: list[dict[str, object]] = []
    quiet_ns = int(policy["quiet_period_ns"])
    for window in range(1, int(policy["maximum_windows"]) + 1):
        try:
            quiet_start = clock_ns()
            sleeper(quiet_ns / 1_000_000_000)
            quiet_finish = clock_ns()
        except ShadowRetryCampaignErrorV2:
            raise
        except BaseException as error:
            _fail(
                _SCHEDULING_ADMISSION_FAILURE_V2,
                f"quiet observation failed: {type(error).__name__}:{error}",
                terminal=True,
            )
        if type(quiet_start) is not int or quiet_start < 0:
            _fail(_SCHEDULING_ADMISSION_FAILURE_V2, "quiet clock differs", terminal=True)
        if (
            type(quiet_finish) is not int
            or not 0 <= quiet_start <= quiet_finish < 2**64
            or quiet_finish - quiet_start < quiet_ns
        ):
            _fail(_SCHEDULING_ADMISSION_FAILURE_V2, "quiet period was not observed", terminal=True)
        bursts: list[dict[str, object]] = []
        for burst in range(1, int(policy["bursts_per_window"]) + 1):
            try:
                burst_row = burst_runner(
                    evidence_fd, slot_id, window, burst, policy, clock_ns=clock_ns,
                )
            except ShadowRetryCampaignErrorV2:
                raise
            except BaseException as error:
                _fail(
                    _SCHEDULING_ADMISSION_FAILURE_V2,
                    f"burst {window}:{burst} failed: "
                    f"{type(error).__name__}:{error}",
                    terminal=True,
                )
            if not _scheduling_admission_burst_exact_v6(
                policy, slot_id, window, burst_row
            ) or burst_row.get("burst_ordinal") != burst:
                _fail(_SCHEDULING_ADMISSION_FAILURE_V2, "burst evidence differs", terminal=True)
            bursts.append(burst_row)
        sampler_rows = [
            sample for burst_row in bursts
            for sample in burst_row["sampler_rows_in_actor_order"]
        ]
        critical_passed = all(
            burst_row["passed"] is True for burst_row in bursts
        )
        sustained: dict[str, object] | None = None
        if critical_passed:
            try:
                sustained = sustained_runner(
                    evidence_fd,
                    slot_id,
                    window,
                    policy,
                    clock_ns=clock_ns,
                    sleeper=sleeper,
                )
            except ShadowRetryCampaignErrorV2:
                raise
            except BaseException as error:
                _fail(
                    _SCHEDULING_ADMISSION_FAILURE_V2,
                    f"sustained window {window} failed: "
                    f"{type(error).__name__}:{error}",
                    terminal=True,
                )
            if not _scheduling_sustained_evidence_exact_v6(
                policy, slot_id, window, sustained
            ):
                _fail(
                    _SCHEDULING_ADMISSION_FAILURE_V2,
                    "sustained publication evidence differs",
                    terminal=True,
                )
        passed = (
            critical_passed
            and sustained is not None
            and sustained["passed"] is True
        )
        compact_bursts = [
            _compact_scheduling_admission_burst_v6(
                policy, slot_id, window, burst_row
            )
            for burst_row in bursts
        ]
        rows.append({
            "window_ordinal": window, "quiet_start_ns": quiet_start,
            "quiet_finish_ns": quiet_finish,
            "quiet_duration_ns": quiet_finish - quiet_start,
            "bursts_in_strict_order": compact_bursts,
            "sustained_publication_or_null": sustained,
            "maximum_sampler_deadline_to_post_enqueue_anchor_ns": max(
                sample["deadline_to_post_enqueue_anchor_ns"] for sample in sampler_rows
            ),
            "maximum_projected_gap_ns": max(
                sample["projected_gap_ns"] for sample in sampler_rows
            ),
            "maximum_burst_drain_ns": max(
                burst_row["burst_drain_ns"] for burst_row in bursts
            ),
            "maximum_queue_high_watermark_records": max(
                burst_row["queue_high_watermark_records"] for burst_row in bursts
            ),
            "passed": passed,
        })
        if passed:
            result: dict[str, object] = {
                "schema_version": _SCHEDULING_ADMISSION_EVIDENCE_SCHEMA_V6,
                "slot_id": slot_id,
                "policy": deepcopy(policy),
                "windows_in_strict_order": rows,
                "selected_window_ordinal": window, "passed": True,
            }
            if not _scheduling_admission_evidence_exact_v2(contract, result):
                _fail(_SCHEDULING_ADMISSION_FAILURE_V2, "admission evidence differs", terminal=True)
            return result
    result = {
        "schema_version": _SCHEDULING_ADMISSION_EVIDENCE_SCHEMA_V6,
        "slot_id": slot_id,
        "policy": deepcopy(policy),
        "windows_in_strict_order": rows,
        "selected_window_ordinal": None,
        "passed": False,
    }
    if not _scheduling_admission_evidence_semantics_v6(
        contract, result, require_passed=False
    ):
        _fail(_SCHEDULING_ADMISSION_FAILURE_V2, "failed evidence differs", terminal=True)
    detail = _scheduling_admission_failure_detail_v6(result)
    if detail is None:
        _fail(
            _SCHEDULING_ADMISSION_FAILURE_V2,
            "failed admission detail reconstruction differs",
            terminal=True,
        )
    _fail(
        _SCHEDULING_ADMISSION_FAILURE_V2,
        detail,
        terminal=True,
        receipt=result,
    )


def _bounded_receipt_or_null(value: object) -> dict[str, object] | None:
    if type(value) is not dict:
        return None
    try:
        candidate = deepcopy(value)
        if len(_canonical_json_bytes(candidate)) <= _MAX_ERROR_RECEIPT_BYTES:
            return candidate
    except BaseException:
        return None
    return {
        "schema_version": _BOUNDED_RECEIPT_OMISSION_SCHEMA_V2,
        "status": "RECEIPT_OMITTED_OVER_BOUND",
    }


def _error_snapshot_v2(error: BaseException) -> dict[str, object]:
    return {
        "error_code": str(
            getattr(error, "code", type(error).__name__)
        )[:256],
        "error_detail": str(
            getattr(error, "detail", str(error))
        )[:4096],
        "error_terminal": getattr(error, "terminal", False) is True,
        "error_receipt_or_null": _bounded_receipt_or_null(
            getattr(error, "receipt", None)
        ),
    }


def _terminalization_failure_receipt_v2(
    original_error: BaseException,
    terminalization_error: BaseException,
) -> dict[str, object]:
    value = {
        "schema_version": (
            "hegel-phase3a-q05b-shadow-retry-terminalization-failure/2"
        ),
        "phase": "TERMINALIZATION_FAILED",
        "original_error": _error_snapshot_v2(original_error),
        "terminalization_error": _error_snapshot_v2(terminalization_error),
    }
    if len(_canonical_json_bytes(value)) > _MAX_ERROR_RECEIPT_BYTES:
        value["original_error"]["error_receipt_or_null"] = None
        value["terminalization_error"]["error_receipt_or_null"] = None
    return value


def _raise_postpublication_uncertain(
    label: str,
    error: BaseException | None = None,
    *,
    publication_basename_or_null: str | None = None,
) -> NoReturn:
    """Fail closed after an immutable helper has reported success."""

    if (
        isinstance(error, ShadowRetryCampaignErrorV2)
        and error.code == "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN"
    ):
        raise error
    if error is None:
        origin = "replay_or_identity_mismatch"
        origin_snapshot = None
    else:
        origin = (
            f"{getattr(error, 'code', type(error).__name__)}:"
            f"{getattr(error, 'detail', str(error))}"
        )[:4096]
        origin_snapshot = _error_snapshot_v2(error)
    receipt = {
        "schema_version": (
            "hegel-phase3a-q05b-shadow-retry-publication-uncertainty/2"
        ),
        "phase": "PUBLICATION_CONTINUATION_UNCERTAIN",
        "publication_label": str(label)[:256],
        "publication_basename_or_null": (
            publication_basename_or_null[:128]
            if type(publication_basename_or_null) is str
            else None
        ),
        "origin_error_or_null": origin_snapshot,
    }
    _fail(
        "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN",
        f"{label} uncertain after immutable publication helper success: {origin}",
        terminal=True,
        receipt=receipt,
    )


def _guard_publication_attempt_failure(
    label: str,
    basename: str,
    known_failure_code: str,
    error: BaseException,
) -> NoReturn:
    """Preserve proven pre-create failures; classify every other gap unknown."""

    if isinstance(error, ShadowRetryCampaignErrorV2):
        if error.code == "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN":
            raise error
        receipt = error.receipt
        if (
            error.code == known_failure_code
            and type(receipt) is dict
            and receipt.get("publication_outcome")
            == "KNOWN_NO_PUBLICATION"
        ):
            raise error
    _raise_postpublication_uncertain(
        label,
        error,
        publication_basename_or_null=basename,
    )


def _postpublication_directory_identity(
    path: Path,
    descriptor: int,
    label: str,
    expected: Mapping[str, object],
) -> dict[str, object]:
    try:
        observed = _verify_held_directory_path(path, descriptor, label)
    except BaseException as error:
        _raise_postpublication_uncertain(label, error)
    if observed != expected:
        _raise_postpublication_uncertain(label)
    return observed


def _record_digest(
    contract: Mapping[str, object],
    domain_key: str,
    record: Mapping[str, object],
    digest_field: str = "record_digest",
) -> dict[str, object]:
    result = deepcopy(dict(record))
    if digest_field not in result or result[digest_field] is not None:
        _fail("FAIL_CAMPAIGN_LEDGER", "record digest slot differs", terminal=True)
    unsigned = deepcopy(result)
    unsigned.pop(digest_field)
    domains = contract["hash_domains"]
    assert type(domains) is dict
    domain = domains.get(domain_key)
    if type(domain) is not str:
        _fail("FAIL_CAMPAIGN_CONTRACT", f"record domain missing: {domain_key}")
    result[digest_field] = _domain_digest(domain, unsigned)
    return result


def _executable_identity(path_text: str, label: str) -> dict[str, object]:
    path = Path(path_text)
    raw, status = _read_regular_nofollow(
        path,
        f"{label} executable",
        1 << 30,
        required_uid=0,
    )
    mode = stat.S_IMODE(status.st_mode)
    if mode & 0o111 == 0 or mode & 0o022 != 0:
        _fail(
            "FAIL_CAMPAIGN_SOURCE",
            f"{label} executable mode is not root-owned immutable-to-nonroot",
        )
    return {
        "path": path_text,
        "raw_sha256": sha256(raw).hexdigest(),
        "mode": status.st_mode,
        "uid": status.st_uid,
        "nlink": status.st_nlink,
        "size": status.st_size,
        "dev": status.st_dev,
        "ino": status.st_ino,
    }


def _path_nonoverlap(paths: Sequence[str]) -> None:
    normalized: list[Path] = []
    for item in paths:
        if type(item) is not str:
            _fail("FAIL_CAMPAIGN_CONTRACT", "campaign path type differs")
        path = Path(item)
        if not path.is_absolute() or os.path.normpath(item) != item:
            _fail("FAIL_CAMPAIGN_CONTRACT", f"campaign path differs: {item}")
        normalized.append(path)
    if len(set(normalized)) != len(normalized):
        _fail("FAIL_CAMPAIGN_CONTRACT", "campaign paths are not unique")
    for index, left in enumerate(normalized):
        for right in normalized[index + 1 :]:
            if left in right.parents or right in left.parents:
                _fail("FAIL_CAMPAIGN_CONTRACT", "campaign roots overlap by ancestry")


def _assert_existing_directory_roots_nonaliased(
    paths: Sequence[Path],
    label: str,
) -> dict[str, dict[str, int]]:
    rows: dict[str, dict[str, int]] = {}
    seen_inode: dict[tuple[int, int], str] = {}
    for path in paths:
        if not os.path.lexists(path):
            continue
        descriptor, status_value = _open_directory_nofollow(path, f"{label} {path}")
        try:
            mount_id = _fd_mount_id(descriptor, f"{label} {path}")
        finally:
            os.close(descriptor)
        identity = (status_value.st_dev, status_value.st_ino)
        if identity in seen_inode:
            _fail(
                "FAIL_CAMPAIGN_FILESYSTEM",
                f"{label} directory alias: {seen_inode[identity]} and {path}",
                terminal=True,
            )
        seen_inode[identity] = os.fspath(path)
        rows[os.fspath(path)] = {
            "dev": status_value.st_dev,
            "ino": status_value.st_ino,
            "mount_id": mount_id,
            "mode": status_value.st_mode,
            "uid": status_value.st_uid,
        }
    return rows


_CORE_MODULE: Any | None = None
_CORE_DIGEST: str | None = None


def _load_core(contract: Mapping[str, object]) -> Any:
    global _CORE_MODULE, _CORE_DIGEST
    binding = contract["campaign_source_binding"]
    assert type(binding) is dict
    raw, _status = _read_regular_nofollow(
        CORE_PATH,
        "campaign pure core",
        8 * 1024 * 1024,
    )
    digest = sha256(raw).hexdigest()
    if digest != binding["core_raw_sha256"]:
        _fail("FAIL_CAMPAIGN_SOURCE", "campaign pure core hash differs")
    if _CORE_MODULE is not None:
        if _CORE_DIGEST != digest:
            _fail("FAIL_CAMPAIGN_SOURCE", "loaded campaign core identity drifted")
        return _CORE_MODULE
    module = types.ModuleType("_hegel_q05b_shadow_retry_campaign_v2_core")
    module.__file__ = os.fspath(CORE_PATH)
    sys.modules[module.__name__] = module
    try:
        exec(compile(raw, os.fspath(CORE_PATH), "exec", dont_inherit=True), module.__dict__)
    except BaseException as error:
        sys.modules.pop(module.__name__, None)
        _fail("FAIL_CAMPAIGN_SOURCE", f"campaign core load failed: {type(error).__name__}:{error}")
    _CORE_MODULE = module
    _CORE_DIGEST = digest
    return module


def _source_binding_preimage(binding: Mapping[str, object]) -> dict[str, object]:
    result = deepcopy(dict(binding))
    if result.get("source_binding_digest") is None:
        result.pop("source_binding_digest")
    else:
        result.pop("source_binding_digest", None)
    return result


def _require_v2r8_instance_isolation_v2(
    contract: Mapping[str, object],
) -> None:
    protected = contract.get("protected_state_policy")
    expected_protected_fields = {
        "protected_roots_in_exact_order",
        "prior_v2_campaign_contract_id",
        "prior_v2_capsule_digest",
        "prior_v2_authorization_digest",
        "prior_v2_terminal_is_historical_nonresumable_evidence_only",
        "prior_v2_campaign_root_open_allowed",
        "prior_v2_campaign_root_access_allowed",
        "prior_v2_capsule_authorization_or_ledger_reuse_allowed",
        "prior_v2r1_campaign_contract_id",
        "prior_v2r1_contract_digest",
        "prior_v2r1_capsule_digest",
        "prior_v2r1_authorization_digest",
        "prior_v2r1_terminal_ledger_record_digest",
        "prior_v2r1_completion_ledger_record_digest",
        "prior_v2r1_terminal_is_historical_nonresumable_evidence_only",
        "prior_v2r1_campaign_root_open_allowed",
        "prior_v2r1_campaign_root_access_allowed",
        "prior_v2r1_capsule_authorization_or_ledger_reuse_allowed",
        "prior_v2r2_campaign_contract_id",
        "prior_v2r2_controller_commit",
        "prior_v2r2_controller_tree",
        "prior_v2r2_target_commit",
        "prior_v2r2_campaign_source_binding_digest",
        "prior_v2r2_contract_digest",
        "prior_v2r2_capsule_digest",
        "prior_v2r2_owner_confirmation_digest",
        "prior_v2r2_authorization_digest",
        "prior_v2r2_terminal_status",
        "prior_v2r2_terminal_body_digest",
        "prior_v2r2_terminal_ledger_record_digest",
        "prior_v2r2_completion_body_digest",
        "prior_v2r2_completion_ledger_record_digest",
        "prior_v2r2_supervisor_invocation_count",
        "prior_v2r2_each_slot_trajectory_event_count",
        "prior_v2r2_each_slot_retained_failure_rows",
        "prior_v2r2_each_slot_retained_cleanup_rows",
        "prior_v2r2_each_slot_unpartitioned_event_count",
        "prior_v2r2_terminal_is_historical_nonresumable_evidence_only",
        "prior_v2r2_campaign_root_open_allowed",
        "prior_v2r2_campaign_root_access_allowed",
        "prior_v2r2_capsule_authorization_or_ledger_reuse_allowed",
        "prior_v2r3_campaign_contract_id",
        "prior_v2r3_controller_commit",
        "prior_v2r3_controller_tree",
        "prior_v2r3_target_commit",
        "prior_v2r3_target_tree",
        "prior_v2r3_campaign_source_binding_digest",
        "prior_v2r3_contract_digest",
        "prior_v2r3_capsule_digest",
        "prior_v2r3_owner_confirmation_digest",
        "prior_v2r3_authorization_digest",
        "prior_v2r3_terminal_status",
        "prior_v2r3_terminal_body_digest",
        "prior_v2r3_terminal_ledger_record_digest",
        "prior_v2r3_completion_body_digest",
        "prior_v2r3_completion_ledger_record_digest",
        "prior_v2r3_supervisor_invocation_count",
        "prior_v2r3_ledger_record_count",
        "prior_v2r3_last_consumed_slot_id",
        "prior_v2r3_slot_01_decision",
        "prior_v2r3_slot_01_trajectory_status",
        "prior_v2r3_slot_01_trajectory_event_count",
        "prior_v2r3_slot_01_retained_failure_rows",
        "prior_v2r3_slot_01_retained_cleanup_rows",
        "prior_v2r3_slot_01_open_span_count",
        "prior_v2r3_slot_01_terminal_manifest_present",
        "prior_v2r3_slot_01_partition_certificate_present",
        "prior_v2r3_slot_01_unpartitioned_event_count_or_null",
        "prior_v2r3_slot_02_invoked",
        "prior_v2r3_slot_03_invoked",
        "prior_v2r3_terminal_is_historical_nonresumable_evidence_only",
        "prior_v2r3_campaign_root_open_allowed",
        "prior_v2r3_campaign_root_access_allowed",
        "prior_v2r3_capsule_authorization_or_ledger_reuse_allowed",
        *_PRIOR_V2R4_PROTECTED_POLICY_FACTS,
        *_PRIOR_V2R5_PROTECTED_POLICY_FACTS,
        *_PRIOR_V2R6_PROTECTED_POLICY_FACTS,
        *_PRIOR_V2R7_PROTECTED_POLICY_FACTS,
        "historical_attempt2_or_attempt3_path_open_allowed",
        "meta_or_production_path_open_allowed",
        "protected_state_mutation_allowed",
        "protected_root_hardlink_bind_mount_or_inode_alias_exclusion_proved",
        "protected_root_separation_evidence",
        "before_and_after_negative_alias_replay_scope",
        "operational_threat_boundary",
        "campaign_outputs_are_never_current_or_adopted",
    }
    fixed = contract.get("fixed_external_roots")
    authorization = contract.get("authorization_policy")
    ledger = contract.get("ledger_policy")
    slots = contract.get("slot_registry")
    if (
        contract.get("contract_id")
        != "hegel-phase3a-q05b-shadow-retry-campaign-v2r8"
        or type(protected) is not dict
        or set(protected) != expected_protected_fields
        or len(_EXPECTED_PROTECTED_ROOTS_V2R8) != 154
        or protected.get("protected_roots_in_exact_order")
        != list(_EXPECTED_PROTECTED_ROOTS_V2R8)
        or protected.get("prior_v2_campaign_contract_id")
        != _PRIOR_V2_CONTRACT_ID
        or protected.get("prior_v2_capsule_digest")
        != _PRIOR_V2_CAPSULE_DIGEST
        or protected.get("prior_v2_authorization_digest")
        != _PRIOR_V2_AUTHORIZATION_DIGEST
        or protected.get("prior_v2r1_campaign_contract_id")
        != _PRIOR_V2R1_CONTRACT_ID
        or protected.get("prior_v2r1_contract_digest")
        != _PRIOR_V2R1_CONTRACT_DIGEST
        or protected.get("prior_v2r1_capsule_digest")
        != _PRIOR_V2R1_CAPSULE_DIGEST
        or protected.get("prior_v2r1_authorization_digest")
        != _PRIOR_V2R1_AUTHORIZATION_DIGEST
        or protected.get("prior_v2r1_terminal_ledger_record_digest")
        != _PRIOR_V2R1_TERMINAL_LEDGER_RECORD_DIGEST
        or protected.get("prior_v2r1_completion_ledger_record_digest")
        != _PRIOR_V2R1_COMPLETION_LEDGER_RECORD_DIGEST
        or protected.get("prior_v2r2_campaign_contract_id")
        != _PRIOR_V2R2_CONTRACT_ID
        or protected.get("prior_v2r2_controller_commit")
        != _PRIOR_V2R2_CONTROLLER_COMMIT
        or protected.get("prior_v2r2_controller_tree")
        != _PRIOR_V2R2_CONTROLLER_TREE
        or protected.get("prior_v2r2_target_commit")
        != _PRIOR_V2R2_TARGET_COMMIT
        or protected.get("prior_v2r2_campaign_source_binding_digest")
        != _PRIOR_V2R2_CAMPAIGN_SOURCE_BINDING_DIGEST
        or protected.get("prior_v2r2_contract_digest")
        != _PRIOR_V2R2_CONTRACT_DIGEST
        or protected.get("prior_v2r2_capsule_digest")
        != _PRIOR_V2R2_CAPSULE_DIGEST
        or protected.get("prior_v2r2_owner_confirmation_digest")
        != _PRIOR_V2R2_OWNER_CONFIRMATION_DIGEST
        or protected.get("prior_v2r2_authorization_digest")
        != _PRIOR_V2R2_AUTHORIZATION_DIGEST
        or protected.get("prior_v2r2_terminal_status")
        != _PRIOR_V2R2_TERMINAL_STATUS
        or protected.get("prior_v2r2_terminal_body_digest")
        != _PRIOR_V2R2_TERMINAL_BODY_DIGEST
        or protected.get("prior_v2r2_terminal_ledger_record_digest")
        != _PRIOR_V2R2_TERMINAL_LEDGER_RECORD_DIGEST
        or protected.get("prior_v2r2_completion_body_digest")
        != _PRIOR_V2R2_COMPLETION_BODY_DIGEST
        or protected.get("prior_v2r2_completion_ledger_record_digest")
        != _PRIOR_V2R2_COMPLETION_LEDGER_RECORD_DIGEST
        or protected.get("prior_v2r2_supervisor_invocation_count") != 3
        or protected.get("prior_v2r2_each_slot_trajectory_event_count") != 42
        or protected.get("prior_v2r2_each_slot_retained_failure_rows") != 7
        or protected.get("prior_v2r2_each_slot_retained_cleanup_rows") != 8
        or protected.get("prior_v2r2_each_slot_unpartitioned_event_count") != 0
        or protected.get("prior_v2r3_campaign_contract_id")
        != _PRIOR_V2R3_CONTRACT_ID
        or protected.get("prior_v2r3_controller_commit")
        != _PRIOR_V2R3_CONTROLLER_COMMIT
        or protected.get("prior_v2r3_controller_tree")
        != _PRIOR_V2R3_CONTROLLER_TREE
        or protected.get("prior_v2r3_target_commit")
        != _PRIOR_V2R3_TARGET_COMMIT
        or protected.get("prior_v2r3_target_tree")
        != _PRIOR_V2R3_TARGET_TREE
        or protected.get("prior_v2r3_campaign_source_binding_digest")
        != _PRIOR_V2R3_CAMPAIGN_SOURCE_BINDING_DIGEST
        or protected.get("prior_v2r3_contract_digest")
        != _PRIOR_V2R3_CONTRACT_DIGEST
        or protected.get("prior_v2r3_capsule_digest")
        != _PRIOR_V2R3_CAPSULE_DIGEST
        or protected.get("prior_v2r3_owner_confirmation_digest")
        != _PRIOR_V2R3_OWNER_CONFIRMATION_DIGEST
        or protected.get("prior_v2r3_authorization_digest")
        != _PRIOR_V2R3_AUTHORIZATION_DIGEST
        or protected.get("prior_v2r3_terminal_status")
        != _PRIOR_V2R3_TERMINAL_STATUS
        or protected.get("prior_v2r3_terminal_body_digest")
        != _PRIOR_V2R3_TERMINAL_BODY_DIGEST
        or protected.get("prior_v2r3_terminal_ledger_record_digest")
        != _PRIOR_V2R3_TERMINAL_LEDGER_RECORD_DIGEST
        or protected.get("prior_v2r3_completion_body_digest")
        != _PRIOR_V2R3_COMPLETION_BODY_DIGEST
        or protected.get("prior_v2r3_completion_ledger_record_digest")
        != _PRIOR_V2R3_COMPLETION_LEDGER_RECORD_DIGEST
        or protected.get("prior_v2r3_supervisor_invocation_count") != 1
        or protected.get("prior_v2r3_ledger_record_count") != 8
        or protected.get("prior_v2r3_last_consumed_slot_id") != "SLOT_01"
        or protected.get("prior_v2r3_slot_01_decision")
        != "TERMINAL_NONRETRYABLE"
        or protected.get("prior_v2r3_slot_01_trajectory_status")
        != "INCOMPLETE_PREFIX"
        or protected.get("prior_v2r3_slot_01_trajectory_event_count") != 5_410
        or protected.get("prior_v2r3_slot_01_retained_failure_rows") != 0
        or protected.get("prior_v2r3_slot_01_retained_cleanup_rows") != 0
        or protected.get("prior_v2r3_slot_01_open_span_count") != 6
        or protected.get("prior_v2r3_slot_01_terminal_manifest_present")
        is not False
        or protected.get("prior_v2r3_slot_01_partition_certificate_present")
        is not False
        or protected.get(
            "prior_v2r3_slot_01_unpartitioned_event_count_or_null"
        )
        is not None
        or protected.get("prior_v2r3_slot_02_invoked") is not False
        or protected.get("prior_v2r3_slot_03_invoked") is not False
        or any(
            not _json_value_type_exact_v1(protected.get(field), expected)
            for field, expected in _PRIOR_V2R4_PROTECTED_POLICY_FACTS.items()
        )
        or any(
            not _json_value_type_exact_v1(protected.get(field), expected)
            for field, expected in _PRIOR_V2R5_PROTECTED_POLICY_FACTS.items()
        )
        or any(
            not _json_value_type_exact_v1(protected.get(field), expected)
            for field, expected in _PRIOR_V2R6_PROTECTED_POLICY_FACTS.items()
        )
        or any(
            not _json_value_type_exact_v1(protected.get(field), expected)
            for field, expected in _PRIOR_V2R7_PROTECTED_POLICY_FACTS.items()
        )
        or protected.get(
            "prior_v2_terminal_is_historical_nonresumable_evidence_only"
        )
        is not True
        or protected.get(
            "prior_v2r1_terminal_is_historical_nonresumable_evidence_only"
        )
        is not True
        or protected.get(
            "prior_v2r2_terminal_is_historical_nonresumable_evidence_only"
        )
        is not True
        or protected.get(
            "prior_v2r3_terminal_is_historical_nonresumable_evidence_only"
        )
        is not True
        or any(
            protected.get(field) is not False
            for field in (
                "prior_v2_campaign_root_open_allowed",
                "prior_v2_campaign_root_access_allowed",
                "prior_v2_capsule_authorization_or_ledger_reuse_allowed",
                "prior_v2r1_campaign_root_open_allowed",
                "prior_v2r1_campaign_root_access_allowed",
                "prior_v2r1_capsule_authorization_or_ledger_reuse_allowed",
                "prior_v2r2_campaign_root_open_allowed",
                "prior_v2r2_campaign_root_access_allowed",
                "prior_v2r2_capsule_authorization_or_ledger_reuse_allowed",
                "prior_v2r3_campaign_root_open_allowed",
                "prior_v2r3_campaign_root_access_allowed",
                "prior_v2r3_capsule_authorization_or_ledger_reuse_allowed",
                "prior_v2r4_campaign_root_open_allowed",
                "prior_v2r4_campaign_root_access_allowed",
                "prior_v2r4_capsule_authorization_or_ledger_reuse_allowed",
                "prior_v2r5_campaign_root_open_allowed",
                "prior_v2r5_campaign_root_access_allowed",
                "prior_v2r5_capsule_authorization_or_ledger_reuse_allowed",
                "prior_v2r6_campaign_root_open_allowed",
                "prior_v2r6_campaign_root_access_allowed",
                "prior_v2r6_capsule_authorization_or_ledger_reuse_allowed",
                "prior_v2r7_campaign_root_open_allowed",
                "prior_v2r7_campaign_root_access_allowed",
                "prior_v2r7_capsule_authorization_or_ledger_reuse_allowed",
            )
        )
        or type(fixed) is not dict
        or fixed.get("authorization_root") != f"{_V2R8_STATE_PREFIX}-authorization"
        or fixed.get("ledger_root") != f"{_V2R8_STATE_PREFIX}-ledger"
        or type(authorization) is not dict
        or authorization.get("authorization_root")
        != f"{_V2R8_STATE_PREFIX}-authorization"
        or authorization.get("future_confirmation_must_bind_in_exact_order")
        != [
            "contract_id",
            "contract_digest",
            "target_supervisor_commit",
            "target_source_binding_digest",
            "campaign_source_binding_digest",
            "capsule_digest",
            "python310_dependency_preflight_receipt_digest",
            "authorization_root",
            "ledger_root",
            "three_exact_slot_path_records",
            "maximum_supervisor_invocations_3",
            "strictly_sequential_one_shot_slots",
            "exact_retry_predicate_digest",
            "success_predicate_digest",
            "terminal_no_resume_policy",
            "forbidden_scope_digest",
        ]
        or type(ledger) is not dict
        or ledger.get("ledger_root") != f"{_V2R8_STATE_PREFIX}-ledger"
        or type(slots) is not list
        or len(slots) != 3
    ):
        _fail(
            "FAIL_CAMPAIGN_CONTRACT",
            "v2r8 instance or historical protected-state isolation differs",
        )
    for ordinal, slot in enumerate(slots, start=1):
        if type(slot) is not dict:
            _fail("FAIL_CAMPAIGN_CONTRACT", "v2r8 slot isolation differs")
        prefix = f"{_V2R8_STATE_PREFIX}-slot-{ordinal:02d}"
        clone = f"{prefix}-clone"
        evidence = f"{prefix}-evidence"
        if (
            slot.get("clone_root") != clone
            or slot.get("clone_project_root") != f"{clone}/{PROJECT_ROOT.name}"
            or slot.get("cargo_cache_root") != f"{prefix}-cargo-cache"
            or slot.get("supervisor_artifact_path")
            != f"{clone}/{PROJECT_ROOT.name}/artifacts/phase3_q05b_dual_qualification_v2.json"
            or slot.get("artifact_parent_root") != f"{prefix}-artifact"
            or slot.get("artifact_path")
            != f"{prefix}-artifact/phase3_q05b_dual_qualification_v2.json"
            or slot.get("evidence_root") != evidence
            or slot.get("trace_root") != f"{prefix}-trace"
            or slot.get("trajectory_trace_id")
            != f"q05b-shadow-retry-campaign-v2r8-slot-{ordinal:02d}"
            or slot.get("stdout_path") != f"{evidence}/supervisor.stdout"
            or slot.get("stderr_path") != f"{evidence}/supervisor.stderr"
            or slot.get("external_trajectory_binding_path")
            != f"{evidence}/trajectory-binding.json"
            or slot.get("slot_result_path") != f"{evidence}/slot-result.json"
        ):
            _fail("FAIL_CAMPAIGN_CONTRACT", "v2r8 slot isolation differs")


def load_phase3_q05b_shadow_retry_campaign_contract_v2() -> dict[str, object]:
    config_raw, _config_status = _read_regular_nofollow(
        CONFIG_PATH,
        "campaign contract",
        8 * 1024 * 1024,
    )
    decoded = _pretty_json(config_raw, "campaign contract")
    if type(decoded) is not dict:
        _fail("FAIL_CAMPAIGN_CONTRACT", "campaign contract is not an object")
    contract = decoded
    if (
        contract.get("schema_version")
        != "hegel-phase3a-q05b-shadow-retry-campaign/2"
        or contract.get("contract_id")
        != "hegel-phase3a-q05b-shadow-retry-campaign-v2r8"
    ):
        _fail("FAIL_CAMPAIGN_CONTRACT", "campaign contract header differs")
    expected_top_fields = {
        "schema_version",
        "contract_id",
        "status",
        "claim_scope",
        "canonicalization",
        "hash_domains",
        "contract_digest",
        "authority_boundary",
        "target_commit_a_binding",
        "campaign_source_binding",
        "fixed_external_roots",
        "slot_registry",
        "path_isolation_policy",
        "authorization_policy",
        "execution_policy",
        "trajectory_binding_policy",
        "retry_policy",
        "success_policy",
        "terminal_failure_policy",
        "ledger_policy",
        "protected_state_policy",
        "forbidden_scope",
        "implementation_state",
    }
    if set(contract) != expected_top_fields:
        _fail("FAIL_CAMPAIGN_CONTRACT", "campaign contract fields differ")
    _require_v2r8_instance_isolation_v2(contract)
    canonicalization = contract["canonicalization"]
    authority = contract["authority_boundary"]
    execution_policy = contract["execution_policy"]
    forbidden = contract["forbidden_scope"]
    implementation = contract["implementation_state"]
    if (
        type(canonicalization) is not dict
        or canonicalization.get(
            "runtime_authorization_record_fields_not_stored_in_the_contract"
        )
        != [
            "capsule_digest",
            "python310_dependency_preflight_receipt_digest",
            "owner_confirmation_digest",
            "authorization_digest",
        ]
        or contract["status"]
        != "V2R8_FINAL_BOUND_AWAITING_EXACT_OWNER_AUTHORIZATION_NOT_EXECUTED"
        or contract["claim_scope"]
        != "THREE_SLOT_SHADOW_ONLY_NONADOPTING_Q05B_RETRY_CAMPAIGN"
        or type(authority) is not dict
        or authority.get("campaign_execution_authorized") is not False
        or authority.get("preflight_authorized") is not True
        or authority.get("network_authorized") is not False
        or authority.get("maximum_supervisor_invocations_after_future_authorization")
        != 3
        or type(execution_policy) is not dict
        or execution_policy.get("maximum_supervisor_invocations") != 3
        or execution_policy.get("slot_count") != 3
        or execution_policy.get("strictly_sequential") is not True
        or execution_policy.get("each_slot_is_one_shot") is not True
        or execution_policy.get("network_during_clone_cache_or_execution_allowed")
        is not False
        or execution_policy.get("supervisor_interpreter")
        != "/usr/bin/python3.10"
        or execution_policy.get("git_executable") != "/usr/bin/git"
        or execution_policy.get("docker_executable") != "/usr/bin/docker"
        or execution_policy.get("docker_host")
        != "unix:///var/run/docker.sock"
        or execution_policy.get("clone_source_repository")
        != os.fspath(WORKTREE_ROOT)
        or execution_policy.get("clone_source_project_relative_path")
        != PROJECT_ROOT.name
        or execution_policy.get("clone_target_ref_name")
        != _CLONE_TARGET_REF_V2R8
        or execution_policy.get("clone_ref_policy")
        != (
            "EXACTLY_ONE_PERMANENT_INTERNAL_REF_BOUND_TO_TARGET_COMMIT_A;"
            "ALL_SOURCE_REFS_AND_STASH_REFS_REMOVED;"
            "REF_NAME_AND_OBJECT_ID_REPLAYED_BEFORE_AND_AFTER_SUPERVISOR"
        )
        or execution_policy.get("cargo_copy_source")
        != "/home/erzhu419/.local/state/hegel-machine/rust-cargo-cache"
        or execution_policy.get("fixed_no_network_environment")
        != _EXPECTED_FIXED_ENVIRONMENT
        or execution_policy.get("supervisor_timeout_seconds") != 7200
        or execution_policy.get("stdout_max_bytes") != 67_108_864
        or execution_policy.get("stderr_max_bytes") != 67_108_864
        or execution_policy.get("sigint_grace_seconds") != 120
        or execution_policy.get("sigkill_reap_seconds") != 30
        or type(forbidden) is not dict
        or any(
            forbidden.get(field) is not True
            for field in (
                "network_forbidden",
                "meta_access_forbidden",
                "production_access_forbidden",
                "q1_execution_forbidden",
                "active_transition_forbidden",
                "adoption_forbidden",
                "promotion_forbidden",
                "current_pointer_write_forbidden",
                "historical_attempt2_access_forbidden",
                "historical_attempt3_access_forbidden",
                "generic_retry_forbidden",
                "slot_04_forbidden",
                "artifact_copy_move_link_outside_fixed_external_path_forbidden",
                "claim_elevation_forbidden",
            )
        )
        or type(implementation) is not dict
        or implementation.get("target_commit_a_bound") is not True
        or implementation.get("campaign_source_bound") is not True
        or implementation.get("campaign_executed") is not False
        or implementation.get("supervisor_invocation_count") != 0
    ):
        _fail("FAIL_CAMPAIGN_CONTRACT", "campaign authority boundary differs")
    _scheduling_admission_policy_v2(contract)
    _python310_dependency_preflight_policy_v1(contract)
    domains = contract.get("hash_domains")
    if type(domains) is not dict or domains != _EXPECTED_HASH_DOMAINS_V2R8:
        _fail("FAIL_CAMPAIGN_CONTRACT", "campaign hash-domain registry differs")
    trajectory_policy = contract.get("trajectory_binding_policy")
    if (
        type(trajectory_policy) is not dict
        or trajectory_policy.get("binding_digest_domain")
        != domains["trajectory_binding"]
        or trajectory_policy.get("producer_id") != "Q05B_SUPERVISOR_V4"
    ):
        _fail(
            "FAIL_CAMPAIGN_CONTRACT",
            "trajectory binding domain or v2r8 producer differs",
        )
    digest_fields = {
        "retry_policy": "retry_predicate_digest",
        "success_policy": "success_predicate_digest",
        "ledger_policy": "ledger_state_registry_digest",
        "forbidden_scope": "forbidden_scope_digest",
    }
    for policy_name, digest_field in digest_fields.items():
        policy = contract.get(policy_name)
        if type(policy) is not dict:
            _fail("FAIL_CAMPAIGN_CONTRACT", f"{policy_name} differs")
        unsigned_policy = deepcopy(policy)
        observed_digest = unsigned_policy.pop(digest_field, None)
        expected_digest = _domain_digest(
            _PREDICATE_DIGEST_DOMAINS[policy_name],
            unsigned_policy,
        )
        if observed_digest != expected_digest:
            _fail(
                "FAIL_CAMPAIGN_CONTRACT",
                f"{policy_name} digest differs",
            )
    ledger_policy = contract["ledger_policy"]
    assert type(ledger_policy) is dict
    if ledger_policy.get("record_names_in_exact_order") != list(
        _LEDGER_RECORD_NAMES_V2
    ):
        _fail(
            "FAIL_CAMPAIGN_CONTRACT",
            "ledger filename registry differs",
        )
    contract_digest = contract.get("contract_digest")
    if type(contract_digest) is not str or _TAGGED_SHA256.fullmatch(contract_digest) is None:
        _fail("FAIL_CAMPAIGN_CONTRACT", "campaign contract is not final-bound")
    unsigned_contract = deepcopy(contract)
    unsigned_contract.pop("contract_digest")
    if _domain_digest(domains["contract"], unsigned_contract) != contract_digest:
        _fail("FAIL_CAMPAIGN_CONTRACT", "campaign contract digest differs")
    binding = contract.get("campaign_source_binding")
    if type(binding) is not dict:
        _fail("FAIL_CAMPAIGN_SOURCE", "campaign source binding differs")
    expected_paths = [
        CORE_PATH.relative_to(PROJECT_ROOT).as_posix(),
        CORE_TEST_PATH.relative_to(PROJECT_ROOT).as_posix(),
        TOOL_PATH.relative_to(PROJECT_ROOT).as_posix(),
        TOOL_TEST_PATH.relative_to(PROJECT_ROOT).as_posix(),
        DOC_PATH.relative_to(PROJECT_ROOT).as_posix(),
    ]
    if (
        binding.get("expected_future_paths_in_exact_order") != expected_paths
        or binding.get(
            "controller_commit_b_changed_paths_from_target_in_exact_order"
        )
        != list(_CONTROLLER_CHANGED_PATHS_FROM_TARGET_IN_EXACT_ORDER)
    ):
        _fail("FAIL_CAMPAIGN_SOURCE", "campaign source path registry differs")
    target_binding = contract.get("target_commit_a_binding")
    if (
        type(target_binding) is not dict
        or target_binding.get("required_paths_in_exact_order")
        != list(_TARGET_REQUIRED_PATHS_V2R8)
        or type(target_binding.get("required_raw_sha256_by_path")) is not dict
        or list(target_binding["required_raw_sha256_by_path"])
        != list(_TARGET_REQUIRED_PATHS_V2R8)
    ):
        _fail("FAIL_CAMPAIGN_SOURCE", "target path registry differs")
    file_specs = (
        (CORE_PATH, "core_raw_sha256", "campaign core"),
        (CORE_TEST_PATH, "core_test_raw_sha256", "campaign core tests"),
        (TOOL_TEST_PATH, "tool_test_raw_sha256", "campaign tool tests"),
        (DOC_PATH, "document_raw_sha256", "campaign document"),
        (V1_SAFETY_KERNEL_PATH, "v1_safety_kernel_raw_sha256", "v1 safety kernel"),
    )
    source_hashes: dict[str, str] = {}
    for path, field, label in file_specs:
        raw, _status = _read_regular_nofollow(path, label, 64 * 1024 * 1024)
        observed = sha256(raw).hexdigest()
        if binding.get(field) != observed:
            _fail("FAIL_CAMPAIGN_SOURCE", f"{label} hash differs")
        source_hashes[field] = observed
    tool_raw, _tool_status = _read_regular_nofollow(
        TOOL_PATH,
        "campaign tool",
        32 * 1024 * 1024,
    )
    tool_raw_sha = sha256(tool_raw).hexdigest()
    if binding.get("tool_raw_sha256") != tool_raw_sha:
        _fail("FAIL_CAMPAIGN_SOURCE", "campaign tool raw hash differs")
    static_surface_value = {
        "expected_paths": expected_paths,
        "core_raw_sha256": source_hashes["core_raw_sha256"],
        "core_test_raw_sha256": source_hashes["core_test_raw_sha256"],
        "tool_raw_sha256": tool_raw_sha,
        "tool_test_raw_sha256": source_hashes["tool_test_raw_sha256"],
        "document_raw_sha256": source_hashes["document_raw_sha256"],
        "v1_safety_kernel_path": binding.get("v1_safety_kernel_path"),
        "v1_safety_kernel_raw_sha256": source_hashes["v1_safety_kernel_raw_sha256"],
    }
    static_surface = sha256(_canonical_json_bytes(static_surface_value)).hexdigest()
    if binding.get("static_surface_sha256") != static_surface:
        _fail("FAIL_CAMPAIGN_SOURCE", "campaign static surface differs")
    expected_binding_digest = _domain_digest(
        domains["campaign_source_binding"],
        _source_binding_preimage(binding),
    )
    if binding.get("source_binding_digest") != expected_binding_digest:
        _fail("FAIL_CAMPAIGN_SOURCE", "campaign source binding digest differs")
    _load_core(contract)
    return contract



def _process_group_present(pid: int) -> bool | None:
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    except BaseException:
        return None
    return True


def _require_single_threaded_spawn_controller_v2() -> None:
    """Require one Linux task at both sides of the consumption callback."""

    try:
        task_names = os.listdir("/proc/self/task")
    except BaseException as error:
        _fail(
            "FAIL_CAMPAIGN_PROCESS",
            "cannot prove single-threaded spawn controller: "
            f"{type(error).__name__}:{error}",
            terminal=True,
        )
    expected = {str(os.getpid())}
    if set(task_names) != expected or len(task_names) != 1:
        _fail(
            "FAIL_CAMPAIGN_PROCESS",
            "spawn controller is not exactly single-threaded: "
            f"task_ids={sorted(task_names)!r}",
            terminal=True,
        )


def _require_default_sigchld_v2() -> None:
    """Reject automatic or foreign child reaping before any consumption."""

    try:
        disposition = signal.getsignal(signal.SIGCHLD)
    except BaseException as error:
        _fail(
            "FAIL_CAMPAIGN_PROCESS",
            "cannot prove default SIGCHLD disposition: "
            f"{type(error).__name__}:{error}",
            terminal=True,
        )
    if disposition != signal.SIG_DFL:
        _fail(
            "FAIL_CAMPAIGN_PROCESS",
            "spawn controller requires SIGCHLD=SIG_DFL before consumption "
            "and spawn",
            terminal=True,
        )


def _read_spawn_registration_receipt_v2(
    read_fd: int,
    timeout_seconds: float = _SPAWN_RECEIPT_WAIT_SECONDS,
) -> tuple[bytes, bool]:
    """Read one bounded registration row; bool means EOF was observed."""

    raw = bytearray()
    deadline = time.monotonic() + timeout_seconds
    os.set_blocking(read_fd, False)
    eof = False
    while len(raw) <= _SPAWN_RECEIPT_MAX_BYTES:
        try:
            chunk = os.read(
                read_fd,
                _SPAWN_RECEIPT_MAX_BYTES + 1 - len(raw),
            )
        except BlockingIOError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            select.select([read_fd], [], [], min(remaining, 0.05))
            continue
        if not chunk:
            eof = True
            break
        raw.extend(chunk)
    return bytes(raw), eof


def _parse_spawn_registration_receipt_v2(
    raw: bytes,
    eof: bool,
) -> tuple[str, int | None]:
    if raw == b"":
        return "MISSING", None
    if b"\n" not in raw:
        return "TORN", None
    if (
        len(raw) > _SPAWN_RECEIPT_MAX_BYTES
        or re.fullmatch(rb"[1-9][0-9]{0,10}\n", raw) is None
    ):
        return "MALFORMED", None
    pid = int(raw[:-1])
    if pid < 2 or pid > 2**31 - 1:
        return "MALFORMED", None
    if not eof:
        return "TORN", None
    return "VALID", pid


def _partial_popen_pid_v2(process: subprocess.Popen[bytes]) -> int | None:
    pid = getattr(process, "pid", None)
    return pid if type(pid) is int and 2 <= pid <= 2**31 - 1 else None


def _cleanup_registered_spawn_v2(
    process: subprocess.Popen[bytes],
    *,
    registered_pid_or_null: int | None,
    pid_match_or_null: bool | None,
    reap_seconds: float,
) -> dict[str, object]:
    """Kill and directly reap only an identity proven by Popen or receipt."""

    popen_pid = _partial_popen_pid_v2(process)
    target_pid = popen_pid
    if target_pid is None and registered_pid_or_null is not None:
        target_pid = registered_pid_or_null
    already_reaped = getattr(process, "returncode", None) is not None
    errors: list[str] = []
    kill_attempted = False
    direct_child_reaped = False
    group_absent = False
    if target_pid is not None and already_reaped:
        # Popen can reap an initial-exec failure before raising.  From that
        # instant the numeric PID/PGID can be reused, so neither signalling nor
        # a later group-presence probe identifies the child that we owned.
        direct_child_reaped = True
        errors.append("process-already-reaped-no-original-group-proof")
    elif target_pid is not None:
        kill_attempted = True
        try:
            os.killpg(target_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except BaseException as error:
            errors.append(f"killpg:{type(error).__name__}:{error}")
        try:
            if popen_pid == target_pid:
                process.wait(timeout=reap_seconds)
                direct_child_reaped = getattr(process, "returncode", None) is not None
            else:
                deadline = time.monotonic() + reap_seconds
                while True:
                    waited_pid, _status = os.waitpid(target_pid, os.WNOHANG)
                    if waited_pid == target_pid:
                        direct_child_reaped = True
                        break
                    if time.monotonic() >= deadline:
                        errors.append("waitpid:timeout")
                        break
                    time.sleep(0.01)
        except BaseException as error:
            errors.append(f"waitpid:{type(error).__name__}:{error}")
        present = _process_group_present(target_pid)
        group_absent = present is False
        if present is not False:
            errors.append(f"process-group-presence:{present!r}")
    else:
        errors.append("spawn-identity-unavailable")
    for name in ("stdin", "stdout", "stderr"):
        stream = getattr(process, name, None)
        if stream is not None:
            try:
                stream.close()
            except BaseException as error:
                errors.append(f"{name}-close:{type(error).__name__}:{error}")
    identity_consistent = pid_match_or_null is not False
    return {
        "cleanup_target_pid_or_null": target_pid,
        "cleanup_kill_attempted": kill_attempted,
        "cleanup_direct_child_reaped": direct_child_reaped,
        "cleanup_process_group_absent": group_absent,
        "cleanup_identity_consistent": identity_consistent,
        "cleanup_errors": errors,
        "cleanup_certain": (
            target_pid is not None
            and direct_child_reaped
            and group_absent
            and identity_consistent
            and not errors
        ),
    }


def _spawn_registration_receipt_v2(
    argv: Sequence[str],
    *,
    raw: bytes,
    state: str,
    registered_pid_or_null: int | None,
    process: subprocess.Popen[bytes],
    constructor_error_or_null: BaseException | None,
    cleanup: Mapping[str, object],
) -> dict[str, object]:
    popen_pid = _partial_popen_pid_v2(process)
    pid_match = (
        popen_pid == registered_pid_or_null
        if popen_pid is not None and registered_pid_or_null is not None
        else None
    )
    return {
        "schema_version": _SPAWN_REGISTRATION_SCHEMA_V2,
        "phase": "SPAWN_REGISTRATION",
        "argv": list(argv),
        "registration_state": state,
        "registration_raw_hex": raw[: _SPAWN_RECEIPT_MAX_BYTES + 1].hex(),
        "registered_pid_or_null": registered_pid_or_null,
        "popen_pid_or_null": popen_pid,
        "popen_child_created": getattr(process, "_child_created", False) is True,
        "pid_match_or_null": pid_match,
        "constructor_error_or_null": (
            _error_snapshot_v2(constructor_error_or_null)
            if constructor_error_or_null is not None
            else None
        ),
        **deepcopy(dict(cleanup)),
        "later_ledger_write_permitted": (
            state == "VALID"
            and pid_match is not False
            and cleanup.get("cleanup_certain") is True
        ),
    }


def _kill_and_reap(
    process: subprocess.Popen[bytes],
    *,
    first_signal: int,
    grace_seconds: float,
    reap_seconds: float,
) -> bool:
    group_control = True
    try:
        try:
            os.killpg(process.pid, first_signal)
        except ProcessLookupError:
            pass
        except BaseException:
            group_control = False
        deadline = time.monotonic() + grace_seconds
        while time.monotonic() < deadline:
            present = _process_group_present(process.pid)
            if present is False:
                break
            if present is None:
                group_control = False
                break
            time.sleep(0.05)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except BaseException:
            group_control = False
        try:
            process.wait(timeout=reap_seconds)
        except (subprocess.TimeoutExpired, ChildProcessError):
            return False
        deadline = time.monotonic() + reap_seconds
        present = _process_group_present(process.pid)
        while present is True and time.monotonic() < deadline:
            time.sleep(0.05)
            present = _process_group_present(process.pid)
        return process.poll() is not None and present is False and group_control
    finally:
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except BaseException:
                    pass


def _bounded_process(
    argv: Sequence[str],
    *,
    cwd: str,
    env: Mapping[str, str],
    timeout_seconds: float,
    stdout_max_bytes: int,
    stderr_max_bytes: int,
    first_signal: int = signal.SIGKILL,
    grace_seconds: float = 0.0,
    reap_seconds: float = 30.0,
    before_popen: Any | None = None,
) -> dict[str, object]:
    if (
        type(argv) not in (list, tuple)
        or not argv
        or any(type(item) is not str or "\x00" in item for item in argv)
        or type(cwd) is not str
        or not Path(cwd).is_absolute()
        or type(timeout_seconds) not in (int, float)
        or isinstance(timeout_seconds, bool)
        or timeout_seconds <= 0
        or type(stdout_max_bytes) is not int
        or type(stderr_max_bytes) is not int
        or min(stdout_max_bytes, stderr_max_bytes) < 0
    ):
        _fail("FAIL_CAMPAIGN_PROCESS", "bounded process inputs differ", terminal=True)
    _require_default_sigchld_v2()
    _require_single_threaded_spawn_controller_v2()
    old_umask = os.umask(0o077)
    previous_mask: set[signal.Signals] | None = None
    process: subprocess.Popen[bytes] | None = None
    read_fd = -1
    write_fd = -1
    spawn_attempted = False
    spawn_error: BaseException | None = None
    callback_error: BaseException | None = None
    restore_errors: list[str] = []
    try:
        read_fd, write_fd = os.pipe2(os.O_CLOEXEC)
        actual_argv = list(argv)
        child_env = dict(env)
        if any(
            type(key) is not str
            or type(value) is not str
            or "\x00" in key
            or "\x00" in value
            or "=" in key
            for key, value in child_env.items()
        ):
            _fail(
                "FAIL_CAMPAIGN_PROCESS",
                "bounded process environment differs",
                terminal=True,
            )
        launcher_argv = [
            _SPAWN_LAUNCHER_INTERPRETER_V2,
            "-I",
            "-S",
            "-B",
            "-X",
            "utf8",
            "-c",
            _SPAWN_LAUNCHER_SOURCE_V2,
            str(write_fd),
            *actual_argv,
        ]
        if before_popen is not None:
            try:
                before_popen()
            except ShadowRetryCampaignErrorV2 as error:
                callback_error = error
            except BaseException as error:
                callback_error = ShadowRetryCampaignErrorV2(
                    "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN",
                    "before-spawn consumption publication failed: "
                    f"{type(error).__name__}:{error}",
                    terminal=True,
                )
        if callback_error is None:
            process = subprocess.Popen.__new__(subprocess.Popen)
            # Make cleanup and __del__ safe even if __init__ is interrupted
            # before its own first stores.  __init__ overwrites these values.
            process.pid = None  # type: ignore[assignment]
            process.returncode = None
            process.stdin = None
            process.stdout = None
            process.stderr = None
            process._child_created = False
            # SIGINT is the one Python-installed asynchronous handler in the
            # exact isolated CLI.  Block it across the final task-count proof
            # and Popen's owned handle stores.  Default fatal signals retain
            # the documented last-durable-prefix evidence ceiling.
            previous_mask = signal.pthread_sigmask(
                signal.SIG_BLOCK,
                {signal.SIGINT},
            )
            _require_default_sigchld_v2()
            _require_single_threaded_spawn_controller_v2()
            spawn_attempted = True
            subprocess.Popen.__init__(
                process,
                launcher_argv,
                cwd=cwd,
                env=child_env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                close_fds=True,
                pass_fds=(write_fd,),
                restore_signals=True,
            )
    except BaseException as error:
        spawn_error = error
    finally:
        if write_fd >= 0:
            try:
                os.close(write_fd)
            except BaseException as error:
                restore_errors.append(
                    f"registration-write-close:{type(error).__name__}:{error}"
                )
            write_fd = -1
        if previous_mask is not None:
            try:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            except BaseException as error:
                restore_errors.append(f"signal-mask:{type(error).__name__}:{error}")
        try:
            os.umask(old_umask)
        except BaseException as error:
            restore_errors.append(f"umask:{type(error).__name__}:{error}")
    if callback_error is not None:
        if read_fd >= 0:
            try:
                os.close(read_fd)
            except BaseException:
                pass
        raise callback_error
    if not spawn_attempted:
        if read_fd >= 0:
            try:
                os.close(read_fd)
            except BaseException:
                pass
        _fail(
            "FAIL_CAMPAIGN_PROCESS",
            "spawn setup failed before Popen: "
            + (
                f"{type(spawn_error).__name__}:{spawn_error}"
                if spawn_error is not None
                else "unknown setup failure"
            ),
            terminal=True,
        )
    if process is None:
        _fail(
            "FAIL_CAMPAIGN_SPAWN_UNCERTAIN",
            "Popen attempt has no recoverable process object",
            terminal=True,
            receipt={
                "schema_version": _SPAWN_REGISTRATION_SCHEMA_V2,
                "phase": "SPAWN_REGISTRATION",
                "registration_state": "PROCESS_OBJECT_MISSING",
                "later_ledger_write_permitted": False,
            },
        )

    raw_receipt = b""
    receipt_eof = False
    receipt_read_error: BaseException | None = None
    try:
        raw_receipt, receipt_eof = _read_spawn_registration_receipt_v2(read_fd)
    except BaseException as error:
        receipt_read_error = error
    finally:
        if read_fd >= 0:
            try:
                os.close(read_fd)
            except BaseException as error:
                restore_errors.append(
                    f"registration-read-close:{type(error).__name__}:{error}"
                )
    if receipt_read_error is None:
        registration_state, registered_pid = (
            _parse_spawn_registration_receipt_v2(raw_receipt, receipt_eof)
        )
    else:
        registration_state, registered_pid = "READ_ERROR", None
    popen_pid = _partial_popen_pid_v2(process)
    pid_match = (
        popen_pid == registered_pid
        if popen_pid is not None and registered_pid is not None
        else None
    )
    normal_spawn = (
        spawn_error is None
        and receipt_read_error is None
        and registration_state == "VALID"
        and popen_pid is not None
        and pid_match is True
        and getattr(process, "_child_created", False) is True
        and not restore_errors
    )
    if not normal_spawn:
        cleanup_error: BaseException | None = None
        try:
            cleanup = _cleanup_registered_spawn_v2(
                process,
                registered_pid_or_null=registered_pid,
                pid_match_or_null=pid_match,
                reap_seconds=reap_seconds,
            )
        except BaseException as error:
            cleanup_error = error
            cleanup = {
                "cleanup_target_pid_or_null": popen_pid or registered_pid,
                "cleanup_kill_attempted": False,
                "cleanup_direct_child_reaped": False,
                "cleanup_process_group_absent": False,
                "cleanup_identity_consistent": pid_match is not False,
                "cleanup_errors": [
                    f"cleanup:{type(error).__name__}:{error}"
                ],
                "cleanup_certain": False,
            }
        constructor_error = spawn_error or receipt_read_error or cleanup_error
        receipt = _spawn_registration_receipt_v2(
            argv,
            raw=raw_receipt,
            state=registration_state,
            registered_pid_or_null=registered_pid,
            process=process,
            constructor_error_or_null=constructor_error,
            cleanup=cleanup,
        )
        uncertain = (
            registration_state != "VALID"
            or pid_match is False
            or cleanup.get("cleanup_certain") is not True
        )
        _fail(
            (
                "FAIL_CAMPAIGN_SPAWN_UNCERTAIN"
                if uncertain
                else "FAIL_CAMPAIGN_PROCESS"
            ),
            "registered Popen did not cross its commit boundary: "
            f"registration={registration_state}; pid_match={pid_match!r}; "
            f"restore_errors={restore_errors!r}; "
            f"error={type(constructor_error).__name__ if constructor_error is not None else None}:"
            f"{constructor_error if constructor_error is not None else None}",
            terminal=True,
            receipt=receipt,
        )

    selector: selectors.BaseSelector | None = None
    buffers: dict[str, bytearray] | None = None
    overflow: dict[str, bool] | None = None
    timed_out = False
    child_reaped = False
    group_absent: bool | None = None
    result: dict[str, object] | None = None
    caught_error: BaseException | None = None
    try:
        # Every allocation and selector operation after Popen is inside this
        # cleanup boundary.  A setup failure must not strand a live child.
        buffers = {"stdout": bytearray(), "stderr": bytearray()}
        overflow = {"stdout": False, "stderr": False}
        selector = selectors.DefaultSelector()
        deadline = time.monotonic() + float(timeout_seconds)
        streams = {process.stdout: "stdout", process.stderr: "stderr"}
        for stream, name in streams.items():
            if stream is None:
                _fail("FAIL_CAMPAIGN_PROCESS", f"{name} pipe is absent", terminal=True)
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, name)
        while selector.get_map() or process.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            for key, _mask in selector.select(min(remaining, 0.25)):
                try:
                    chunk = os.read(key.fileobj.fileno(), 65536)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                name = key.data
                limit = stdout_max_bytes if name == "stdout" else stderr_max_bytes
                remaining_capacity = max(0, limit - len(buffers[name]))
                buffers[name].extend(chunk[:remaining_capacity])
                if len(chunk) > remaining_capacity:
                    overflow[name] = True
                    break
            if any(overflow.values()):
                break
        interrupted = timed_out or any(overflow.values())
        if interrupted:
            child_reaped = _kill_and_reap(
                process,
                first_signal=first_signal,
                grace_seconds=grace_seconds,
                reap_seconds=reap_seconds,
            )
            group_absent = _process_group_present(process.pid)
        else:
            try:
                process.wait(timeout=reap_seconds)
                child_reaped = process.poll() is not None
            except (subprocess.TimeoutExpired, ChildProcessError):
                child_reaped = _kill_and_reap(
                    process,
                    first_signal=signal.SIGKILL,
                    grace_seconds=0.0,
                    reap_seconds=reap_seconds,
                )
            for stream, name in streams.items():
                if stream is None:
                    continue
                while True:
                    try:
                        chunk = os.read(stream.fileno(), 65536)
                    except BlockingIOError:
                        break
                    if not chunk:
                        break
                    limit = stdout_max_bytes if name == "stdout" else stderr_max_bytes
                    capacity = max(0, limit - len(buffers[name]))
                    buffers[name].extend(chunk[:capacity])
                    if len(chunk) > capacity:
                        overflow[name] = True
            group_absent = _process_group_present(process.pid)
            # A PID/PGID can be reused as soon as the direct child is reaped.
            # Never signal that numeric process-group identity after wait(); a
            # present or uncertain observation is cleanup uncertainty for the
            # caller to terminalize, not authority to kill a possibly
            # unrelated group.
        result = {
            "return_code": process.returncode,
            "stdout": bytes(buffers["stdout"]),
            "stderr": bytes(buffers["stderr"]),
            "stdout_overflow": overflow["stdout"],
            "stderr_overflow": overflow["stderr"],
            "timed_out": timed_out,
            "child_reaped": child_reaped,
            "process_group_absent": group_absent is False,
            "wrapper_resources_closed": False,
        }
    except BaseException as error:
        caught_error = error
        # poll() may itself reap the direct child.  Signalling its numeric PGID
        # after that point risks acting on a reused identifier.
        if process.poll() is None:
            cleanup_reaped = _kill_and_reap(
                process,
                first_signal=signal.SIGKILL,
                grace_seconds=0.0,
                reap_seconds=reap_seconds,
            )
            if cleanup_reaped is not True:
                _fail(
                    "FAIL_CAMPAIGN_PROCESS",
                    "bounded process raised and live-child cleanup could not "
                    f"be proven: {type(error).__name__}:{error}",
                    terminal=True,
                )
        raise
    finally:
        close_errors: list[str] = []
        if selector is not None:
            try:
                selector.close()
            except BaseException as error:
                close_errors.append(
                    f"selector:{type(error).__name__}:{error}"
                )
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except BaseException as error:
                    close_errors.append(
                        f"pipe:{type(error).__name__}:{error}"
                    )
        if close_errors:
            partial = result
            if partial is None:
                partial = {
                    "return_code": process.returncode,
                    "stdout": (
                        bytes(buffers["stdout"])
                        if buffers is not None
                        else b""
                    ),
                    "stderr": (
                        bytes(buffers["stderr"])
                        if buffers is not None
                        else b""
                    ),
                    "stdout_overflow": (
                        overflow["stdout"] if overflow is not None else False
                    ),
                    "stderr_overflow": (
                        overflow["stderr"] if overflow is not None else False
                    ),
                    "timed_out": timed_out,
                    "child_reaped": child_reaped,
                    "process_group_absent": group_absent is False,
                    "wrapper_resources_closed": False,
                }
            diagnostic = _bounded_process_diagnostic(
                "bounded process wrapper resource closure",
                argv,
                partial,
                classification={
                    "accepted": False,
                    "close_errors": close_errors,
                    "prior_error_or_null": (
                        f"{type(caught_error).__name__}:{caught_error}"
                        if caught_error is not None
                        else None
                    ),
                },
            )
            _fail(
                "FAIL_CAMPAIGN_PROCESS",
                "wrapper resource closure failed: " + ";".join(close_errors),
                terminal=True,
                receipt=diagnostic,
            )
        if result is not None:
            result["wrapper_resources_closed"] = True
    if result is None:
        _fail(
            "FAIL_CAMPAIGN_PROCESS",
            "bounded process produced no result",
            terminal=True,
        )
    return result


def _bounded_process_diagnostic(
    label: str,
    argv: Sequence[str],
    result: Mapping[str, object],
    *,
    classification: Mapping[str, object] | None = None,
) -> dict[str, object]:
    stdout = result["stdout"]
    stderr = result["stderr"]
    if type(stdout) is not bytes or type(stderr) is not bytes:
        _fail(
            "FAIL_CAMPAIGN_PROCESS",
            "bounded diagnostic stream type differs",
            terminal=True,
        )

    def stream(raw: bytes) -> dict[str, object]:
        excerpt = raw[:1024]
        return {
            "byte_count": len(raw),
            "raw_sha256": sha256(raw).hexdigest(),
            "excerpt_hex": excerpt.hex(),
            "excerpt_byte_count": len(excerpt),
            "excerpt_truncated": len(excerpt) != len(raw),
        }

    return {
        "schema_version": _BOUNDED_COMMAND_DIAGNOSTIC_SCHEMA_V2,
        "label": label,
        "argv": list(argv),
        "return_code": result["return_code"],
        "timed_out": result["timed_out"],
        "stdout_overflow": result["stdout_overflow"],
        "stderr_overflow": result["stderr_overflow"],
        "child_reaped": result["child_reaped"],
        "process_group_absent": result["process_group_absent"],
        "wrapper_resources_closed": result["wrapper_resources_closed"],
        "stdout": stream(stdout),
        "stderr": stream(stderr),
        "classification": (
            deepcopy(dict(classification))
            if classification is not None
            else None
        ),
    }


def _bounded_process_exception_diagnostic(
    label: str,
    argv: Sequence[str],
    error: BaseException,
) -> dict[str, object]:
    prior_receipt = getattr(error, "receipt", None)
    return {
        "schema_version": _BOUNDED_COMMAND_EXCEPTION_DIAGNOSTIC_SCHEMA_V2,
        "label": label,
        "argv": list(argv),
        "phase": "BOUNDED_PROCESS_RAISED_BEFORE_RESULT",
        "error_code": getattr(error, "code", type(error).__name__),
        "error_detail": str(getattr(error, "detail", str(error)))[:4096],
        "return_code": None,
        "timed_out": None,
        "stdout_overflow": None,
        "stderr_overflow": None,
        "child_reaped": None,
        "process_group_absent": None,
        "wrapper_resources_closed": None,
        "stdout": {"available": False},
        "stderr": {"available": False},
        "prior_receipt_or_null": (
            deepcopy(dict(prior_receipt))
            if type(prior_receipt) is dict
            else None
        ),
    }


def _raise_bounded_process_exception(
    label: str,
    argv: Sequence[str],
    error: BaseException,
) -> NoReturn:
    diagnostic = _bounded_process_exception_diagnostic(label, argv, error)
    code = (
        error.code
        if isinstance(error, ShadowRetryCampaignErrorV2)
        else "FAIL_CAMPAIGN_PROCESS"
    )
    _fail(
        code,
        f"{label} bounded process raised before a result: "
        f"{diagnostic['error_code']}:{diagnostic['error_detail']}",
        terminal=True,
        receipt=diagnostic,
    )


def _run_exact_command(
    argv: Sequence[str],
    *,
    cwd: str,
    env: Mapping[str, str],
    timeout_seconds: float,
    maximum_bytes: int,
    label: str,
    require_empty_stdout: bool = False,
    require_empty_stderr: bool = True,
) -> bytes:
    try:
        result = _bounded_process(
            argv,
            cwd=cwd,
            env=env,
            timeout_seconds=timeout_seconds,
            stdout_max_bytes=maximum_bytes,
            stderr_max_bytes=maximum_bytes,
        )
    except BaseException as error:
        _raise_bounded_process_exception(label, argv, error)
    if (
        result["return_code"] != 0
        or result["timed_out"] is not False
        or result["stdout_overflow"] is not False
        or result["stderr_overflow"] is not False
        or result["child_reaped"] is not True
        or result["process_group_absent"] is not True
        or result["wrapper_resources_closed"] is not True
        or (require_empty_stdout and result["stdout"] != b"")
        or (require_empty_stderr and result["stderr"] != b"")
    ):
        diagnostic = _bounded_process_diagnostic(
            label,
            argv,
            result,
            classification={
                "require_empty_stdout": require_empty_stdout,
                "require_empty_stderr": require_empty_stderr,
                "accepted": False,
            },
        )
        _fail(
            "FAIL_CAMPAIGN_PROCESS",
            f"{label} failed closed: return_code={result['return_code']}; "
            f"timed_out={result['timed_out']}; "
            f"stdout_overflow={result['stdout_overflow']}; "
            f"stderr_overflow={result['stderr_overflow']}; "
            f"stderr_excerpt_hex={diagnostic['stderr']['excerpt_hex']}",
            terminal=True,
            receipt=diagnostic,
        )
    return result["stdout"]


def _certain_registered_git_failure_diagnostic_v2(
    contract: Mapping[str, object],
    value: object,
    detail: object,
) -> bool:
    if (
        type(value) is not dict
        or value.get("schema_version")
        != _BOUNDED_COMMAND_DIAGNOSTIC_SCHEMA_V2
        or set(value)
        != {
            "schema_version",
            "label",
            "argv",
            "return_code",
            "timed_out",
            "stdout_overflow",
            "stderr_overflow",
            "child_reaped",
            "process_group_absent",
            "wrapper_resources_closed",
            "stdout",
            "stderr",
            "classification",
        }
        or type(detail) is not str
    ):
        return False
    classification = value["classification"]
    if (
        type(classification) is not dict
        or set(classification)
        != {
            "require_empty_stdout",
            "require_empty_stderr",
            "accepted",
        }
        or classification["accepted"] is not False
        or type(classification["require_empty_stdout"]) is not bool
        or type(classification["require_empty_stderr"]) is not bool
        or type(value["return_code"]) is not int
        or value["return_code"] < 0
        or value["timed_out"] is not False
        or value["stdout_overflow"] is not False
        or value["stderr_overflow"] is not False
        or value["child_reaped"] is not True
        or value["process_group_absent"] is not True
        or value["wrapper_resources_closed"] is not True
        or not _receipt_stream_semantics_v2(value["stdout"])
        or not _receipt_stream_semantics_v2(value["stderr"])
        or not _bounded_command_binding_is_exact_v2(
            contract,
            value["label"],
            value["argv"],
        )
        or value["argv"][0]
        != contract["execution_policy"]["git_executable"]
    ):
        return False
    ordinary_failure = (
        value["return_code"] > 0
        or (
            classification["require_empty_stdout"]
            and value["stdout"]["byte_count"] > 0
        )
        or (
            classification["require_empty_stderr"]
            and value["stderr"]["byte_count"] > 0
        )
    )
    return ordinary_failure and detail == (
        f"{value['label']} failed closed: "
        f"return_code={value['return_code']}; "
        f"timed_out={value['timed_out']}; "
        f"stdout_overflow={value['stdout_overflow']}; "
        f"stderr_overflow={value['stderr_overflow']}; "
        f"stderr_excerpt_hex={value['stderr']['excerpt_hex']}"
    )


def _git(
    contract: Mapping[str, object],
    arguments: Sequence[str],
    label: str,
    *,
    require_empty_stdout: bool = False,
    git_worktree: str | None = None,
) -> bytes:
    execution = contract["execution_policy"]
    assert type(execution) is dict
    path = execution["git_executable"]
    environment = _runtime_environment(
        contract,
        git_worktree=git_worktree,
    )
    bounds = execution["readonly_command_bounds"]["git"]
    argv = [str(path), *arguments]
    try:
        return _run_exact_command(
            argv,
            cwd="/",
            env=environment,
            timeout_seconds=float(bounds["timeout_seconds"]),
            maximum_bytes=max(
                int(bounds["stdout_max_bytes"]),
                int(bounds["stderr_max_bytes"]),
            ),
            label=label,
            require_empty_stdout=require_empty_stdout,
        )
    except ShadowRetryCampaignErrorV2 as error:
        if (
            error.code == "FAIL_CAMPAIGN_PROCESS"
            and _certain_registered_git_failure_diagnostic_v2(
                contract,
                error.receipt,
                error.detail,
            )
        ):
            _fail(
                "FAIL_CAMPAIGN_SOURCE",
                error.detail,
                terminal=error.terminal,
                receipt=error.receipt,
            )
        raise


def _runtime_environment(
    contract: Mapping[str, object],
    *,
    git_worktree: str | None = None,
) -> dict[str, str]:
    execution = contract["execution_policy"]
    assert type(execution) is dict
    raw = execution["fixed_no_network_environment"]
    if type(raw) is not dict or any(
        type(key) is not str or type(value) is not str
        for key, value in raw.items()
    ):
        _fail("FAIL_CAMPAIGN_CONTRACT", "runtime environment differs")
    environment = dict(raw)
    expected_placeholder = "SLOT.clone_root"
    if environment.get("GIT_CONFIG_VALUE_2") != expected_placeholder:
        _fail("FAIL_CAMPAIGN_CONTRACT", "runtime Git worktree placeholder differs")
    environment["GIT_CONFIG_VALUE_2"] = (
        git_worktree
        if git_worktree is not None
        else str(execution["clone_source_repository"])
    )
    return environment


def _require_no_xattrs(descriptor: int, label: str) -> None:
    if not hasattr(os, "listxattr"):
        _fail("FAIL_CAMPAIGN_FILESYSTEM", f"{label} xattr API unavailable")
    try:
        attributes = os.listxattr(descriptor)
    except OSError as error:
        _fail("FAIL_CAMPAIGN_FILESYSTEM", f"{label} xattr replay failed: {error}")
    if attributes:
        _fail("FAIL_CAMPAIGN_FILESYSTEM", f"{label} has extended attributes")


def _scan_tree(
    root: Path,
    *,
    label: str,
    maximum_entries: int = 100_000,
    maximum_total_bytes: int = 4 * 1024 * 1024 * 1024,
    manifest_domain: str = (
        "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R8/CARGO_MANIFEST/V2"
    ),
    allowed_uids: Sequence[int] | None = None,
    reject_group_or_world_writable: bool = False,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    allowed_uid_set = (
        {os.geteuid()}
        if allowed_uids is None
        else set(allowed_uids)
    )
    if (
        not allowed_uid_set
        or any(type(uid) is not int or uid < 0 for uid in allowed_uid_set)
        or type(reject_group_or_world_writable) is not bool
    ):
        _fail("FAIL_CAMPAIGN_CONTRACT", f"{label} owner policy differs")
    root_fd, root_stat = _open_directory_nofollow(root, label)
    rows: list[dict[str, object]] = []
    total_bytes = 0
    directory_count = 1
    file_count = 0
    if (
        root_stat.st_uid not in allowed_uid_set
        or (
            reject_group_or_world_writable
            and stat.S_IMODE(root_stat.st_mode) & 0o022 != 0
        )
    ):
        os.close(root_fd)
        _fail("FAIL_CAMPAIGN_FILESYSTEM", f"{label} root ownership differs")
    _require_no_xattrs(root_fd, f"{label} root")

    def walk(directory_fd: int, prefix: PurePosixPath) -> None:
        nonlocal total_bytes, directory_count, file_count
        try:
            names = sorted(os.listdir(directory_fd))
        except OSError as error:
            _fail("FAIL_CAMPAIGN_FILESYSTEM", f"{label} directory list failed: {error}")
        for name in names:
            if (
                type(name) is not str
                or name in ("", ".", "..")
                or "/" in name
                or "\x00" in name
            ):
                _fail("FAIL_CAMPAIGN_FILESYSTEM", f"{label} entry name differs")
            relative = prefix / name
            relative_text = relative.as_posix()
            if relative.is_absolute() or ".." in relative.parts:
                _fail("FAIL_CAMPAIGN_FILESYSTEM", f"{label} relative path differs")
            before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISDIR(before.st_mode) and not stat.S_ISLNK(before.st_mode):
                flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                child_fd = os.open(name, flags, dir_fd=directory_fd)
                try:
                    opened = os.fstat(child_fd)
                    if (
                        (before.st_dev, before.st_ino, before.st_mode)
                        != (opened.st_dev, opened.st_ino, opened.st_mode)
                        or opened.st_uid not in allowed_uid_set
                        or opened.st_nlink < 2
                        or (
                            reject_group_or_world_writable
                            and stat.S_IMODE(opened.st_mode) & 0o022 != 0
                        )
                    ):
                        _fail("FAIL_CAMPAIGN_FILESYSTEM", f"{label} directory identity differs")
                    _require_no_xattrs(child_fd, f"{label} directory {relative_text}")
                    rows.append(
                        {
                            "relative_path": relative_text,
                            "kind": "DIRECTORY",
                            "mode": stat.S_IMODE(opened.st_mode),
                            "uid": opened.st_uid,
                            "size": 0,
                            "raw_sha256_or_null": None,
                        }
                    )
                    directory_count += 1
                    if directory_count + file_count > maximum_entries:
                        _fail("FAIL_CAMPAIGN_FILESYSTEM", f"{label} entry bound exceeded")
                    walk(child_fd, relative)
                    after = os.fstat(child_fd)
                    pathname = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                    if (
                        before.st_dev,
                        before.st_ino,
                        before.st_mode,
                        before.st_uid,
                        before.st_nlink,
                    ) != (
                        after.st_dev,
                        after.st_ino,
                        after.st_mode,
                        after.st_uid,
                        after.st_nlink,
                    ) or (
                        before.st_dev,
                        before.st_ino,
                        before.st_mode,
                        before.st_uid,
                        before.st_nlink,
                    ) != (
                        pathname.st_dev,
                        pathname.st_ino,
                        pathname.st_mode,
                        pathname.st_uid,
                        pathname.st_nlink,
                    ):
                        _fail("FAIL_CAMPAIGN_FILESYSTEM", f"{label} directory changed")
                finally:
                    os.close(child_fd)
                continue
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                _fail("FAIL_CAMPAIGN_FILESYSTEM", f"{label} contains special or aliased entry")
            flags = os.O_RDONLY | os.O_CLOEXEC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            file_fd = os.open(name, flags, dir_fd=directory_fd)
            try:
                opened = os.fstat(file_fd)
                if (
                    (before.st_dev, before.st_ino, before.st_mode, before.st_size)
                    != (opened.st_dev, opened.st_ino, opened.st_mode, opened.st_size)
                    or opened.st_uid not in allowed_uid_set
                    or opened.st_nlink != 1
                    or (
                        reject_group_or_world_writable
                        and stat.S_IMODE(opened.st_mode) & 0o022 != 0
                    )
                ):
                    _fail("FAIL_CAMPAIGN_FILESYSTEM", f"{label} file identity differs")
                _require_no_xattrs(file_fd, f"{label} file {relative_text}")
                digest = sha256()
                observed_size = 0
                while True:
                    chunk = os.read(file_fd, 1 << 20)
                    if not chunk:
                        break
                    digest.update(chunk)
                    observed_size += len(chunk)
                    total_bytes += len(chunk)
                    if total_bytes > maximum_total_bytes:
                        _fail("FAIL_CAMPAIGN_FILESYSTEM", f"{label} byte bound exceeded")
                after = os.fstat(file_fd)
                pathname = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                identity = (
                    before.st_dev,
                    before.st_ino,
                    before.st_mode,
                    before.st_uid,
                    before.st_nlink,
                    before.st_size,
                    before.st_mtime_ns,
                    before.st_ctime_ns,
                )
                if observed_size != before.st_size or identity != (
                    after.st_dev,
                    after.st_ino,
                    after.st_mode,
                    after.st_uid,
                    after.st_nlink,
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                ) or identity != (
                    pathname.st_dev,
                    pathname.st_ino,
                    pathname.st_mode,
                    pathname.st_uid,
                    pathname.st_nlink,
                    pathname.st_size,
                    pathname.st_mtime_ns,
                    pathname.st_ctime_ns,
                ):
                    _fail("FAIL_CAMPAIGN_FILESYSTEM", f"{label} file changed")
                rows.append(
                    {
                        "relative_path": relative_text,
                        "kind": "REGULAR",
                        "mode": stat.S_IMODE(opened.st_mode),
                        "uid": opened.st_uid,
                        "size": observed_size,
                        "raw_sha256_or_null": digest.hexdigest(),
                    }
                )
                file_count += 1
                if directory_count + file_count > maximum_entries:
                    _fail("FAIL_CAMPAIGN_FILESYSTEM", f"{label} entry bound exceeded")
            finally:
                os.close(file_fd)

    try:
        walk(root_fd, PurePosixPath())
        after_root = os.fstat(root_fd)
        pathname_root = os.stat(root, follow_symlinks=False)
        if (
            root_stat.st_dev,
            root_stat.st_ino,
            root_stat.st_mode,
            root_stat.st_uid,
            root_stat.st_nlink,
        ) != (
            after_root.st_dev,
            after_root.st_ino,
            after_root.st_mode,
            after_root.st_uid,
            after_root.st_nlink,
        ) or (
            root_stat.st_dev,
            root_stat.st_ino,
            root_stat.st_mode,
            root_stat.st_uid,
            root_stat.st_nlink,
        ) != (
            pathname_root.st_dev,
            pathname_root.st_ino,
            pathname_root.st_mode,
            pathname_root.st_uid,
            pathname_root.st_nlink,
        ):
            _fail("FAIL_CAMPAIGN_FILESYSTEM", f"{label} root changed")
    finally:
        os.close(root_fd)
    rows.sort(key=lambda row: str(row["relative_path"]))
    manifest_value = {
        "schema_version": "hegel-phase3a-q05b-cargo-tree-manifest/2",
        "ordered_rows": rows,
    }
    identity = {
        "root": os.fspath(root),
        "root_dev": root_stat.st_dev,
        "root_ino": root_stat.st_ino,
        "root_mode": root_stat.st_mode,
        "root_uid": root_stat.st_uid,
        "file_count": file_count,
        "directory_count": directory_count,
        "total_bytes": total_bytes,
        "manifest_digest": _domain_digest(
            manifest_domain,
            manifest_value,
        ),
        "content_manifest_digest": _domain_digest(
            manifest_domain + "/CONTENT",
            {
                "schema_version": (
                    "hegel-phase3a-q05b-cargo-tree-content-manifest/2"
                ),
                "ordered_rows": [
                    {key: value for key, value in row.items() if key != "uid"}
                    for row in rows
                ],
            },
        ),
    }
    return identity, rows


def _open_relative_directory(root_fd: int, relative: PurePosixPath) -> int:
    current = os.dup(root_fd)
    try:
        for part in relative.parts:
            if part in ("", ".", ".."):
                _fail("FAIL_CAMPAIGN_FILESYSTEM", "relative directory differs", terminal=True)
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            next_fd = os.open(part, flags, dir_fd=current)
            os.close(current)
            current = next_fd
        result = current
        current = -1
        return result
    finally:
        if current >= 0:
            os.close(current)


def _copy_tree_exact(
    source_root: Path,
    target_root: Path,
    expected_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if os.path.lexists(target_root):
        _fail("FAIL_CAMPAIGN_FRESHNESS", "slot Cargo cache already exists", terminal=True)
    source_fd, _source_stat = _open_directory_nofollow(source_root, "Cargo source")
    parent_fd, target_fd = _create_private_root(target_root, "slot Cargo cache")
    os.close(parent_fd)
    try:
        directories = [row for row in expected_rows if row["kind"] == "DIRECTORY"]
        directories.sort(key=lambda row: (len(PurePosixPath(str(row["relative_path"])).parts), str(row["relative_path"])))
        for row in directories:
            relative = PurePosixPath(str(row["relative_path"]))
            parent_relative = relative.parent
            parent = _open_relative_directory(target_fd, parent_relative)
            try:
                os.mkdir(relative.name, int(row["mode"]), dir_fd=parent)
                child = _open_relative_directory(target_fd, relative)
                try:
                    os.fchmod(child, int(row["mode"]))
                    os.fsync(child)
                finally:
                    os.close(child)
                os.fsync(parent)
            finally:
                os.close(parent)
        for row in expected_rows:
            if row["kind"] != "REGULAR":
                continue
            relative = PurePosixPath(str(row["relative_path"]))
            source_parent = _open_relative_directory(source_fd, relative.parent)
            target_parent = _open_relative_directory(target_fd, relative.parent)
            source_file = -1
            target_file = -1
            try:
                read_flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
                source_file = os.open(relative.name, read_flags, dir_fd=source_parent)
                source_before = os.fstat(source_file)
                if (
                    not stat.S_ISREG(source_before.st_mode)
                    or source_before.st_nlink != 1
                    or stat.S_IMODE(source_before.st_mode) != row["mode"]
                    or source_before.st_size != row["size"]
                ):
                    _fail("FAIL_CAMPAIGN_FILESYSTEM", "Cargo source file changed before copy", terminal=True)
                write_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
                target_file = os.open(relative.name, write_flags, int(row["mode"]), dir_fd=target_parent)
                digest = sha256()
                total = 0
                while True:
                    chunk = os.read(source_file, 1 << 20)
                    if not chunk:
                        break
                    digest.update(chunk)
                    total += len(chunk)
                    view = memoryview(chunk)
                    offset = 0
                    while offset < len(view):
                        written = os.write(target_file, view[offset:])
                        if written <= 0:
                            _fail("FAIL_CAMPAIGN_FILESYSTEM", "Cargo copy made no progress", terminal=True)
                        offset += written
                if total != row["size"] or digest.hexdigest() != row["raw_sha256_or_null"]:
                    _fail("FAIL_CAMPAIGN_FILESYSTEM", "Cargo source bytes changed during copy", terminal=True)
                source_after = os.fstat(source_file)
                if (
                    source_before.st_dev,
                    source_before.st_ino,
                    source_before.st_mode,
                    source_before.st_size,
                    source_before.st_mtime_ns,
                    source_before.st_ctime_ns,
                ) != (
                    source_after.st_dev,
                    source_after.st_ino,
                    source_after.st_mode,
                    source_after.st_size,
                    source_after.st_mtime_ns,
                    source_after.st_ctime_ns,
                ):
                    _fail("FAIL_CAMPAIGN_FILESYSTEM", "Cargo source file changed during copy", terminal=True)
                os.fchmod(target_file, int(row["mode"]))
                os.fsync(target_file)
                os.fsync(target_parent)
            finally:
                if source_file >= 0:
                    os.close(source_file)
                if target_file >= 0:
                    os.close(target_file)
                os.close(source_parent)
                os.close(target_parent)
        os.fsync(target_fd)
    finally:
        os.close(source_fd)
        os.close(target_fd)
    source_identity, source_rows = _scan_tree(
        source_root,
        label="Cargo source replay",
        allowed_uids=(os.geteuid(), 0),
        reject_group_or_world_writable=True,
    )
    target_identity, target_rows = _scan_tree(target_root, label="slot Cargo cache replay")
    source_content_rows = [
        {key: value for key, value in row.items() if key != "uid"}
        for row in source_rows
    ]
    target_content_rows = [
        {key: value for key, value in row.items() if key != "uid"}
        for row in target_rows
    ]
    if (
        list(expected_rows) != source_rows
        or source_content_rows != target_content_rows
        or any(row["uid"] != os.geteuid() for row in target_rows)
    ):
        _fail("FAIL_CAMPAIGN_FILESYSTEM", "Cargo source/target manifest rows differ", terminal=True)
    if (
        source_identity["content_manifest_digest"]
        != target_identity["content_manifest_digest"]
    ):
        _fail("FAIL_CAMPAIGN_FILESYSTEM", "Cargo source/target manifest digest differs", terminal=True)
    return target_identity


def _relative_entry_identity_or_none(
    root_fd: int,
    relative: PurePosixPath,
    label: str,
) -> dict[str, int] | None:
    if relative.is_absolute() or not relative.parts:
        _fail("FAIL_CAMPAIGN_SOURCE", f"{label} relative path differs")
    current = os.dup(root_fd)
    try:
        for part in relative.parent.parts:
            if part in ("", ".", ".."):
                _fail("FAIL_CAMPAIGN_SOURCE", f"{label} relative path differs")
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            try:
                next_fd = os.open(part, flags, dir_fd=current)
            except FileNotFoundError:
                return None
            os.close(current)
            current = next_fd
        try:
            observed = os.stat(
                relative.name,
                dir_fd=current,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return None
        return {
            "dev": observed.st_dev,
            "ino": observed.st_ino,
            "mode": observed.st_mode,
            "uid": observed.st_uid,
            "nlink": observed.st_nlink,
            "size": observed.st_size,
        }
    finally:
        os.close(current)


def _verify_git_source_self_contained(
    contract: Mapping[str, object],
    repository: str,
) -> dict[str, object]:
    common_text = _git(
        contract,
        ["-C", repository, "rev-parse", "--git-common-dir"],
        "target Git common directory replay",
    ).decode("utf-8").strip()
    git_dir_text = _git(
        contract,
        ["-C", repository, "rev-parse", "--absolute-git-dir"],
        "target Git worktree directory replay",
    ).decode("utf-8").strip()
    object_text = _git(
        contract,
        ["-C", repository, "rev-parse", "--git-path", "objects"],
        "target Git object directory replay",
    ).decode("utf-8").strip()
    shallow_text = _git(
        contract,
        ["-C", repository, "rev-parse", "--is-shallow-repository"],
        "target Git shallow-state replay",
    ).decode("ascii").strip()
    object_format_text = _git(
        contract,
        ["-C", repository, "rev-parse", "--show-object-format"],
        "target Git object-format replay",
    ).decode("ascii").strip()
    config_raw = _git(
        contract,
        [
            "-C",
            repository,
            "config",
            "--local",
            "--no-includes",
            "--null",
            "--list",
        ],
        "target Git local config replay",
    )
    common = Path(common_text)
    git_dir = Path(git_dir_text)
    objects = Path(object_text)
    for path, label in (
        (common, "Git common directory"),
        (git_dir, "Git worktree directory"),
        (objects, "Git object directory"),
    ):
        _normalized_absolute_parts(path, label)
    if (
        objects != common / "objects"
        or shallow_text != "false"
        or object_format_text != "sha1"
    ):
        _fail(
            "FAIL_CAMPAIGN_SOURCE",
            "target Git common-object, shallow, or object-format relation differs",
        )
    config_rows: list[dict[str, str]] = []
    for raw_row in config_raw.split(b"\x00"):
        if not raw_row:
            continue
        try:
            raw_key, raw_value = raw_row.split(b"\n", 1)
            key = raw_key.decode("utf-8").lower()
            value = raw_value.decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            _fail("FAIL_CAMPAIGN_SOURCE", "target Git local config row differs")
        config_rows.append({"key": key, "value": value})
        if (
            key == "extensions.partialclone"
            or key.startswith("include.")
            or key.startswith("includeif.")
            or key.startswith("fsck.")
            or re.fullmatch(r"remote\..+\.promisor", key) is not None
            or re.fullmatch(r"remote\..+\.partialclonefilter", key) is not None
        ):
            _fail(
                "FAIL_CAMPAIGN_SOURCE",
                "target Git include/partial/promisor/fsck-weakening config "
                f"is forbidden: {key}",
            )
    repository_formats = [
        row["value"]
        for row in config_rows
        if row["key"] == "core.repositoryformatversion"
    ]
    if repository_formats != ["0"]:
        _fail("FAIL_CAMPAIGN_SOURCE", "target Git repository format differs")
    common_fd = -1
    object_fd = -1
    git_dir_fd = -1
    try:
        common_fd, _common_status = _open_directory_nofollow(
            common,
            "target Git common directory",
        )
        object_fd, _object_status = _open_directory_nofollow(
            objects,
            "target Git object directory",
        )
        git_dir_fd, _git_dir_status = _open_directory_nofollow(
            git_dir,
            "target Git worktree directory",
        )
        common_identity = _verify_held_directory_path(
            common,
            common_fd,
            "target Git common directory",
        )
        object_identity = _verify_held_directory_path(
            objects,
            object_fd,
            "target Git object directory",
        )
        git_dir_identity = _verify_held_directory_path(
            git_dir,
            git_dir_fd,
            "target Git worktree directory",
        )
        forbidden_entries = {
            "objects_info_alternates": _relative_entry_identity_or_none(
                object_fd,
                PurePosixPath("info/alternates"),
                "Git alternates",
            ),
            "objects_info_http_alternates": _relative_entry_identity_or_none(
                object_fd,
                PurePosixPath("info/http-alternates"),
                "Git HTTP alternates",
            ),
            "common_shallow": _relative_entry_identity_or_none(
                common_fd,
                PurePosixPath("shallow"),
                "Git shallow file",
            ),
            "common_info_grafts": _relative_entry_identity_or_none(
                common_fd,
                PurePosixPath("info/grafts"),
                "Git grafts file",
            ),
        }
        if any(value is not None for value in forbidden_entries.values()):
            names = sorted(
                name for name, value in forbidden_entries.items() if value is not None
            )
            _fail(
                "FAIL_CAMPAIGN_SOURCE",
                "target Git external/shallow/graft metadata is forbidden: "
                + ",".join(names),
            )
        pack_names: list[str] = []
        pack_fd = -1
        try:
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            try:
                pack_fd = os.open("pack", flags, dir_fd=object_fd)
            except FileNotFoundError:
                pass
            if pack_fd >= 0:
                pack_names = sorted(os.listdir(pack_fd))
        finally:
            if pack_fd >= 0:
                os.close(pack_fd)
        promisor_names = [name for name in pack_names if name.endswith(".promisor")]
        if promisor_names:
            _fail(
                "FAIL_CAMPAIGN_SOURCE",
                "target Git promisor pack metadata is forbidden: "
                + ",".join(promisor_names[:16]),
            )
    finally:
        for descriptor in (git_dir_fd, object_fd, common_fd):
            if descriptor >= 0:
                os.close(descriptor)
    return {
        "git_common_dir": os.fspath(common),
        "git_common_dir_identity": common_identity,
        "git_worktree_dir": os.fspath(git_dir),
        "git_worktree_dir_identity": git_dir_identity,
        "git_object_dir": os.fspath(objects),
        "git_object_dir_identity": object_identity,
        "local_config_listing_sha256": sha256(config_raw).hexdigest(),
        "local_config_row_count": len(config_rows),
        "repository_format_version": 0,
        "object_format": object_format_text,
        "is_shallow_repository": False,
        "alternates_present": False,
        "http_alternates_present": False,
        "grafts_present": False,
        "promisor_metadata_present": False,
    }


def _verify_target_git_objects(
    contract: Mapping[str, object],
) -> dict[str, object]:
    target = contract["target_commit_a_binding"]
    execution = contract["execution_policy"]
    assert type(target) is dict and type(execution) is dict
    repository = str(execution["clone_source_repository"])
    commit = target["target_supervisor_commit"]
    if type(commit) is not str or _COMMIT.fullmatch(commit) is None:
        _fail("FAIL_CAMPAIGN_SOURCE", "target supervisor commit differs")
    source_repository = _verify_git_source_self_contained(
        contract,
        repository,
    )
    tree = _git(
        contract,
        ["-C", repository, "rev-parse", f"{commit}^{{tree}}"],
        "target tree replay",
    ).decode("ascii").strip()
    parent = _git(
        contract,
        ["-C", repository, "rev-parse", f"{commit}^"],
        "target parent replay",
    ).decode("ascii").strip()
    commit_object = _git(
        contract,
        ["-C", repository, "cat-file", "commit", commit],
        "target commit object replay",
    )
    if (
        tree != target["target_supervisor_tree"]
        or parent != target["target_supervisor_parent_commit"]
        or sha256(commit_object).hexdigest()
        != target["target_supervisor_commit_object_raw_sha256"]
    ):
        _fail("FAIL_CAMPAIGN_SOURCE", "target Git lineage differs")
    _git(
        contract,
        [
            "-C",
            repository,
            "fsck",
            "--full",
            "--strict",
            "--no-dangling",
            "--no-reflogs",
            commit,
        ],
        "target local object closure fsck",
        require_empty_stdout=True,
    )
    required_paths = target["required_paths_in_exact_order"]
    required_hashes = target["required_raw_sha256_by_path"]
    if (
        type(required_paths) is not list
        or type(required_hashes) is not dict
        or list(required_hashes) != required_paths
    ):
        _fail("FAIL_CAMPAIGN_SOURCE", "target path registry differs")
    observed_payloads: dict[str, bytes] = {}
    project_prefix = str(execution["clone_source_project_relative_path"])
    for relative in required_paths:
        if (
            type(relative) is not str
            or PurePosixPath(relative).is_absolute()
            or ".." in PurePosixPath(relative).parts
        ):
            _fail("FAIL_CAMPAIGN_SOURCE", "target relative path differs")
        object_path = (PurePosixPath(project_prefix) / relative).as_posix()
        payload = _git(
            contract,
            ["-C", repository, "show", f"{commit}:{object_path}"],
            f"target object replay {relative}",
            require_empty_stdout=False,
        )
        if sha256(payload).hexdigest() != required_hashes[relative]:
            _fail("FAIL_CAMPAIGN_SOURCE", f"target object hash differs: {relative}")
        observed_payloads[relative] = payload
    dependency_policy = _python310_dependency_preflight_policy_v1(contract)
    cargo_relative = str(dependency_policy["cargo_lock_relative_path"])
    cargo_object_path = (
        PurePosixPath(project_prefix) / cargo_relative
    ).as_posix()
    cargo_object_spec = f"{commit}:{cargo_object_path}"
    cargo_payload = _git(
        contract,
        ["-C", repository, "show", cargo_object_spec],
        "target Cargo.lock object replay",
    )
    current_cargo_payload, _current_cargo_status = _read_regular_nofollow(
        PROJECT_ROOT / cargo_relative,
        "current target Cargo.lock replay",
        1 * 1024 * 1024,
    )
    if (
        len(cargo_payload) != dependency_policy["cargo_lock_byte_count"]
        or sha256(cargo_payload).hexdigest()
        != dependency_policy["cargo_lock_raw_sha256"]
        or current_cargo_payload != cargo_payload
    ):
        _fail(
            "FAIL_CAMPAIGN_SOURCE",
            "target/current Cargo.lock object binding differs",
        )
    target_config_raw = observed_payloads[
        "config/phase3_q05b_dual_isolation_v2.json"
    ]
    decoded_target_config = _decoded_json_bytes(
        target_config_raw,
        "target supervisor config object",
    )
    if type(decoded_target_config) is not dict:
        _fail("FAIL_CAMPAIGN_SOURCE", "target supervisor config is not an object")
    static_policy = {
        key: value
        for key, value in decoded_target_config.items()
        if key not in {"engineering_status", "actual_preconditions"}
    }
    static_policy_root = sha256(
        _TARGET_STATIC_POLICY_DOMAIN + _canonical_json_bytes(static_policy)
    ).hexdigest()
    target_contract_digest = "sha256:" + sha256(target_config_raw).hexdigest()
    target_source_preimage = {
        "target_supervisor_commit": commit,
        "target_supervisor_tree": tree,
        "target_supervisor_parent_commit": parent,
        "target_supervisor_commit_object_raw_sha256": sha256(
            commit_object
        ).hexdigest(),
        "required_raw_sha256_by_path": deepcopy(required_hashes),
        "target_config_static_policy_root": static_policy_root,
        "target_contract_digest": target_contract_digest,
    }
    target_source_binding_digest = _domain_digest(
        _TARGET_SOURCE_BINDING_DOMAIN,
        target_source_preimage,
    )
    if (
        target.get("status") != "TARGET_COMMIT_A_BOUND"
        or target.get("target_config_static_policy_root")
        != static_policy_root
        or target.get("target_contract_digest") != target_contract_digest
        or target.get("target_source_binding_digest")
        != target_source_binding_digest
    ):
        _fail("FAIL_CAMPAIGN_SOURCE", "target derived binding differs")
    labels = execution["target_docker_label_policy"]
    if (
        type(labels) is not dict
        or labels.get("source_commit_label_value") != commit
        or labels.get("pre_spawn_and_post_reap_readonly_absence_filter")
        != f"label={labels.get('source_commit_label_key')}={commit}"
    ):
        _fail("FAIL_CAMPAIGN_CONTRACT", "target Docker label binding differs")
    listing = _git(
        contract,
        ["-C", repository, "ls-tree", "-r", "-z", "--full-tree", commit],
        "target tree special-entry replay",
    )
    for entry in listing.split(b"\x00"):
        if not entry:
            continue
        metadata, path = entry.split(b"\t", 1)
        mode, kind, _object_id = metadata.split(b" ", 2)
        if mode in (b"120000", b"160000") or kind == b"commit":
            _fail(
                "FAIL_CAMPAIGN_SOURCE",
                f"target tree contains symlink or gitlink: {path!r}",
            )
    return {
        "commit": commit,
        "tree": tree,
        "parent": parent,
        "commit_object_raw_sha256": sha256(commit_object).hexdigest(),
        "required_raw_sha256_by_path": deepcopy(required_hashes),
        "target_config_static_policy_root": static_policy_root,
        "target_contract_digest": target_contract_digest,
        "target_source_binding_digest": target_source_binding_digest,
        "cargo_lock_git_object": {
            "object_spec": cargo_object_spec,
            "project_relative_path": cargo_relative,
            "byte_count": len(cargo_payload),
            "raw_sha256": sha256(cargo_payload).hexdigest(),
            "matches_current_controller_worktree": True,
        },
        "local_object_closure_fsck_passed": True,
        "source_repository_self_containment": source_repository,
    }


def _controller_git_identity(contract: Mapping[str, object]) -> dict[str, object]:
    execution = contract["execution_policy"]
    target = contract["target_commit_a_binding"]
    assert type(execution) is dict and type(target) is dict
    repository = str(execution["clone_source_repository"])
    head = _git(
        contract,
        ["-C", repository, "rev-parse", "HEAD"],
        "controller HEAD replay",
    ).decode("ascii").strip()
    tree = _git(
        contract,
        ["-C", repository, "rev-parse", "HEAD^{tree}"],
        "controller tree replay",
    ).decode("ascii").strip()
    parent = _git(
        contract,
        ["-C", repository, "rev-parse", "HEAD^"],
        "controller parent replay",
    ).decode("ascii").strip()
    lineage_raw = _git(
        contract,
        ["-C", repository, "rev-list", "--parents", "-n", "1", "HEAD"],
        "controller exact one-parent lineage replay",
    )
    status_raw = _git(
        contract,
        ["-C", repository, "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        "controller cleanliness replay",
    )
    top = _git(
        contract,
        ["-C", repository, "rev-parse", "--show-toplevel"],
        "controller toplevel replay",
    ).decode("utf-8").strip()
    changed_raw = _git(
        contract,
        [
            "-C",
            repository,
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "--no-renames",
            "-r",
            "-z",
            "HEAD^",
            "HEAD",
        ],
        "controller exact six-path diff-tree replay",
    )
    project_prefix = str(execution["clone_source_project_relative_path"])
    expected_changed_from_repository = [
        (PurePosixPath(project_prefix) / relative).as_posix()
        for relative in _CONTROLLER_CHANGED_PATHS_FROM_TARGET_IN_EXACT_ORDER
    ]
    expected_changed_raw = b"".join(
        os.fsencode(relative) + b"\x00"
        for relative in expected_changed_from_repository
    )
    expected_parent = target.get("target_supervisor_commit")
    if (
        _COMMIT.fullmatch(head) is None
        or _COMMIT.fullmatch(tree) is None
        or _COMMIT.fullmatch(parent) is None
        or _COMMIT.fullmatch(str(expected_parent)) is None
        or parent != expected_parent
        or lineage_raw
        != f"{head} {expected_parent}\n".encode("ascii")
        or status_raw != b""
        or top != repository
        or changed_raw != expected_changed_raw
    ):
        _fail(
            "FAIL_CAMPAIGN_SOURCE",
            "controller Git identity, cleanliness, exact Commit-B parent, or "
            "exact six-path diff binding to target Commit A differs",
        )
    return {
        "commit": head,
        "tree": tree,
        "parent": parent,
        "parent_count": 1,
        "clean": True,
        "changed_paths_from_target_in_exact_order": list(
            _CONTROLLER_CHANGED_PATHS_FROM_TARGET_IN_EXACT_ORDER
        ),
    }


def _python310_dependency_preflight_v1(
    contract: Mapping[str, object],
    *,
    target_git: Mapping[str, object],
    controller_git: Mapping[str, object],
    python_identity: Mapping[str, object],
) -> dict[str, object]:
    """Prove the isolated 3.10 artifact path and real Cargo.lock parse."""

    policy = _python310_dependency_preflight_policy_v1(contract)
    target_binding = contract["target_commit_a_binding"]
    execution = contract["execution_policy"]
    assert type(target_binding) is dict and type(execution) is dict
    cargo_git = target_git.get("cargo_lock_git_object")
    expected_cargo_object_spec = (
        f"{target_binding['target_supervisor_commit']}:"
        f"{execution['clone_source_project_relative_path']}/"
        f"{policy['cargo_lock_relative_path']}"
    )
    if (
        type(cargo_git) is not dict
        or target_git.get("commit")
        != target_binding["target_supervisor_commit"]
        or target_git.get("target_source_binding_digest")
        != target_binding["target_source_binding_digest"]
        or target_git.get("required_raw_sha256_by_path")
        != target_binding["required_raw_sha256_by_path"]
        or cargo_git.get("object_spec") != expected_cargo_object_spec
        or cargo_git.get("project_relative_path")
        != policy["cargo_lock_relative_path"]
        or cargo_git.get("byte_count") != policy["cargo_lock_byte_count"]
        or cargo_git.get("raw_sha256") != policy["cargo_lock_raw_sha256"]
        or cargo_git.get("matches_current_controller_worktree") is not True
        or controller_git.get("parent")
        != target_binding["target_supervisor_commit"]
        or controller_git.get("parent_count") != 1
        or controller_git.get("clean") is not True
        or controller_git.get("changed_paths_from_target_in_exact_order")
        != list(_CONTROLLER_CHANGED_PATHS_FROM_TARGET_IN_EXACT_ORDER)
        or python_identity.get("path") != policy["interpreter_path"]
        or type(python_identity.get("raw_sha256")) is not str
        or _RAW_SHA256.fullmatch(str(python_identity["raw_sha256"])) is None
    ):
        _fail(
            "FAIL_CAMPAIGN_SOURCE",
            "dependency preflight provenance binding differs",
        )
    supervisor_relative = str(policy["supervisor_relative_path"])
    artifact_relative = str(policy["artifact_relative_path"])
    supervisor_raw, _supervisor_status = _read_regular_nofollow(
        PROJECT_ROOT / supervisor_relative,
        "dependency preflight supervisor source",
        32 * 1024 * 1024,
    )
    artifact_raw, _artifact_status = _read_regular_nofollow(
        PROJECT_ROOT / artifact_relative,
        "dependency preflight artifact source",
        8 * 1024 * 1024,
    )
    target_hashes = target_binding["required_raw_sha256_by_path"]
    if (
        type(target_hashes) is not dict
        or sha256(supervisor_raw).hexdigest()
        != target_hashes.get(supervisor_relative)
        or sha256(artifact_raw).hexdigest()
        != target_hashes.get(artifact_relative)
    ):
        _fail(
            "FAIL_CAMPAIGN_SOURCE",
            "dependency preflight supervisor/artifact source differs",
        )
    argv = [
        *list(_DEPENDENCY_PREFLIGHT_ARGV_PREFIX_V1),
        _PYTHON310_DEPENDENCY_PREFLIGHT_PROGRAM_V1,
        os.fspath(PROJECT_ROOT),
        os.fspath(PROJECT_ROOT / str(policy["cargo_lock_relative_path"])),
    ]
    runtime_environment = _runtime_environment(contract)
    result = _bounded_process(
        argv,
        cwd=str(policy["cwd"]),
        env=runtime_environment,
        timeout_seconds=float(policy["timeout_seconds"]),
        stdout_max_bytes=int(policy["stdout_max_bytes"]),
        stderr_max_bytes=int(policy["stderr_max_bytes"]),
        reap_seconds=float(execution["sigkill_reap_seconds"]),
    )
    process_facts = {
        key: result.get(key)
        for key in (
            "return_code",
            "timed_out",
            "stdout_overflow",
            "stderr_overflow",
            "child_reaped",
            "process_group_absent",
            "wrapper_resources_closed",
        )
    }
    stdout = result.get("stdout")
    stderr = result.get("stderr")
    if (
        process_facts
        != {
            "return_code": 0,
            "timed_out": False,
            "stdout_overflow": False,
            "stderr_overflow": False,
            "child_reaped": True,
            "process_group_absent": True,
            "wrapper_resources_closed": True,
        }
        or type(stdout) is not bytes
        or type(stderr) is not bytes
        or stderr != b""
    ):
        _fail(
            "FAIL_CAMPAIGN_SOURCE",
            "isolated Python dependency child process differs",
        )
    child = _strict_json(stdout, "isolated Python dependency child receipt")
    expected_child_fields = {
        "schema_version",
        "status",
        "python_flags",
        "tomli_top_level_spec_or_null",
        "artifact_module_name",
        "vendored_tomli_module_name",
        "top_level_tomli_in_sys_modules",
        "host_replay_module_name",
        "host_source_allowlist_count",
        "host_source_allowlist_canonical_lf_sha256",
        "vendor_closure_rows",
        "vendor_closure_digest",
        "loaded_vendored_modules",
        "cargo_lock",
    }
    expected_cargo_fields = {
        "relative_path",
        "byte_count",
        "raw_sha256",
        "format",
        "package_count",
        "registry_package_count",
        "registry_package_rows",
        "registry_projection_digest",
    }
    if (
        type(child) is not dict
        or set(child) != expected_child_fields
        or stdout != _canonical_json_bytes(child)
    ):
        _fail("FAIL_CAMPAIGN_SOURCE", "dependency child receipt fields differ")
    child_cargo = child["cargo_lock"]
    registry_rows = (
        child_cargo.get("registry_package_rows")
        if type(child_cargo) is dict
        else None
    )
    registry_rows_valid = (
        type(registry_rows) is list
        and len(registry_rows) == policy["cargo_lock_registry_package_count"]
        and all(
            type(row) is list
            and len(row) == 4
            and all(type(item) is str for item in row)
            and row[2].startswith("registry+")
            and _RAW_SHA256.fullmatch(row[3]) is not None
            for row in registry_rows
        )
        and registry_rows == sorted(registry_rows)
    )
    registry_projection_digest = (
        _domain_digest(
            str(policy["cargo_lock_registry_projection_domain"]),
            registry_rows,
        )
        if registry_rows_valid
        else None
    )
    if (
        child.get("schema_version")
        != _PYTHON310_DEPENDENCY_PREFLIGHT_CHILD_SCHEMA_V1
        or child.get("status") != "PASS"
        or not _json_value_type_exact_v1(
            child.get("python_flags"),
            {
                "isolated": 1,
                "no_site": 1,
                "ignore_environment": 1,
                "dont_write_bytecode": True,
                "utf8_mode": 1,
                "pycache_prefix": _DEPENDENCY_PREFLIGHT_PYCACHE_PREFIX_V1,
                "pycache_prefix_absent_before_import": True,
                "pycache_prefix_absent_after_import": True,
            },
        )
        or child.get("tomli_top_level_spec_or_null") is not None
        or child.get("artifact_module_name")
        != "hegel_machine.phase3_q05b_actual_artifact_v2"
        or child.get("vendored_tomli_module_name")
        != policy["vendored_tomli_module_name"]
        or child.get("top_level_tomli_in_sys_modules") is not False
        or child.get("host_replay_module_name")
        != "hegel_machine.phase3_q05b_host_replay_v1"
        or child.get("host_source_allowlist_count")
        != policy["host_source_allowlist_count"]
        or child.get("host_source_allowlist_canonical_lf_sha256")
        != policy["host_source_allowlist_canonical_lf_sha256"]
        or not _json_value_type_exact_v1(
            child.get("vendor_closure_rows"),
            policy["vendor_closure_rows_in_exact_order"],
        )
        or child.get("vendor_closure_digest")
        != policy["vendor_closure_digest"]
        or child.get("loaded_vendored_modules")
        != policy["loaded_vendored_modules_in_exact_order"]
        or type(child_cargo) is not dict
        or set(child_cargo) != expected_cargo_fields
        or child_cargo.get("relative_path")
        != policy["cargo_lock_relative_path"]
        or type(child_cargo.get("byte_count")) is not int
        or child_cargo.get("byte_count") != policy["cargo_lock_byte_count"]
        or child_cargo.get("raw_sha256") != policy["cargo_lock_raw_sha256"]
        or type(child_cargo.get("format")) is not int
        or child_cargo.get("format") != policy["cargo_lock_format"]
        or type(child_cargo.get("package_count")) is not int
        or child_cargo.get("package_count")
        != policy["cargo_lock_package_count"]
        or type(child_cargo.get("registry_package_count")) is not int
        or child_cargo.get("registry_package_count")
        != policy["cargo_lock_registry_package_count"]
        or not registry_rows_valid
        or child_cargo.get("registry_projection_digest")
        != policy["cargo_lock_registry_projection_digest"]
        or registry_projection_digest
        != policy["cargo_lock_registry_projection_digest"]
    ):
        _fail("FAIL_CAMPAIGN_SOURCE", "dependency child semantics differ")
    receipt: dict[str, object] = {
        "schema_version": (
            _PYTHON310_DEPENDENCY_PREFLIGHT_RECEIPT_SCHEMA_V1
        ),
        "status": "PASS",
        "target_commit": target_git["commit"],
        "target_source_binding_digest": target_git[
            "target_source_binding_digest"
        ],
        "controller_commit": controller_git["commit"],
        "controller_changed_paths_from_target_in_exact_order": list(
            _CONTROLLER_CHANGED_PATHS_FROM_TARGET_IN_EXACT_ORDER
        ),
        "python_executable": {
            "path": python_identity["path"],
            "raw_sha256": python_identity["raw_sha256"],
        },
        "argv_prefix_before_program": list(
            _DEPENDENCY_PREFLIGHT_ARGV_PREFIX_V1
        ),
        "program_raw_sha256": policy["program_raw_sha256"],
        "runtime_environment_digest": _domain_digest(
            str(policy["runtime_environment_digest_domain"]),
            runtime_environment,
        ),
        "supervisor_source": {
            "relative_path": supervisor_relative,
            "raw_sha256": sha256(supervisor_raw).hexdigest(),
        },
        "artifact_source": {
            "relative_path": artifact_relative,
            "raw_sha256": sha256(artifact_raw).hexdigest(),
        },
        "artifact_module_name": child["artifact_module_name"],
        "host_replay_module_name": child["host_replay_module_name"],
        "vendored_tomli_module_name": child["vendored_tomli_module_name"],
        "tomli_top_level_spec_or_null": child[
            "tomli_top_level_spec_or_null"
        ],
        "top_level_tomli_in_sys_modules": child[
            "top_level_tomli_in_sys_modules"
        ],
        "python_flags": deepcopy(child["python_flags"]),
        "vendor_closure_rows": deepcopy(child["vendor_closure_rows"]),
        "vendor_closure_digest": child["vendor_closure_digest"],
        "loaded_vendored_modules": deepcopy(
            child["loaded_vendored_modules"]
        ),
        "host_source_allowlist_count": child[
            "host_source_allowlist_count"
        ],
        "host_source_allowlist_canonical_lf_sha256": child[
            "host_source_allowlist_canonical_lf_sha256"
        ],
        "cargo_lock_git_object": deepcopy(cargo_git),
        "cargo_lock_parse": deepcopy(child_cargo),
        "child_receipt": deepcopy(child),
        "process_facts": process_facts,
        "stdout": {
            "byte_count": len(stdout),
            "raw_sha256": sha256(stdout).hexdigest(),
            "canonical_single_lf": True,
        },
        "stderr": {
            "byte_count": 0,
            "raw_sha256": sha256(b"").hexdigest(),
            "empty": True,
        },
        "receipt_digest": None,
    }
    unsigned = deepcopy(receipt)
    unsigned.pop("receipt_digest")
    receipt["receipt_digest"] = _domain_digest(
        str(policy["receipt_digest_domain"]),
        unsigned,
    )
    return _validate_dependency_receipt_v1(
        contract,
        receipt,
        expected_target_git=target_git,
        expected_controller_git=controller_git,
        expected_python_identity=python_identity,
    )


def _validate_dependency_receipt_v1(
    contract: Mapping[str, object],
    value: object,
    *,
    expected_target_git: object,
    expected_controller_git: object,
    expected_python_identity: object,
) -> dict[str, object]:
    """Producer-free replay against the capsule's independently bound inputs."""

    policy = _python310_dependency_preflight_policy_v1(contract)
    target = contract["target_commit_a_binding"]
    execution = contract["execution_policy"]
    assert type(target) is dict and type(execution) is dict
    expected_fields = {
        "schema_version",
        "status",
        "target_commit",
        "target_source_binding_digest",
        "controller_commit",
        "controller_changed_paths_from_target_in_exact_order",
        "python_executable",
        "argv_prefix_before_program",
        "program_raw_sha256",
        "runtime_environment_digest",
        "supervisor_source",
        "artifact_source",
        "artifact_module_name",
        "host_replay_module_name",
        "vendored_tomli_module_name",
        "tomli_top_level_spec_or_null",
        "top_level_tomli_in_sys_modules",
        "python_flags",
        "vendor_closure_rows",
        "vendor_closure_digest",
        "loaded_vendored_modules",
        "host_source_allowlist_count",
        "host_source_allowlist_canonical_lf_sha256",
        "cargo_lock_git_object",
        "cargo_lock_parse",
        "child_receipt",
        "process_facts",
        "stdout",
        "stderr",
        "receipt_digest",
    }
    if type(value) is not dict or set(value) != expected_fields:
        _fail("FAIL_CAMPAIGN_SOURCE", "dependency receipt fields differ")
    receipt = value
    if (
        type(expected_target_git) is not dict
        or type(expected_controller_git) is not dict
        or type(expected_python_identity) is not dict
    ):
        _fail(
            "FAIL_CAMPAIGN_SOURCE",
            "dependency receipt expected context fields differ",
        )
    python_executable = receipt["python_executable"]
    supervisor_source = receipt["supervisor_source"]
    artifact_source = receipt["artifact_source"]
    cargo_git = receipt["cargo_lock_git_object"]
    cargo_parse = receipt["cargo_lock_parse"]
    process_facts = receipt["process_facts"]
    stdout = receipt["stdout"]
    stderr = receipt["stderr"]
    child_receipt = receipt["child_receipt"]
    rows = (
        cargo_parse.get("registry_package_rows")
        if type(cargo_parse) is dict
        else None
    )
    rows_valid = (
        type(rows) is list
        and len(rows) == policy["cargo_lock_registry_package_count"]
        and all(
            type(row) is list
            and len(row) == 4
            and all(type(item) is str for item in row)
            and row[2].startswith("registry+")
            and _RAW_SHA256.fullmatch(row[3]) is not None
            for row in rows
        )
        and rows == sorted(rows)
    )
    expected_cargo_object_spec = (
        f"{target['target_supervisor_commit']}:"
        f"{execution['clone_source_project_relative_path']}/"
        f"{policy['cargo_lock_relative_path']}"
    )
    target_hashes = target["required_raw_sha256_by_path"]
    expected_runtime_environment_digest = _domain_digest(
        str(policy["runtime_environment_digest_domain"]),
        _runtime_environment(contract),
    )
    expected_cargo_git = {
        "object_spec": expected_cargo_object_spec,
        "project_relative_path": policy["cargo_lock_relative_path"],
        "byte_count": policy["cargo_lock_byte_count"],
        "raw_sha256": policy["cargo_lock_raw_sha256"],
        "matches_current_controller_worktree": True,
    }
    if (
        receipt.get("schema_version")
        != _PYTHON310_DEPENDENCY_PREFLIGHT_RECEIPT_SCHEMA_V1
        or receipt.get("status") != "PASS"
        or receipt.get("target_commit") != target["target_supervisor_commit"]
        or receipt.get("target_source_binding_digest")
        != target["target_source_binding_digest"]
        or expected_target_git.get("commit") != receipt.get("target_commit")
        or expected_target_git.get("target_source_binding_digest")
        != receipt.get("target_source_binding_digest")
        or expected_target_git.get("required_raw_sha256_by_path")
        != target_hashes
        or expected_target_git.get("cargo_lock_git_object")
        != expected_cargo_git
        or receipt.get("controller_commit")
        != expected_controller_git.get("commit")
        or type(receipt.get("controller_commit")) is not str
        or _COMMIT.fullmatch(str(receipt["controller_commit"])) is None
        or expected_controller_git.get("parent")
        != target["target_supervisor_commit"]
        or expected_controller_git.get("parent_count") != 1
        or expected_controller_git.get("clean") is not True
        or expected_controller_git.get(
            "changed_paths_from_target_in_exact_order"
        )
        != list(_CONTROLLER_CHANGED_PATHS_FROM_TARGET_IN_EXACT_ORDER)
        or receipt.get("controller_changed_paths_from_target_in_exact_order")
        != list(_CONTROLLER_CHANGED_PATHS_FROM_TARGET_IN_EXACT_ORDER)
        or type(python_executable) is not dict
        or set(python_executable) != {"path", "raw_sha256"}
        or python_executable.get("path") != policy["interpreter_path"]
        or python_executable.get("path")
        != expected_python_identity.get("path")
        or type(python_executable.get("raw_sha256")) is not str
        or _RAW_SHA256.fullmatch(str(python_executable["raw_sha256"])) is None
        or python_executable.get("raw_sha256")
        != expected_python_identity.get("raw_sha256")
        or receipt.get("argv_prefix_before_program")
        != policy["argv_prefix_before_program"]
        or receipt.get("program_raw_sha256") != policy["program_raw_sha256"]
        or receipt.get("runtime_environment_digest")
        != expected_runtime_environment_digest
        or type(supervisor_source) is not dict
        or supervisor_source
        != {
            "relative_path": policy["supervisor_relative_path"],
            "raw_sha256": target_hashes[policy["supervisor_relative_path"]],
        }
        or type(artifact_source) is not dict
        or artifact_source
        != {
            "relative_path": policy["artifact_relative_path"],
            "raw_sha256": target_hashes[policy["artifact_relative_path"]],
        }
        or receipt.get("artifact_module_name")
        != "hegel_machine.phase3_q05b_actual_artifact_v2"
        or receipt.get("host_replay_module_name")
        != "hegel_machine.phase3_q05b_host_replay_v1"
        or receipt.get("vendored_tomli_module_name")
        != policy["vendored_tomli_module_name"]
        or receipt.get("tomli_top_level_spec_or_null") is not None
        or receipt.get("top_level_tomli_in_sys_modules") is not False
        or not _json_value_type_exact_v1(
            receipt.get("python_flags"),
            {
                "isolated": 1,
                "no_site": 1,
                "ignore_environment": 1,
                "dont_write_bytecode": True,
                "utf8_mode": 1,
                "pycache_prefix": _DEPENDENCY_PREFLIGHT_PYCACHE_PREFIX_V1,
                "pycache_prefix_absent_before_import": True,
                "pycache_prefix_absent_after_import": True,
            },
        )
        or not _json_value_type_exact_v1(
            receipt.get("vendor_closure_rows"),
            policy["vendor_closure_rows_in_exact_order"],
        )
        or receipt.get("vendor_closure_digest")
        != policy["vendor_closure_digest"]
        or receipt.get("loaded_vendored_modules")
        != policy["loaded_vendored_modules_in_exact_order"]
        or type(receipt.get("host_source_allowlist_count")) is not int
        or receipt.get("host_source_allowlist_count")
        != policy["host_source_allowlist_count"]
        or receipt.get("host_source_allowlist_canonical_lf_sha256")
        != policy["host_source_allowlist_canonical_lf_sha256"]
        or not _json_value_type_exact_v1(cargo_git, expected_cargo_git)
        or type(cargo_parse) is not dict
        or set(cargo_parse)
        != {
            "relative_path",
            "byte_count",
            "raw_sha256",
            "format",
            "package_count",
            "registry_package_count",
            "registry_package_rows",
            "registry_projection_digest",
        }
        or cargo_parse.get("relative_path")
        != policy["cargo_lock_relative_path"]
        or type(cargo_parse.get("byte_count")) is not int
        or cargo_parse.get("byte_count") != policy["cargo_lock_byte_count"]
        or cargo_parse.get("raw_sha256") != policy["cargo_lock_raw_sha256"]
        or type(cargo_parse.get("format")) is not int
        or cargo_parse.get("format") != policy["cargo_lock_format"]
        or type(cargo_parse.get("package_count")) is not int
        or cargo_parse.get("package_count") != policy["cargo_lock_package_count"]
        or type(cargo_parse.get("registry_package_count")) is not int
        or cargo_parse.get("registry_package_count")
        != policy["cargo_lock_registry_package_count"]
        or not rows_valid
        or cargo_parse.get("registry_projection_digest")
        != policy["cargo_lock_registry_projection_digest"]
        or _domain_digest(
            str(policy["cargo_lock_registry_projection_domain"]),
            rows,
        )
        != policy["cargo_lock_registry_projection_digest"]
        or not _json_value_type_exact_v1(
            process_facts,
            {
                "return_code": 0,
                "timed_out": False,
                "stdout_overflow": False,
                "stderr_overflow": False,
                "child_reaped": True,
                "process_group_absent": True,
                "wrapper_resources_closed": True,
            },
        )
        or not _json_value_type_exact_v1(
            stderr,
            {
                "byte_count": 0,
                "raw_sha256": sha256(b"").hexdigest(),
                "empty": True,
            },
        )
        or type(receipt.get("receipt_digest")) is not str
        or _TAGGED_SHA256.fullmatch(str(receipt["receipt_digest"])) is None
    ):
        _fail("FAIL_CAMPAIGN_SOURCE", "dependency receipt semantics differ")
    expected_child_receipt = {
        "schema_version": _PYTHON310_DEPENDENCY_PREFLIGHT_CHILD_SCHEMA_V1,
        "status": "PASS",
        "python_flags": deepcopy(receipt["python_flags"]),
        "tomli_top_level_spec_or_null": receipt[
            "tomli_top_level_spec_or_null"
        ],
        "artifact_module_name": receipt["artifact_module_name"],
        "vendored_tomli_module_name": receipt["vendored_tomli_module_name"],
        "top_level_tomli_in_sys_modules": receipt[
            "top_level_tomli_in_sys_modules"
        ],
        "host_replay_module_name": receipt["host_replay_module_name"],
        "host_source_allowlist_count": receipt[
            "host_source_allowlist_count"
        ],
        "host_source_allowlist_canonical_lf_sha256": receipt[
            "host_source_allowlist_canonical_lf_sha256"
        ],
        "vendor_closure_rows": deepcopy(receipt["vendor_closure_rows"]),
        "vendor_closure_digest": receipt["vendor_closure_digest"],
        "loaded_vendored_modules": deepcopy(
            receipt["loaded_vendored_modules"]
        ),
        "cargo_lock": deepcopy(receipt["cargo_lock_parse"]),
    }
    expected_stdout = _canonical_json_bytes(expected_child_receipt)
    if (
        not _json_value_type_exact_v1(
            child_receipt,
            expected_child_receipt,
        )
        or not _json_value_type_exact_v1(
            stdout,
            {
                "byte_count": len(expected_stdout),
                "raw_sha256": sha256(expected_stdout).hexdigest(),
                "canonical_single_lf": True,
            },
        )
    ):
        _fail(
            "FAIL_CAMPAIGN_SOURCE",
            "dependency child/stdout receipt replay differs",
        )
    unsigned = deepcopy(receipt)
    observed_digest = unsigned.pop("receipt_digest")
    if observed_digest != _domain_digest(
        str(policy["receipt_digest_domain"]),
        unsigned,
    ):
        _fail("FAIL_CAMPAIGN_SOURCE", "dependency receipt digest differs")
    return deepcopy(receipt)


def _verify_clone(
    contract: Mapping[str, object],
    slot: Mapping[str, object],
    *,
    artifact_state: str,
) -> dict[str, object]:
    target = contract["target_commit_a_binding"]
    assert type(target) is dict
    clone_root = Path(str(slot["clone_root"]))
    project_root = Path(str(slot["clone_project_root"]))
    if project_root != clone_root / PROJECT_ROOT.name:
        _fail("FAIL_CAMPAIGN_SOURCE", "slot clone project root differs", terminal=True)
    clone_fd, clone_stat = _open_directory_nofollow(clone_root, "slot clone root")
    os.close(clone_fd)
    commit_graph_raw, _commit_graph_stat = _read_regular_nofollow(
        clone_root / ".git/objects/info/commit-graph",
        "slot clone reachable commit-graph",
        256 * 1024 * 1024,
    )
    if not commit_graph_raw:
        _fail("FAIL_CAMPAIGN_SOURCE", "slot clone commit-graph is empty", terminal=True)
    _git(
        contract,
        ["-C", os.fspath(clone_root), "commit-graph", "verify"],
        "slot clone commit-graph verify",
        require_empty_stdout=True,
        git_worktree=os.fspath(clone_root),
    )
    head = _git(
        contract,
        ["-C", os.fspath(clone_root), "rev-parse", "HEAD"],
        "slot clone HEAD replay",
        git_worktree=os.fspath(clone_root),
    ).decode("ascii").strip()
    tree = _git(
        contract,
        ["-C", os.fspath(clone_root), "rev-parse", "HEAD^{tree}"],
        "slot clone tree replay",
        git_worktree=os.fspath(clone_root),
    ).decode("ascii").strip()
    parent = _git(
        contract,
        ["-C", os.fspath(clone_root), "rev-parse", "HEAD^"],
        "slot clone parent replay",
        git_worktree=os.fspath(clone_root),
    ).decode("ascii").strip()
    tracked = _git(
        contract,
        ["-C", os.fspath(clone_root), "status", "--porcelain=v1", "-z", "--untracked-files=no"],
        "slot clone tracked cleanliness replay",
        git_worktree=os.fspath(clone_root),
    )
    untracked = _git(
        contract,
        ["-C", os.fspath(clone_root), "ls-files", "--others", "-z"],
        "slot clone untracked replay",
        git_worktree=os.fspath(clone_root),
    )
    remotes = _git(
        contract,
        ["-C", os.fspath(clone_root), "remote"],
        "slot clone remote replay",
        git_worktree=os.fspath(clone_root),
    )
    branch = _git(
        contract,
        ["-C", os.fspath(clone_root), "branch", "--show-current"],
        "slot clone detached replay",
        git_worktree=os.fspath(clone_root),
    )
    refs = _git(
        contract,
        [
            "-C",
            os.fspath(clone_root),
            "for-each-ref",
            "--format=%(refname)%00%(objectname)",
        ],
        "slot clone ref replay",
        git_worktree=os.fspath(clone_root),
    )
    expected_ref = (
        str(contract["execution_policy"]["clone_target_ref_name"])
        + "\x00"
        + str(target["target_supervisor_commit"])
        + "\n"
    ).encode("ascii")
    graph_enabled_fsck = _git(
        contract,
        [
            "-C",
            os.fspath(clone_root),
            "fsck",
            "--full",
            "--strict",
            "--unreachable",
            "--no-reflogs",
        ],
        "slot clone graph-enabled strict fsck",
        require_empty_stdout=True,
        git_worktree=os.fspath(clone_root),
    )
    graph_disabled_fsck = _git(
        contract,
        [
            "-c",
            "core.commitGraph=false",
            "-C",
            os.fspath(clone_root),
            "fsck",
            "--full",
            "--strict",
            "--unreachable",
            "--no-reflogs",
        ],
        "slot clone graph-disabled strict fsck",
        require_empty_stdout=True,
        git_worktree=os.fspath(clone_root),
    )
    expected_untracked = b""
    artifact = Path(str(slot["supervisor_artifact_path"]))
    if artifact_state in {"VALID", "RESIDUAL_INVALID", "PRESENT"}:
        expected_untracked = artifact.relative_to(clone_root).as_posix().encode("utf-8") + b"\x00"
    elif artifact_state != "ABSENT":
        _fail("FAIL_CAMPAIGN_SOURCE", "slot clone artifact-state input differs", terminal=True)
    if (
        head != target["target_supervisor_commit"]
        or tree != target["target_supervisor_tree"]
        or parent != target["target_supervisor_parent_commit"]
        or tracked != b""
        or untracked != expected_untracked
        or remotes != b""
        or branch != b""
        or refs != expected_ref
        or graph_enabled_fsck != b""
        or graph_disabled_fsck != b""
        or stat.S_IMODE(clone_stat.st_mode) not in (0o700, 0o755)
        or clone_stat.st_uid != os.geteuid()
    ):
        _fail("FAIL_CAMPAIGN_SOURCE", "slot clone replay differs", terminal=True)
    if os.path.lexists(clone_root / ".git/objects/info/alternates"):
        _fail("FAIL_CAMPAIGN_SOURCE", "slot clone retains Git alternates", terminal=True)
    for relative, expected in target["required_raw_sha256_by_path"].items():
        raw, status = _read_regular_nofollow(
            project_root / str(relative),
            f"slot clone target {relative}",
            256 * 1024 * 1024,
        )
        if sha256(raw).hexdigest() != expected or status.st_nlink != 1:
            _fail("FAIL_CAMPAIGN_SOURCE", f"slot clone target differs: {relative}", terminal=True)
    clone_tree_identity, _clone_tree_rows = _scan_tree(
        clone_root,
        label="slot clone no-alias tree replay",
        maximum_entries=200_000,
        manifest_domain=(
            "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R8/CLONE_TREE_MANIFEST/V2"
        ),
    )
    artifact_relative = artifact.relative_to(clone_root).as_posix()
    closure_rows_without_artifact = [
        row
        for row in _clone_tree_rows
        if row["relative_path"] != artifact_relative
    ]
    closure_without_artifact = _domain_digest(
        "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R8/"
        "CLONE_CLOSURE_WITHOUT_ARTIFACT/V2",
        closure_rows_without_artifact,
    )
    return {
        "clone_root_dev": clone_stat.st_dev,
        "clone_root_ino": clone_stat.st_ino,
        "head": head,
        "tree": tree,
        "parent": parent,
        "artifact_state": artifact_state,
        "clone_tree_manifest_digest": clone_tree_identity[
            "manifest_digest"
        ],
        "clone_closure_without_artifact_digest": closure_without_artifact,
        "clone_tree_entry_count": len(_clone_tree_rows),
        "clone_closure_rows_internal": closure_rows_without_artifact,
    }


def _prepare_clone(
    contract: Mapping[str, object],
    slot: Mapping[str, object],
) -> dict[str, object]:
    execution = contract["execution_policy"]
    target = contract["target_commit_a_binding"]
    assert type(execution) is dict and type(target) is dict
    source = str(execution["clone_source_repository"])
    destination = str(slot["clone_root"])
    if os.path.lexists(destination):
        _fail("FAIL_CAMPAIGN_FRESHNESS", "slot clone already exists", terminal=True)
    _git(
        contract,
        [
            "clone",
            "--no-checkout",
            "--no-hardlinks",
            "--no-tags",
            "--no-local",
            "--quiet",
            source,
            destination,
        ],
        "slot no-local clone",
        require_empty_stdout=True,
        git_worktree=destination,
    )
    _git(
        contract,
        [
            "-C",
            destination,
            "checkout",
            "--detach",
            "--quiet",
            str(target["target_supervisor_commit"]),
        ],
        "slot detached checkout",
        require_empty_stdout=True,
        git_worktree=destination,
    )
    _git(
        contract,
        ["-C", destination, "remote", "remove", "origin"],
        "slot remote removal",
        require_empty_stdout=True,
        git_worktree=destination,
    )
    refs = _git(
        contract,
        ["-C", destination, "for-each-ref", "--format=%(refname)"],
        "slot cloned ref enumeration",
        git_worktree=destination,
    )
    for raw_ref in refs.splitlines():
        try:
            ref = raw_ref.decode("ascii")
        except UnicodeDecodeError:
            _fail("FAIL_CAMPAIGN_SOURCE", "slot cloned ref is not ASCII", terminal=True)
        if not ref.startswith("refs/") or ".." in ref or "\x00" in ref:
            _fail("FAIL_CAMPAIGN_SOURCE", "slot cloned ref differs", terminal=True)
        _git(
            contract,
            ["-C", destination, "update-ref", "-d", ref],
            "slot cloned ref deletion",
            require_empty_stdout=True,
            git_worktree=destination,
        )
    _git(
        contract,
        ["-C", destination, "reflog", "expire", "--expire=now", "--all"],
        "slot clone reflog expiration",
        require_empty_stdout=True,
        git_worktree=destination,
    )
    _git(
        contract,
        ["-C", destination, "gc", "--prune=now", "--quiet"],
        "slot clone exact closure prune",
        require_empty_stdout=True,
        git_worktree=destination,
    )
    graph_ref = str(execution["clone_target_ref_name"])
    _git(
        contract,
        [
            "-C",
            destination,
            "update-ref",
            graph_ref,
            str(target["target_supervisor_commit"]),
        ],
        "slot clone target graph ref creation",
        require_empty_stdout=True,
        git_worktree=destination,
    )
    _git(
        contract,
        ["-C", destination, "commit-graph", "write", "--reachable"],
        "slot clone reachable commit-graph write",
        require_empty_stdout=True,
        git_worktree=destination,
    )
    return _verify_clone(contract, slot, artifact_state="ABSENT")


def _slot_registry(contract: Mapping[str, object]) -> list[dict[str, object]]:
    raw_slots = contract["slot_registry"]
    if type(raw_slots) is not list or len(raw_slots) != 3:
        _fail("FAIL_CAMPAIGN_CONTRACT", "slot registry differs")
    slots: list[dict[str, object]] = []
    root_paths: list[str] = [
        str(contract["fixed_external_roots"]["authorization_root"]),
        str(contract["fixed_external_roots"]["ledger_root"]),
    ]
    for index, raw_slot in enumerate(raw_slots, 1):
        if type(raw_slot) is not dict:
            _fail("FAIL_CAMPAIGN_CONTRACT", "slot row differs")
        slot = raw_slot
        expected_id = f"SLOT_{index:02d}"
        expected_previous = None if index == 1 else f"SLOT_{index - 1:02d}"
        expected_next = None if index == 3 else f"SLOT_{index + 1:02d}"
        if (
            slot.get("slot_id") != expected_id
            or slot.get("slot_ordinal") != index
            or slot.get("previous_slot_id") != expected_previous
            or slot.get("next_slot_id_on_exact_retry_predicate") != expected_next
        ):
            _fail("FAIL_CAMPAIGN_CONTRACT", "slot sequence differs")
        clone_root = Path(str(slot["clone_root"]))
        clone_project = Path(str(slot["clone_project_root"]))
        supervisor_artifact = Path(str(slot["supervisor_artifact_path"]))
        artifact_parent = Path(str(slot["artifact_parent_root"]))
        artifact = Path(str(slot["artifact_path"]))
        evidence = Path(str(slot["evidence_root"]))
        trace = Path(str(slot["trace_root"]))
        if (
            clone_project != clone_root / PROJECT_ROOT.name
            or supervisor_artifact
            != clone_project / "artifacts/phase3_q05b_dual_qualification_v2.json"
            or artifact.parent != artifact_parent
            or artifact.name != "phase3_q05b_dual_qualification_v2.json"
            or Path(str(slot["stdout_path"])) != evidence / "supervisor.stdout"
            or Path(str(slot["stderr_path"])) != evidence / "supervisor.stderr"
            or Path(str(slot["external_trajectory_binding_path"]))
            != evidence / "trajectory-binding.json"
            or Path(str(slot["slot_result_path"])) != evidence / "slot-result.json"
            or slot.get("artifact_parent_initial_state")
            != "ABSENT_FRESHNESS_TARGET"
            or type(slot.get("trajectory_trace_id")) is not str
            or _TRACE_ID.fullmatch(str(slot["trajectory_trace_id"])) is None
        ):
            _fail("FAIL_CAMPAIGN_CONTRACT", f"slot path relation differs: {expected_id}")
        roots = [
            os.fspath(clone_root),
            str(slot["cargo_cache_root"]),
            os.fspath(artifact_parent),
            os.fspath(evidence),
            os.fspath(trace),
        ]
        root_paths.extend(roots)
        slots.append(deepcopy(slot))
    _path_nonoverlap(root_paths)
    execution = contract["execution_policy"]
    assert type(execution) is dict
    source_roots = [
        Path(str(execution["clone_source_repository"])),
        Path(str(execution["cargo_copy_source"])),
    ]
    protected = contract["protected_state_policy"]
    assert type(protected) is dict
    forbidden_roots = [Path(str(item)) for item in protected["protected_roots_in_exact_order"]]
    for target_text in root_paths:
        target = Path(target_text)
        for source in [*source_roots, *forbidden_roots]:
            if target == source or target in source.parents or source in target.parents:
                _fail("FAIL_CAMPAIGN_CONTRACT", "campaign target overlaps source/protected root")
    return slots


def _verify_fresh_targets(
    contract: Mapping[str, object],
    slots: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, int]]:
    ledger_root = Path(str(contract["fixed_external_roots"]["ledger_root"]))
    targets = [ledger_root]
    for slot in slots:
        targets.extend(
            Path(str(slot[field]))
            for field in (
                "clone_root",
                "cargo_cache_root",
                "artifact_parent_root",
                "evidence_root",
                "trace_root",
            )
        )
    parent_rows: dict[str, dict[str, int]] = {}
    for path in targets:
        parent_text = os.fspath(path.parent)
        if parent_text not in parent_rows:
            descriptor, parent_status = _open_directory_nofollow(
                path.parent,
                f"freshness parent for {path}",
            )
            try:
                mount_id = _fd_mount_id(
                    descriptor,
                    f"freshness parent for {path}",
                )
            finally:
                os.close(descriptor)
            parent_rows[parent_text] = {
                "dev": parent_status.st_dev,
                "ino": parent_status.st_ino,
                "mount_id": mount_id,
                "mode": parent_status.st_mode,
                "uid": parent_status.st_uid,
            }
        if os.path.lexists(path):
            _fail("FAIL_CAMPAIGN_FRESHNESS", f"campaign target already exists: {path}")
    return parent_rows


def _freshness_parent_stable_identity(
    value: Mapping[str, object],
) -> dict[str, int]:
    fields = ("dev", "ino", "mount_id", "mode", "uid")
    if set(value).issuperset(fields) is False or any(
        type(value.get(field)) is not int for field in fields
    ):
        _fail(
            "FAIL_CAMPAIGN_FRESHNESS",
            "freshness parent stable identity differs",
            terminal=True,
        )
    return {field: int(value[field]) for field in fields}


def _slot_roots_absence_evidence(
    context: Mapping[str, object],
    slots: Sequence[Mapping[str, object]],
    *,
    label: str,
) -> dict[str, object]:
    """Replay fixed slot-root absence against capsule-bound state parents."""

    rows: list[dict[str, object]] = []
    error_or_null: str | None = None
    try:
        expected_parents = context["freshness_parent_identities"]
        if type(expected_parents) is not dict:
            _fail(
                "FAIL_CAMPAIGN_FRESHNESS",
                "freshness parent identity registry differs",
                terminal=True,
            )
        for slot in slots:
            for field in (
                "clone_root",
                "cargo_cache_root",
                "artifact_parent_root",
                "evidence_root",
                "trace_root",
            ):
                target = Path(str(slot[field]))
                state_parent = target.parent
                descriptor, _status = _open_directory_nofollow(
                    state_parent,
                    f"{label} state parent for {target}",
                )
                try:
                    before = _verify_held_directory_path(
                        state_parent,
                        descriptor,
                        f"{label} state parent for {target}",
                    )
                    expected = expected_parents.get(os.fspath(state_parent))
                    present_identity: dict[str, int] | None = None
                    try:
                        present = os.stat(
                            target.name,
                            dir_fd=descriptor,
                            follow_symlinks=False,
                        )
                        present_identity = {
                            "dev": present.st_dev,
                            "ino": present.st_ino,
                            "mode": present.st_mode,
                            "uid": present.st_uid,
                            "nlink": present.st_nlink,
                        }
                    except FileNotFoundError:
                        pass
                    after = _verify_held_directory_path(
                        state_parent,
                        descriptor,
                        f"{label} final state parent for {target}",
                    )
                finally:
                    os.close(descriptor)
                before_stable = _freshness_parent_stable_identity(before)
                after_stable = _freshness_parent_stable_identity(after)
                row = {
                    "slot_id": slot["slot_id"],
                    "field": field,
                    "target_path": os.fspath(target),
                    "state_parent_path": os.fspath(state_parent),
                    "expected_state_parent_stable_identity": deepcopy(expected),
                    "observed_state_parent_identity": before,
                    "observed_state_parent_stable_identity": before_stable,
                    "final_state_parent_identity": after,
                    "final_state_parent_stable_identity": after_stable,
                    "target_present_identity_or_null": present_identity,
                    "absent_and_anchored": (
                        type(expected) is dict
                        and before_stable == expected
                        and after_stable == expected
                        and present_identity is None
                    ),
                }
                rows.append(row)
                if row["absent_and_anchored"] is not True:
                    _fail(
                        "FAIL_CAMPAIGN_FRESHNESS",
                        f"{label} target is present or parent drifted: {target}",
                        terminal=True,
                    )
    except BaseException as error:
        error_or_null = (
            f"{getattr(error, 'code', type(error).__name__)}:"
            f"{getattr(error, 'detail', str(error))}"
        )[:4096]
    preimage = {
        "label": label,
        "rows": rows,
        "error_or_null": error_or_null,
        "certain": error_or_null is None,
    }
    return {
        **preimage,
        "root": _domain_digest(
            "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R8/"
            "SLOT_ROOT_ABSENCE/V2",
            preimage,
        ),
    }


def _slot_argv(
    contract: Mapping[str, object],
    slot: Mapping[str, object],
) -> list[str]:
    execution = contract["execution_policy"]
    target = contract["target_commit_a_binding"]
    assert type(execution) is dict and type(target) is dict
    prefix = execution["supervisor_argv_prefix"]
    if type(prefix) is not list or prefix != [
        "/usr/bin/python3.10",
        "-I",
        "-S",
        "-B",
        "-X",
        "utf8",
    ]:
        _fail("FAIL_CAMPAIGN_CONTRACT", "supervisor argv prefix differs")
    supervisor = Path(str(slot["clone_project_root"])) / str(
        execution["supervisor_relative_path"]
    )
    return [
        *prefix,
        os.fspath(supervisor),
        "--run",
        "--project-root",
        str(slot["clone_project_root"]),
        "--source-commit",
        str(target["target_supervisor_commit"]),
        "--artifact",
        str(slot["supervisor_artifact_path"]),
        "--cargo-cache-source",
        str(slot["cargo_cache_root"]),
        "--trajectory-root",
        str(slot["trace_root"]),
        "--trajectory-trace-id",
        str(slot["trajectory_trace_id"]),
    ]


def _build_capsule(
    contract: Mapping[str, object],
    *,
    controller_git: Mapping[str, object],
    target_git: Mapping[str, object],
    cargo_identity: Mapping[str, object],
    executable_identities: Mapping[str, object],
    python310_dependency_preflight_receipt: Mapping[str, object],
    slots: Sequence[Mapping[str, object]],
    freshness_parent_identities: Mapping[str, object],
) -> dict[str, object]:
    binding = contract["campaign_source_binding"]
    domains = contract["hash_domains"]
    assert type(binding) is dict and type(domains) is dict
    python_identity = executable_identities.get("python")
    validated_dependency_receipt = _validate_dependency_receipt_v1(
        contract,
        python310_dependency_preflight_receipt,
        expected_target_git=target_git,
        expected_controller_git=controller_git,
        expected_python_identity=python_identity,
    )
    config_raw, _ = _read_regular_nofollow(CONFIG_PATH, "campaign contract capsule input", 8 * 1024 * 1024)
    tool_raw, _ = _read_regular_nofollow(TOOL_PATH, "campaign tool capsule input", 32 * 1024 * 1024)
    capsule: dict[str, object] = {
        "schema_version": "hegel-phase3a-q05b-shadow-retry-campaign-capsule/2",
        "campaign_id": contract["contract_id"],
        "contract_digest": contract["contract_digest"],
        "campaign_source_binding_digest": binding["source_binding_digest"],
        "config_raw_sha256": sha256(config_raw).hexdigest(),
        "tool_raw_sha256": sha256(tool_raw).hexdigest(),
        "controller_git": deepcopy(dict(controller_git)),
        "target_git": deepcopy(dict(target_git)),
        "executable_identities": deepcopy(dict(executable_identities)),
        "python310_dependency_preflight_receipt": deepcopy(
            validated_dependency_receipt
        ),
        "python310_dependency_preflight_receipt_digest": (
            validated_dependency_receipt["receipt_digest"]
        ),
        "cargo_source_identity": deepcopy(dict(cargo_identity)),
        "freshness_parent_identities": deepcopy(
            dict(freshness_parent_identities)
        ),
        "authorization_root": contract["fixed_external_roots"]["authorization_root"],
        "ledger_root": contract["fixed_external_roots"]["ledger_root"],
        "maximum_supervisor_invocations": 3,
        "strictly_sequential_one_shot_slots": True,
        "ordered_slots": [deepcopy(dict(slot)) for slot in slots],
        "ordered_supervisor_argv": [
            _slot_argv(contract, slot) for slot in slots
        ],
        "ordered_runtime_environments": [
            _runtime_environment(contract, git_worktree=str(slot["clone_root"]))
            for slot in slots
        ],
        "retry_predicate_digest": contract["retry_policy"]["retry_predicate_digest"],
        "success_predicate_digest": contract["success_policy"]["success_predicate_digest"],
        "forbidden_scope_digest": contract["forbidden_scope"]["forbidden_scope_digest"],
        "resource_bounds": {
            key: contract["execution_policy"][key]
            for key in (
                "supervisor_timeout_seconds",
                "stdout_max_bytes",
                "stderr_max_bytes",
                "sigint_grace_seconds",
                "sigkill_reap_seconds",
            )
        },
        "shadow_only": True,
        "nonadopting": True,
        "network_authorized": False,
        "production_or_meta_authorized": False,
        "q1_or_active_authorized": False,
        "capsule_digest": None,
    }
    unsigned = deepcopy(capsule)
    unsigned.pop("capsule_digest")
    capsule["capsule_digest"] = _domain_digest(domains["capsule"], unsigned)
    return capsule


def _authorization_template(
    contract: Mapping[str, object],
    capsule: Mapping[str, object],
) -> dict[str, object]:
    domains = contract["hash_domains"]
    assert type(domains) is dict
    dependency_receipt = capsule.get(
        "python310_dependency_preflight_receipt"
    )
    dependency_receipt_digest = capsule.get(
        "python310_dependency_preflight_receipt_digest"
    )
    controller_git = capsule.get("controller_git")
    target_git = capsule.get("target_git")
    executable_identities = capsule.get("executable_identities")
    python_identity = (
        executable_identities.get("python")
        if type(executable_identities) is dict
        else None
    )
    if (
        type(dependency_receipt) is not dict
        or type(dependency_receipt_digest) is not str
        or _TAGGED_SHA256.fullmatch(dependency_receipt_digest) is None
    ):
        _fail(
            "FAIL_CAMPAIGN_AUTHORIZATION",
            "dependency preflight receipt is absent or digest-inconsistent",
        )
    try:
        validated_dependency_receipt = _validate_dependency_receipt_v1(
            contract,
            dependency_receipt,
            expected_target_git=target_git,
            expected_controller_git=controller_git,
            expected_python_identity=python_identity,
        )
    except ShadowRetryCampaignErrorV2 as error:
        _fail(
            "FAIL_CAMPAIGN_AUTHORIZATION",
            "dependency preflight receipt replay differs: "
            f"{error.code}:{error.detail}",
        )
    if (
        validated_dependency_receipt["receipt_digest"]
        != dependency_receipt_digest
    ):
        _fail(
            "FAIL_CAMPAIGN_AUTHORIZATION",
            "dependency preflight receipt digest binding differs",
        )
    confirmation = {
        "campaign_id": contract["contract_id"],
        "contract_digest": contract["contract_digest"],
        "target_supervisor_commit": contract["target_commit_a_binding"][
            "target_supervisor_commit"
        ],
        "target_source_binding_digest": contract["target_commit_a_binding"][
            "target_source_binding_digest"
        ],
        "campaign_source_binding_digest": contract["campaign_source_binding"][
            "source_binding_digest"
        ],
        "capsule_digest": capsule["capsule_digest"],
        "python310_dependency_preflight_receipt_digest": (
            dependency_receipt_digest
        ),
        "authorization_root": contract["fixed_external_roots"][
            "authorization_root"
        ],
        "ledger_root": contract["fixed_external_roots"]["ledger_root"],
        "three_exact_slot_path_records": deepcopy(
            list(contract["slot_registry"])
        ),
        "maximum_supervisor_invocations": 3,
        "strictly_sequential_one_shot_slots": True,
        "retry_predicate_digest": contract["retry_policy"][
            "retry_predicate_digest"
        ],
        "success_predicate_digest": contract["success_policy"][
            "success_predicate_digest"
        ],
        "terminal_no_resume_policy": deepcopy(
            dict(contract["terminal_failure_policy"])
        ),
        "forbidden_scope_digest": contract["forbidden_scope"][
            "forbidden_scope_digest"
        ],
    }
    owner_confirmation_digest = _domain_digest(
        "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R8/OWNER_CONFIRMATION/V2",
        confirmation,
    )
    value: dict[str, object] = {
        "schema_version": "hegel-phase3a-q05b-shadow-retry-campaign-owner-authorization/2",
        "status": "AUTHORIZED_EXACT_CAPSULE_ONCE",
        **confirmation,
        "owner_confirmation_digest": owner_confirmation_digest,
        "shadow_only": True,
        "nonadopting": True,
        "network_authorized": False,
        "production_or_meta_authorized": False,
        "q1_active_adoption_promotion_current_authorized": False,
        "historical_attempt2_or_attempt3_authorized": False,
        "slot_04_authorized": False,
        "authorization_digest": None,
    }
    unsigned = deepcopy(value)
    unsigned.pop("authorization_digest")
    value["authorization_digest"] = _domain_digest(domains["authorization"], unsigned)
    return value


def _preflight_context(contract: Mapping[str, object]) -> dict[str, object]:
    slots = _slot_registry(contract)
    freshness_parent_identities = _verify_fresh_targets(contract, slots)
    execution = contract["execution_policy"]
    assert type(execution) is dict
    target_git = _verify_target_git_objects(contract)
    controller_git = _controller_git_identity(contract)
    cargo_identity, cargo_rows = _scan_tree(
        Path(str(execution["cargo_copy_source"])),
        label="Cargo source preflight",
        allowed_uids=(os.geteuid(), 0),
        reject_group_or_world_writable=True,
    )
    executables = {
        "python": _executable_identity(str(execution["supervisor_interpreter"]), "Python"),
        "git": _executable_identity(str(execution["git_executable"]), "Git"),
        "docker": _executable_identity(str(execution["docker_executable"]), "Docker"),
    }
    dependency_receipt = _python310_dependency_preflight_v1(
        contract,
        target_git=target_git,
        controller_git=controller_git,
        python_identity=executables["python"],
    )
    capsule = _build_capsule(
        contract,
        controller_git=controller_git,
        target_git=target_git,
        cargo_identity=cargo_identity,
        executable_identities=executables,
        python310_dependency_preflight_receipt=dependency_receipt,
        slots=slots,
        freshness_parent_identities=freshness_parent_identities,
    )
    return {
        "slots": slots,
        "controller_git": controller_git,
        "target_git": target_git,
        "cargo_identity": cargo_identity,
        "cargo_rows": cargo_rows,
        "executable_identities": executables,
        "python310_dependency_preflight_receipt": dependency_receipt,
        "freshness_parent_identities": freshness_parent_identities,
        "capsule": capsule,
        "authorization_template": _authorization_template(contract, capsule),
    }


def preflight_phase3_q05b_shadow_retry_campaign_v2() -> dict[str, object]:
    """Read-only derivation of the exact campaign capsule and auth template."""

    contract = load_phase3_q05b_shadow_retry_campaign_contract_v2()
    context = _preflight_context(contract)
    authorization = contract["authorization_policy"]
    assert type(authorization) is dict
    auth_path = Path(str(authorization["authorization_root"])) / str(
        authorization["authorization_record_name"]
    )
    return {
        "schema_version": "hegel-phase3a-q05b-shadow-retry-campaign-preflight/2",
        "status": "CAPSULE_DERIVED_EXECUTION_REQUIRES_EXACT_OWNER_AUTHORIZATION",
        "contract_digest": contract["contract_digest"],
        "campaign_source_binding_digest": contract["campaign_source_binding"]["source_binding_digest"],
        "capsule": context["capsule"],
        "capsule_digest": context["capsule"]["capsule_digest"],
        "python310_dependency_preflight_receipt": context[
            "python310_dependency_preflight_receipt"
        ],
        "python310_dependency_preflight_receipt_digest": context[
            "python310_dependency_preflight_receipt"
        ]["receipt_digest"],
        "required_authorization_template": context["authorization_template"],
        "owner_authorization_present": os.path.lexists(auth_path),
        "execution_authorized": False,
        "ledger_and_slot_freshness_targets_absent": True,
        "supervisor_invocation_count": 0,
        "docker_daemon_contacted": False,
        "clone_cache_ledger_or_evidence_created": False,
    }


def _load_authorization(
    contract: Mapping[str, object],
    capsule: Mapping[str, object],
) -> dict[str, object]:
    policy = contract["authorization_policy"]
    assert type(policy) is dict
    root = Path(str(policy["authorization_root"]))
    path = root / str(policy["authorization_record_name"])
    root_fd, root_stat = _open_directory_nofollow(
        root,
        "owner authorization root",
    )
    try:
        root_identity = _verify_held_directory_path(
            root,
            root_fd,
            "owner authorization root",
        )
        if (
            stat.S_IMODE(root_stat.st_mode) != 0o700
            or root_stat.st_uid != os.geteuid()
        ):
            _fail(
                "FAIL_CAMPAIGN_AUTHORIZATION",
                "authorization root identity differs",
            )
        raw, _status = _read_regular_at_nofollow(
            root_fd,
            path.name,
            "owner authorization record",
            _MAX_LEDGER_BYTES,
            required_mode=0o444,
        )
        if (
            _verify_held_directory_path(
                root,
                root_fd,
                "owner authorization root final replay",
            )
            != root_identity
        ):
            _fail(
                "FAIL_CAMPAIGN_AUTHORIZATION",
                "authorization root changed during record replay",
            )
    finally:
        os.close(root_fd)
    value = _strict_json(raw, "owner authorization record", _MAX_LEDGER_BYTES)
    expected = _authorization_template(contract, capsule)
    if value != expected:
        _fail("FAIL_CAMPAIGN_AUTHORIZATION", "owner authorization differs from exact capsule")
    return value


def _docker_inventory(
    contract: Mapping[str, object],
) -> tuple[bool, list[dict[str, str]], str | None]:
    execution = contract["execution_policy"]
    target = contract["target_commit_a_binding"]
    assert type(execution) is dict and type(target) is dict
    labels = execution["target_docker_label_policy"]
    bounds = execution["readonly_command_bounds"]["docker"]
    assert type(labels) is dict and type(bounds) is dict
    commit = str(target["target_supervisor_commit"])
    argv = _docker_inventory_argv_v2(contract)
    try:
        result = _bounded_process(
            argv,
            cwd="/",
            env=_runtime_environment(contract),
            timeout_seconds=float(bounds["timeout_seconds"]),
            stdout_max_bytes=int(bounds["stdout_max_bytes"]),
            stderr_max_bytes=int(bounds["stderr_max_bytes"]),
        )
    except BaseException as error:
        diagnostic = _bounded_process_exception_diagnostic(
            "Docker read-only inventory",
            argv,
            error,
        )
        return (
            False,
            [],
            _canonical_json_bytes(diagnostic).decode("ascii").rstrip("\n"),
        )
    def diagnostic_text(reason: str) -> str:
        diagnostic = _bounded_process_diagnostic(
            "Docker read-only inventory",
            argv,
            result,
            classification={"accepted": False, "reason": reason},
        )
        return _canonical_json_bytes(diagnostic).decode("ascii").rstrip("\n")

    if (
        result["return_code"] != 0
        or result["timed_out"] is not False
        or result["stdout_overflow"] is not False
        or result["stderr_overflow"] is not False
        or result["child_reaped"] is not True
        or result["process_group_absent"] is not True
        or result["wrapper_resources_closed"] is not True
        or result["stderr"] != b""
    ):
        return False, [], diagnostic_text(
            "process exit, stream, timeout, reap, or group facts differ"
        )
    rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for line in result["stdout"].splitlines():
        try:
            container_id, namespace, slot_name, observed_commit = line.decode("ascii").split("|")
        except (UnicodeDecodeError, ValueError):
            return False, [], diagnostic_text(
                "Docker inventory row shape differs"
            )
        if (
            re.fullmatch(r"[0-9a-f]{64}", container_id) is None
            or re.fullmatch(r"[0-9a-f]{64}", namespace) is None
            or slot_name not in labels["supervisor_slot_label_values_in_exact_order"]
            or observed_commit != commit
            or container_id in seen_ids
        ):
            return False, [], diagnostic_text(
                "Docker inventory row identity differs"
            )
        seen_ids.add(container_id)
        rows.append(
            {
                "container_id": container_id,
                "execution_namespace": namespace,
                "slot": slot_name,
                "source_commit": observed_commit,
            }
        )
    return len(rows) == 0, rows, None


def _trajectory_module(
    contract: Mapping[str, object],
    slot: Mapping[str, object],
) -> Any:
    target = contract["target_commit_a_binding"]
    assert type(target) is dict
    relative = "src/hegel_machine/phase3_q05b_trajectory_v2.py"
    path = Path(str(slot["clone_project_root"])) / relative
    raw, _status = _read_regular_nofollow(path, "clone trajectory core", 8 * 1024 * 1024)
    if sha256(raw).hexdigest() != target["required_raw_sha256_by_path"][relative]:
        _fail("FAIL_CAMPAIGN_TRAJECTORY", "clone trajectory core hash differs", terminal=True)
    name = f"_hegel_q05b_trajectory_slot_{slot['slot_ordinal']}"
    module = types.ModuleType(name)
    module.__file__ = os.fspath(path)
    sys.modules[name] = module
    try:
        exec(compile(raw, os.fspath(path), "exec", dont_inherit=True), module.__dict__)
    except BaseException as error:
        sys.modules.pop(name, None)
        _fail(
            "FAIL_CAMPAIGN_TRAJECTORY",
            f"trajectory verifier load failed: {type(error).__name__}:{error}",
            terminal=True,
        )
    return module


def _row_value(value: object) -> object:
    if is_dataclass(value):
        return asdict(value)
    if type(value) is tuple:
        return [_row_value(item) for item in value]
    if type(value) is list:
        return [_row_value(item) for item in value]
    if type(value) is dict:
        return {str(key): _row_value(item) for key, item in value.items()}
    return value


def _minimal_trajectory_event_row(
    record: Mapping[str, object],
) -> dict[str, object]:
    if any(field not in record for field in _TRAJECTORY_MINIMAL_EVENT_FIELDS):
        _fail(
            "FAIL_CAMPAIGN_TRAJECTORY",
            "verified trajectory event minimal projection differs",
            terminal=True,
        )
    return {
        field: deepcopy(record[field])
        for field in _TRAJECTORY_MINIMAL_EVENT_FIELDS
    }


def _failure_projection_from_event(
    record: Mapping[str, object],
) -> dict[str, object]:
    value = {
        "sequence": record.get("sequence"),
        "timestamp_ns": record.get("timestamp_ns"),
        "span_id": record.get("span_id"),
        "operation": record.get("operation"),
        "stage_or_null": record.get("stage_or_null"),
        "substep_or_null": record.get("substep_or_null"),
        "actor_or_null": record.get("actor_or_null"),
        "detail_or_null": record.get("detail_or_null"),
        "error_code": record.get("error_code_or_null"),
        "error_detail": record.get("error_detail_or_null"),
        "role_id_or_null": record.get("role_id_or_null"),
        "sample_ordinal_or_null": record.get("sample_ordinal_or_null"),
        "previous_finish_ns_or_null": record.get("previous_finish_ns_or_null"),
        "scheduled_start_ns_or_null": record.get("scheduled_start_ns_or_null"),
        "actual_start_ns_or_null": record.get("actual_start_ns_or_null"),
        "finish_ns_or_null": record.get("finish_ns_or_null"),
        "collection_duration_ns_or_null": record.get(
            "collection_duration_ns_or_null"
        ),
        "scheduler_lateness_ns_or_null": record.get(
            "scheduler_lateness_ns_or_null"
        ),
        "target_wait_ns_or_null": record.get("target_wait_ns_or_null"),
        "gap_ns_or_null": record.get("gap_ns_or_null"),
        "maximum_gap_ns_or_null": record.get("maximum_gap_ns_or_null"),
        "gap_exceeded_or_null": record.get("gap_exceeded_or_null"),
        "record_digest": record.get("record_digest"),
    }
    if set(value) != set(_TRAJECTORY_FAILURE_ROW_FIELDS):
        _fail(
            "FAIL_CAMPAIGN_TRAJECTORY",
            "verified trajectory failure projection registry differs",
            terminal=True,
        )
    return value


def _ordered_event_root(
    domain: str,
    rows: Sequence[tuple[int, str]],
) -> str:
    """Return a constant-memory ordered root over sequence/digest pairs."""

    if type(domain) is not str or not domain or "\x00" in domain:
        _fail("FAIL_CAMPAIGN_CONTRACT", "ordered-root hash domain differs")
    try:
        domain_bytes = domain.encode("ascii")
    except UnicodeEncodeError:
        _fail("FAIL_CAMPAIGN_CONTRACT", "ordered-root hash domain is not ASCII")
    root = sha256(domain_bytes + b"\x00EMPTY").digest()
    previous_sequence = 0
    for sequence, tagged_digest in rows:
        if (
            type(sequence) is not int
            or sequence <= previous_sequence
            or sequence >= 1 << 64
            or type(tagged_digest) is not str
            or _TAGGED_SHA256.fullmatch(tagged_digest) is None
        ):
            _fail(
                "FAIL_CAMPAIGN_TRAJECTORY",
                "ordered-root event identity differs",
                terminal=True,
            )
        root = sha256(
            domain_bytes
            + b"\x00ROW\x00"
            + root
            + sequence.to_bytes(8, "big")
            + bytes.fromhex(tagged_digest.removeprefix("sha256:"))
        ).digest()
        previous_sequence = sequence
    return "sha256:" + root.hex()


def _trajectory_partition_evidence(
    contract: Mapping[str, object],
    payloads: Sequence[bytes],
    *,
    trajectory_status: str,
    failure_rows_in_sequence_order: Sequence[object],
) -> dict[str, object]:
    """Bind a verified trace without copying its unbounded event projection.

    The raw immutable event files and their global payload digest remain the
    authoritative trace.  For a complete failure, this helper additionally
    emits a bounded decision certificate that explicitly retains every FAIL
    row and every registered-cleanup BEGIN/OK row within retry limits.  All
    other nonfailure events are order-rooted and counted, never hidden or used
    as an independent retry disqualifier.
    """

    domains = contract.get("hash_domains")
    if type(domains) is not dict:
        _fail("FAIL_CAMPAIGN_CONTRACT", "campaign hash-domain registry differs")
    required_domains = (
        "trajectory_event_payloads",
        "trajectory_suffix",
        "trajectory_other_nonfailure",
        "trajectory_failure_rows",
        "trajectory_cleanup_rows",
        "trajectory_partition_certificate",
    )
    if any(type(domains.get(key)) is not str for key in required_domains):
        _fail("FAIL_CAMPAIGN_CONTRACT", "trajectory certificate domain missing")

    verified_failures = _row_value(list(failure_rows_in_sequence_order))
    if type(verified_failures) is not list or any(
        type(row) is not dict or set(row) != set(_TRAJECTORY_FAILURE_ROW_FIELDS)
        for row in verified_failures
    ):
        _fail(
            "FAIL_CAMPAIGN_TRAJECTORY",
            "verified trajectory failure-row surface differs",
            terminal=True,
        )

    decoded_events: list[dict[str, object]] = []
    projected_failures: list[dict[str, object]] = []
    cleanup_rows_all: list[dict[str, object]] = []
    last_event: dict[str, object] | None = None
    for index, payload in enumerate(payloads, 1):
        record = _strict_json(
            payload,
            f"trajectory event payload {index}",
            _MAX_LEDGER_BYTES,
        )
        if type(record) is not dict or record.get("sequence") != index:
            _fail(
                "FAIL_CAMPAIGN_TRAJECTORY",
                f"verified trajectory event identity differs at sequence {index}",
                terminal=True,
            )
        minimal = _minimal_trajectory_event_row(record)
        decoded_events.append(minimal)
        last_event = minimal
        if record.get("event_type") == "FAIL":
            projected_failures.append(_failure_projection_from_event(record))
        elif (
            record.get("event_type") in ("BEGIN", "OK")
            and record.get("operation") == "HELD_ACTOR_ABORT_CLEANUP"
        ):
            cleanup_rows_all.append(minimal)

    if projected_failures != verified_failures:
        _fail(
            "FAIL_CAMPAIGN_TRAJECTORY",
            "raw FAIL projection differs from verifier failure rows",
            terminal=True,
        )

    payload_digest = _domain_digest(
        str(domains["trajectory_event_payloads"]),
        [sha256(payload).hexdigest() for payload in payloads],
    )
    retained_failures = projected_failures[:_MAX_RETAINED_FAILURE_ROWS]
    retained_cleanups = cleanup_rows_all[:_MAX_RETAINED_CLEANUP_ROWS]
    result: dict[str, object] = {
        "canonical_event_payloads_digest": payload_digest,
        "failure_rows_retained_in_sequence_order": retained_failures,
        "cleanup_rows_retained_in_sequence_order": retained_cleanups,
        "partition_certificate_or_null": None,
        "last_durable_event_row_or_null": deepcopy(last_event),
    }
    if trajectory_status != "COMPLETE_FAIL" or not projected_failures:
        return result

    origin = projected_failures[0]
    origin_sequence = int(origin["sequence"])
    cleanup_rows_all = []
    suffix_pairs: list[tuple[int, str]] = []
    other_pairs: list[tuple[int, str]] = []
    failure_count = 0
    cleanup_count = 0
    other_count = 0
    unpartitioned_count = 0
    for event in decoded_events[origin_sequence - 1 :]:
        sequence = int(event["sequence"])
        record_digest = str(event["record_digest"])
        suffix_pairs.append((sequence, record_digest))
        event_type = event["event_type"]
        if event_type == "FAIL":
            failure_count += 1
        elif (
            event_type in ("BEGIN", "OK")
            and event["operation"] == "HELD_ACTOR_ABORT_CLEANUP"
        ):
            cleanup_count += 1
            cleanup_rows_all.append(event)
        elif event_type in ("BEGIN", "OK"):
            other_count += 1
            other_pairs.append((sequence, record_digest))
        else:
            unpartitioned_count += 1

    partitioned_count = failure_count + cleanup_count + other_count
    retained_cleanups = cleanup_rows_all[:_MAX_RETAINED_CLEANUP_ROWS]
    result["cleanup_rows_retained_in_sequence_order"] = retained_cleanups
    certificate: dict[str, object] = {
        "schema_version": _PARTITION_CERTIFICATE_SCHEMA_V2,
        "trace_id": decoded_events[0].get("trace_id", None),
        "producer_id": decoded_events[0].get("producer_id", None),
        "trajectory_status": "COMPLETE_FAIL",
        "event_count": len(decoded_events),
        "first_event_record_digest": decoded_events[0]["record_digest"],
        "final_event_record_digest": decoded_events[-1]["record_digest"],
        "canonical_event_payloads_digest": payload_digest,
        "origin_sequence": origin_sequence,
        "origin_record_digest": origin["record_digest"],
        "origin_actor_or_null": origin["actor_or_null"],
        "suffix_event_count": len(suffix_pairs),
        "suffix_event_root": _ordered_event_root(
            str(domains["trajectory_suffix"]), suffix_pairs
        ),
        "failure_event_count": failure_count,
        "failure_rows_digest": _domain_digest(
            str(domains["trajectory_failure_rows"]), projected_failures
        ),
        "failure_rows_retained_count": len(retained_failures),
        "failure_rows_truncated": len(retained_failures) < failure_count,
        "cleanup_nonfailure_event_count": cleanup_count,
        "cleanup_rows_digest": _domain_digest(
            str(domains["trajectory_cleanup_rows"]), cleanup_rows_all
        ),
        "cleanup_rows_retained_count": len(retained_cleanups),
        "cleanup_rows_truncated": len(retained_cleanups) < cleanup_count,
        "other_nonfailure_event_count": other_count,
        "other_nonfailure_event_root": _ordered_event_root(
            str(domains["trajectory_other_nonfailure"]), other_pairs
        ),
        "partitioned_event_count": partitioned_count,
        "unpartitioned_event_count": unpartitioned_count,
        "certificate_digest": None,
    }
    # The minimal projection intentionally omits globally repeated trace and
    # producer identifiers; take those exact values from the verified payload.
    first_payload = _strict_json(
        payloads[0], "trajectory first event payload", _MAX_LEDGER_BYTES
    )
    assert type(first_payload) is dict
    certificate["trace_id"] = first_payload.get("trace_id")
    certificate["producer_id"] = first_payload.get("producer_id")
    unsigned = deepcopy(certificate)
    unsigned.pop("certificate_digest")
    certificate["certificate_digest"] = _domain_digest(
        str(domains["trajectory_partition_certificate"]), unsigned
    )
    if len(_canonical_json_bytes(certificate)) > _MAX_PARTITION_CERTIFICATE_BYTES:
        _fail(
            "FAIL_CAMPAIGN_TRAJECTORY",
            "partition certificate exceeds fixed byte bound",
            terminal=True,
        )
    result["partition_certificate_or_null"] = certificate
    return result


def _success_trace_bindings(
    payloads: Sequence[bytes],
) -> dict[str, object] | None:
    stage_roots: dict[int, str] = {}
    adapter_sha: str | None = None
    prepublication_root: str | None = None
    publication_sha: str | None = None
    for index, payload in enumerate(payloads, 1):
        try:
            record = _strict_json(
                payload,
                f"trajectory event payload {index}",
                _MAX_LEDGER_BYTES,
            )
        except ShadowRetryCampaignErrorV2:
            return None
        if type(record) is not dict or record.get("event_type") != "OK":
            continue
        operation = record.get("operation")
        detail = record.get("detail_or_null")
        if type(detail) is not str:
            continue
        if operation == "SUPERVISOR_STAGE":
            stage = record.get("stage_or_null")
            match = re.fullmatch(r"STAGE_(\d{2})", str(stage))
            root_match = re.fullmatch(r"stage_evidence_root=([0-9a-f]{64})", detail)
            if match is None or root_match is None:
                return None
            stage_id = int(match.group(1))
            if stage_id in stage_roots:
                return None
            stage_roots[stage_id] = root_match.group(1)
        elif operation == "STAGE9_ARTIFACT_ASSEMBLY_REPLAY":
            match = re.fullmatch(r"artifact_sha256=([0-9a-f]{64})", detail)
            if match is None or adapter_sha is not None:
                return None
            adapter_sha = match.group(1)
        elif operation == "STAGE10_PREPUBLICATION_REPLAY":
            match = re.fullmatch(
                r"prepublication_live_marker_replay_root=([0-9a-f]{64})",
                detail,
            )
            if match is None or prepublication_root is not None:
                return None
            prepublication_root = match.group(1)
        elif operation == "ARTIFACT_PUBLICATION":
            match = re.fullmatch(r"artifact_sha256=([0-9a-f]{64})", detail)
            if match is None or publication_sha is not None:
                return None
            publication_sha = match.group(1)
    if (
        set(stage_roots) != set(range(1, 11))
        or adapter_sha is None
        or prepublication_root is None
        or publication_sha is None
    ):
        return None
    return {
        "stage_roots": [stage_roots[index] for index in range(1, 11)],
        "adapter_artifact_sha256": adapter_sha,
        "prepublication_live_marker_replay_root": prepublication_root,
        "publication_artifact_sha256": publication_sha,
    }


def _trajectory_evidence(
    contract: Mapping[str, object],
    slot: Mapping[str, object],
) -> dict[str, object]:
    module = _trajectory_module(contract, slot)
    trace_root = Path(str(slot["trace_root"]))
    verification_error: str | None = None
    verified: object | None = None
    complete_verification = False
    try:
        verified = module.verify_durable_trajectory_v2(
            trace_root,
            expected_parent=trace_root.parent,
            allow_incomplete=False,
        )
        complete_verification = True
    except BaseException as error:
        verification_error = (
            f"{getattr(error, 'code', type(error).__name__)}:"
            f"{getattr(error, 'detail', str(error))}"
        )[:4096]
        try:
            verified = module.verify_durable_trajectory_v2(
                trace_root,
                expected_parent=trace_root.parent,
                allow_incomplete=True,
            )
        except BaseException as prefix_error:
            verification_error += (
                "; prefix="
                f"{getattr(prefix_error, 'code', type(prefix_error).__name__)}:"
                f"{getattr(prefix_error, 'detail', str(prefix_error))}"
            )[:4096]
            verified = None
    manifest_path = trace_root / "terminal-manifest-v2.json"
    manifest_raw_sha: str | None = None
    manifest_value: dict[str, object] | None = None
    if os.path.lexists(manifest_path):
        try:
            manifest_raw, _status = _read_regular_nofollow(
                manifest_path,
                "trajectory terminal manifest",
                _MAX_LEDGER_BYTES,
                required_mode=0o444,
            )
            decoded = _strict_json(
                manifest_raw,
                "trajectory terminal manifest",
                _MAX_LEDGER_BYTES,
            )
            if type(decoded) is dict:
                manifest_value = decoded
                manifest_raw_sha = sha256(manifest_raw).hexdigest()
        except ShadowRetryCampaignErrorV2 as error:
            verification_error = (verification_error or "") + "; manifest=" + error.detail
    if verified is None:
        return {
            "verification_status": "INVALID",
            "complete": False,
            "complete_verification_passed": False,
            "verification_error_or_null": verification_error or "trajectory unavailable",
            "trace_id_or_null": None,
            "producer_id_or_null": None,
            "event_count": 0,
            "first_event_record_digest_or_null": None,
            "final_event_record_digest_or_null": None,
            "terminal_outcome_or_null": None,
            "terminal_manifest_digest_or_null": None,
            "terminal_manifest_raw_sha256_or_null": manifest_raw_sha,
            "clock_id_or_null": (
                manifest_value.get("clock_id")
                if manifest_value is not None
                else None
            ),
            "closed_span_count_or_null": (
                manifest_value.get("closed_span_count")
                if manifest_value is not None
                else None
            ),
            "terminal_event_type_or_null": None,
            "terminal_error_code_or_null": None,
            "terminal_error_detail_or_null": None,
            "torn_tail_staging_basename_or_null": None,
            "open_span_ids_in_begin_order": [],
            "open_span_contexts_in_begin_order": [],
            "failure_rows_retained_in_sequence_order": [],
            "cleanup_rows_retained_in_sequence_order": [],
            "partition_certificate_or_null": None,
            "last_durable_event_row_or_null": None,
            "canonical_event_payloads_digest": None,
            "success_trace_bindings_or_null": None,
        }
    payloads = list(verified.canonical_event_payloads_in_order)
    partition = _trajectory_partition_evidence(
        contract,
        payloads,
        trajectory_status=verified.status,
        failure_rows_in_sequence_order=verified.failure_rows_in_sequence_order,
    )
    return {
        "verification_status": verified.status,
        "complete": verified.complete,
        "complete_verification_passed": complete_verification,
        "verification_error_or_null": verification_error,
        "trace_id_or_null": verified.trace_id_or_null,
        "producer_id_or_null": verified.producer_id_or_null,
        "event_count": verified.event_count,
        "first_event_record_digest_or_null": verified.first_event_record_digest_or_null,
        "final_event_record_digest_or_null": verified.final_event_record_digest_or_null,
        "terminal_outcome_or_null": verified.terminal_outcome_or_null,
        "terminal_manifest_digest_or_null": verified.terminal_manifest_digest_or_null,
        "terminal_manifest_raw_sha256_or_null": manifest_raw_sha,
        "clock_id_or_null": (
            manifest_value.get("clock_id")
            if manifest_value is not None
            else None
        ),
        "closed_span_count_or_null": (
            manifest_value.get("closed_span_count")
            if manifest_value is not None
            else None
        ),
        "terminal_event_type_or_null": (
            manifest_value.get("terminal_event_type")
            if manifest_value is not None
            else None
        ),
        "terminal_error_code_or_null": verified.terminal_error_code_or_null,
        "terminal_error_detail_or_null": verified.terminal_error_detail_or_null,
        "torn_tail_staging_basename_or_null": verified.torn_tail_staging_basename_or_null,
        "open_span_ids_in_begin_order": list(verified.open_span_ids_in_begin_order),
        "open_span_contexts_in_begin_order": _row_value(
            verified.open_span_contexts_in_begin_order
        ),
        "failure_rows_retained_in_sequence_order": deepcopy(
            partition["failure_rows_retained_in_sequence_order"]
        ),
        "cleanup_rows_retained_in_sequence_order": deepcopy(
            partition["cleanup_rows_retained_in_sequence_order"]
        ),
        "partition_certificate_or_null": deepcopy(
            partition["partition_certificate_or_null"]
        ),
        "last_durable_event_row_or_null": deepcopy(
            partition["last_durable_event_row_or_null"]
        ),
        "canonical_event_payloads_digest": partition[
            "canonical_event_payloads_digest"
        ],
        "success_trace_bindings_or_null": (
            _success_trace_bindings(payloads)
            if verified.status == "COMPLETE_OK"
            else None
        ),
    }


def _artifact_replay(
    contract: Mapping[str, object],
    slot: Mapping[str, object],
) -> dict[str, object]:
    artifact_path = Path(str(slot["supervisor_artifact_path"]))
    parent_fd = -1
    parent_identity: dict[str, object] | None = None
    try:
        parent_fd, _parent_status = _open_directory_nofollow(
            artifact_path.parent,
            "supervisor artifact parent",
        )
        parent_identity = _verify_held_directory_path(
            artifact_path.parent,
            parent_fd,
            "supervisor artifact parent",
        )
        try:
            os.stat(
                artifact_path.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return {
                "state": "ABSENT",
                "raw_sha256_or_null": None,
                "size_or_null": None,
                "replay_passed": False,
                "summary_or_null": None,
                "error_or_null": None,
                "parent_identity_or_null": parent_identity,
                "file_identity_or_null": None,
            }
        raw, status = _read_regular_at_nofollow(
            parent_fd,
            artifact_path.name,
            "supervisor artifact",
            64 * 1024 * 1024,
            required_mode=0o444,
        )
    except ShadowRetryCampaignErrorV2 as error:
        return {
            "state": "RESIDUAL_INVALID",
            "raw_sha256_or_null": None,
            "size_or_null": None,
            "replay_passed": False,
            "summary_or_null": None,
            "error_or_null": error.detail,
            "parent_identity_or_null": parent_identity,
            "file_identity_or_null": None,
        }
    except BaseException as error:
        return {
            "state": "RESIDUAL_INVALID",
            "raw_sha256_or_null": None,
            "size_or_null": None,
            "replay_passed": False,
            "summary_or_null": None,
            "error_or_null": f"{type(error).__name__}:{error}"[:4096],
            "parent_identity_or_null": parent_identity,
            "file_identity_or_null": None,
        }
    finally:
        if parent_fd >= 0:
            os.close(parent_fd)
    file_identity = {
        "dev": status.st_dev,
        "ino": status.st_ino,
        "mode": stat.S_IMODE(status.st_mode),
        "nlink": status.st_nlink,
        "uid": status.st_uid,
        "size": status.st_size,
    }
    project = Path(str(slot["clone_project_root"]))
    source_root = project / "src"
    module_prefix = "hegel_machine"
    existing = {name for name in sys.modules if name == module_prefix or name.startswith(module_prefix + ".")}
    if existing:
        return {
            "state": "RESIDUAL_INVALID",
            "raw_sha256_or_null": sha256(raw).hexdigest(),
            "size_or_null": len(raw),
            "replay_passed": False,
            "summary_or_null": None,
            "error_or_null": "hegel_machine import namespace was not initially empty",
            "parent_identity_or_null": parent_identity,
            "file_identity_or_null": file_identity,
        }
    sys.path.insert(0, os.fspath(source_root))
    try:
        module = importlib.import_module(
            "hegel_machine.phase3_q05b_actual_artifact_v2"
        )
        expected_module_path = project / "src/hegel_machine/phase3_q05b_actual_artifact_v2.py"
        if Path(str(module.__file__)) != expected_module_path:
            raise RuntimeError("artifact verifier module path differs")
        replayed = module.decode_and_replay_actual_artifact_v1(raw)
        summary = module.actual_artifact_summary_v1(replayed)
        derived = replayed["derived"]
        actual_admission = replayed["sections"]["actual_admission"]
        summary = {
            **summary,
            "qualification_count": derived["qualification_count"],
            "qualification_mask": derived["qualification_mask"],
            "closed_q1_authority": deepcopy(
                dict(derived["closed_q1_authority"])
            ),
            "actual_admission_evidence_root": actual_admission[
                "actual_admission_evidence_root"
            ],
            "actual_admission_source_commit": actual_admission[
                "source_commit"
            ],
            "actual_admission_artifact_path": actual_admission[
                "artifact_path"
            ],
            "pre_artifact_live_marker_replay_root": actual_admission[
                "pre_artifact_live_marker_replay"
            ]["live_marker_replay_root"],
        }
    except BaseException as error:
        return {
            "state": "RESIDUAL_INVALID",
            "raw_sha256_or_null": sha256(raw).hexdigest(),
            "size_or_null": len(raw),
            "replay_passed": False,
            "summary_or_null": None,
            "error_or_null": f"{type(error).__name__}:{error}"[:4096],
            "parent_identity_or_null": parent_identity,
            "file_identity_or_null": file_identity,
        }
    finally:
        try:
            sys.path.remove(os.fspath(source_root))
        except ValueError:
            pass
        for name in list(sys.modules):
            if name == module_prefix or name.startswith(module_prefix + "."):
                sys.modules.pop(name, None)
    return {
        "state": "VALID",
        "raw_sha256_or_null": sha256(raw).hexdigest(),
        "size_or_null": len(raw),
        "replay_passed": True,
        "summary_or_null": summary,
        "error_or_null": None,
        "parent_identity_or_null": parent_identity,
        "file_identity_or_null": file_identity,
    }


def _publish_external_artifact(
    slot: Mapping[str, object],
    artifact_evidence: Mapping[str, object],
    *,
    commit_callback: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    if artifact_evidence.get("state") != "VALID" or artifact_evidence.get("replay_passed") is not True:
        _fail("FAIL_CAMPAIGN_ARTIFACT", "invalid supervisor artifact cannot be exported", terminal=True)
    source = Path(str(slot["supervisor_artifact_path"]))
    target_parent = Path(str(slot["artifact_parent_root"]))
    target = Path(str(slot["artifact_path"]))
    raw, source_status = _read_regular_nofollow(
        source,
        "supervisor artifact export source",
        64 * 1024 * 1024,
        required_mode=0o444,
    )
    expected_source_identity = artifact_evidence.get(
        "file_identity_or_null"
    )
    observed_source_identity = {
        "dev": source_status.st_dev,
        "ino": source_status.st_ino,
        "mode": stat.S_IMODE(source_status.st_mode),
        "nlink": source_status.st_nlink,
        "uid": source_status.st_uid,
        "size": source_status.st_size,
    }
    if (
        sha256(raw).hexdigest() != artifact_evidence["raw_sha256_or_null"]
        or observed_source_identity != expected_source_identity
    ):
        _fail("FAIL_CAMPAIGN_ARTIFACT", "supervisor artifact changed before export", terminal=True)
    parent_fd, root_fd = _create_private_root(
        target_parent,
        "external artifact parent",
    )
    os.close(parent_fd)
    publication_attempted = False
    try:
        root_identity = _verify_held_directory_path(
            target_parent,
            root_fd,
            "external artifact parent",
        )
        publication_attempted = True
        _publish_immutable_at(
            root_fd,
            target.name,
            raw,
            publication_label="external-artifact",
            maximum_bytes=64 * 1024 * 1024,
            known_failure_code="FAIL_CAMPAIGN_ARTIFACT",
        )
        replay, status = _read_regular_at_nofollow(
            root_fd,
            target.name,
            "external artifact",
            64 * 1024 * 1024,
            required_mode=0o444,
        )
        _postpublication_directory_identity(
            target_parent,
            root_fd,
            "external artifact parent final replay",
            root_identity,
        )
        if replay != raw:
            _raise_postpublication_uncertain(
                "external artifact anchored byte replay"
            )
        result = {
            "state": "VALID",
            "raw_sha256_or_null": sha256(replay).hexdigest(),
            "size_or_null": len(replay),
            "replay_passed": True,
            "error_or_null": None,
            "parent_identity_or_null": root_identity,
            "file_identity_or_null": {
                "dev": status.st_dev,
                "ino": status.st_ino,
                "mode": stat.S_IMODE(status.st_mode),
                "nlink": status.st_nlink,
                "uid": status.st_uid,
                "size": status.st_size,
            },
        }
        if commit_callback is not None:
            commit_callback(deepcopy(result))
    except BaseException as error:
        if publication_attempted:
            _guard_publication_attempt_failure(
                "external artifact publication continuation",
                target.name,
                "FAIL_CAMPAIGN_ARTIFACT",
                error,
            )
        raise
    finally:
        active_error = sys.exc_info()[1]
        try:
            os.close(root_fd)
        except BaseException as error:
            if (
                isinstance(active_error, ShadowRetryCampaignErrorV2)
                and active_error.code
                in {
                    "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN",
                    "FAIL_CAMPAIGN_SPAWN_UNCERTAIN",
                }
            ):
                raise active_error
            if publication_attempted:
                _raise_postpublication_uncertain(
                    "external artifact parent descriptor close",
                    error,
                    publication_basename_or_null=target.name,
                )
            raise
    return result


def _absent_external_artifact(
    context: Mapping[str, object],
    slot: Mapping[str, object],
) -> dict[str, object]:
    parent = Path(str(slot["artifact_parent_root"]))
    state_parent = parent.parent
    expected = context["freshness_parent_identities"].get(
        os.fspath(state_parent)
    )
    descriptor, _status = _open_directory_nofollow(
        state_parent,
        "external artifact state parent",
    )
    try:
        observed = _verify_held_directory_path(
            state_parent,
            descriptor,
            "external artifact state parent",
        )
        observed_stable = _freshness_parent_stable_identity(observed)
        absent = False
        try:
            os.stat(parent.name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            absent = True
        if observed_stable != expected:
            absent = False
        final_observed = _verify_held_directory_path(
            state_parent,
            descriptor,
            "external artifact state parent final replay",
        )
        final_stable = _freshness_parent_stable_identity(final_observed)
        if final_stable != expected or final_stable != observed_stable:
            absent = False
    finally:
        os.close(descriptor)
    return {
        "state": "ABSENT" if absent else "RESIDUAL_INVALID",
        "raw_sha256_or_null": None,
        "size_or_null": None,
        "replay_passed": False,
        "error_or_null": (
            None
            if absent
            else "external artifact parent absence or state-parent identity differs"
        ),
        "parent_identity_or_null": observed,
        "file_identity_or_null": None,
    }


def _replay_external_artifact(
    slot: Mapping[str, object],
) -> dict[str, object]:
    parent = Path(str(slot["artifact_parent_root"]))
    target = Path(str(slot["artifact_path"]))
    descriptor, _status = _open_directory_nofollow(
        parent,
        "external artifact final parent replay",
    )
    try:
        parent_identity = _verify_held_directory_path(
            parent,
            descriptor,
            "external artifact final parent replay",
        )
        raw, status = _read_regular_at_nofollow(
            descriptor,
            target.name,
            "external artifact final replay",
            64 * 1024 * 1024,
            required_mode=0o444,
        )
        if (
            _verify_held_directory_path(
                parent,
                descriptor,
                "external artifact final parent identity replay",
            )
            != parent_identity
        ):
            _fail(
                "FAIL_CAMPAIGN_ARTIFACT",
                "external artifact parent changed during final replay",
                terminal=True,
            )
    finally:
        os.close(descriptor)
    return {
        "state": "VALID",
        "raw_sha256_or_null": sha256(raw).hexdigest(),
        "size_or_null": len(raw),
        "replay_passed": True,
        "error_or_null": None,
        "parent_identity_or_null": parent_identity,
        "file_identity_or_null": {
            "dev": status.st_dev,
            "ino": status.st_ino,
            "mode": stat.S_IMODE(status.st_mode),
            "nlink": status.st_nlink,
            "uid": status.st_uid,
            "size": status.st_size,
        },
    }


def _ledger_record_type_for_basename(basename: str) -> str:
    if basename == "campaign.intent.json":
        return "CAMPAIGN_INTENT"
    if basename == "campaign.authorization-consumed.json":
        return "CAMPAIGN_AUTHORIZATION_CONSUMED"
    if basename == "campaign.terminal.json":
        return "CAMPAIGN_TERMINAL"
    if basename == "campaign.completion.json":
        return "CAMPAIGN_COMPLETION"
    if basename.endswith(".intent.json"):
        return "SLOT_INTENT"
    if basename.endswith(".consumed.json"):
        return "SLOT_CONSUMED"
    if basename.endswith(".trajectory-binding.json"):
        return "EXTERNAL_TRAJECTORY_BINDING"
    if basename.endswith(".outcome.json"):
        return "SLOT_OUTCOME"
    _fail(
        "FAIL_CAMPAIGN_LEDGER",
        "ledger filename grammar differs",
        terminal=True,
    )


class _LedgerV2:
    def __init__(
        self,
        contract: Mapping[str, object],
        capsule: Mapping[str, object],
        authorization: Mapping[str, object],
    ) -> None:
        self.contract = contract
        self.capsule = capsule
        self.authorization = authorization
        self.completion_commit_verified = False
        self.completion_result_candidate_or_null: dict[str, object] | None = None
        self.authoritative_completion_result_or_null: dict[str, object] | None = None
        self.published_records: list[dict[str, object]] = []
        root = Path(str(contract["fixed_external_roots"]["ledger_root"]))
        self.root_path = root
        self.parent_fd, self.root_fd = _create_private_root(root, "campaign ledger")
        self.root_identity = _verify_held_directory_path(
            self.root_path,
            self.root_fd,
            "campaign ledger",
        )
        self.locked = False
        try:
            fcntl.flock(self.root_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.locked = True
        except BaseException:
            self.close()
            raise
        self.previous_record_digest: str | None = None
        self.sequence = 0
        self.published_names: list[str] = []
        names = contract["ledger_policy"]["record_names_in_exact_order"]
        if names != list(_LEDGER_RECORD_NAMES_V2):
            self.close()
            _fail("FAIL_CAMPAIGN_LEDGER", "ledger filename registry differs", terminal=True)
        self.allowed_names = list(names)
        self.last_name_index = -1

    def _validate_publish_transition(
        self,
        basename: str,
        record_type: str,
        status_text: str,
    ) -> int:
        if basename not in self.allowed_names:
            _fail(
                "FAIL_CAMPAIGN_LEDGER",
                "ledger filename is not registered",
                terminal=True,
            )
        name_index = self.allowed_names.index(basename)
        if basename in self.published_names or name_index <= self.last_name_index:
            _fail(
                "FAIL_CAMPAIGN_LEDGER",
                "ledger filename order differs",
                terminal=True,
            )
        if basename == "campaign.completion.json":
            if (
                not self.published_names
                or self.published_names[-1] != "campaign.terminal.json"
                or name_index != self.last_name_index + 1
            ):
                _fail(
                    "FAIL_CAMPAIGN_LEDGER",
                    "campaign completion does not immediately follow terminal",
                    terminal=True,
                )
        elif (
            basename != "campaign.terminal.json"
            and name_index != self.last_name_index + 1
        ):
            _fail(
                "FAIL_CAMPAIGN_LEDGER",
                "ledger state transition differs",
                terminal=True,
            )
        expected_type = _ledger_record_type_for_basename(basename)
        if record_type != expected_type or type(status_text) is not str or not status_text:
            _fail(
                "FAIL_CAMPAIGN_LEDGER",
                "ledger type or status differs",
                terminal=True,
            )
        return name_index

    def prepare(
        self,
        basename: str,
        record_type: str,
        status_text: str,
        body: Mapping[str, object],
    ) -> dict[str, object]:
        self._validate_publish_transition(basename, record_type, status_text)
        if type(body) is not dict:
            _fail("FAIL_CAMPAIGN_LEDGER", "ledger body differs", terminal=True)
        next_sequence = self.sequence + 1
        value: dict[str, object] = {
            "schema_version": "hegel-phase3a-q05b-shadow-retry-campaign-ledger-record/2",
            "campaign_id": self.contract["contract_id"],
            "contract_digest": self.contract["contract_digest"],
            "campaign_source_binding_digest": self.contract["campaign_source_binding"]["source_binding_digest"],
            "capsule_digest": self.capsule["capsule_digest"],
            "authorization_digest": self.authorization["authorization_digest"],
            "sequence": next_sequence,
            "record_type": record_type,
            "status": status_text,
            "previous_record_digest_or_null": self.previous_record_digest,
            "body": deepcopy(dict(body)),
            "record_digest": None,
        }
        domains = self.contract["hash_domains"]
        unsigned = deepcopy(value)
        unsigned.pop("record_digest")
        value["record_digest"] = _domain_digest(domains["ledger_record"], unsigned)
        return value

    def prepare_bytes(
        self,
        basename: str,
        record_type: str,
        status_text: str,
        body: Mapping[str, object],
    ) -> bytes:
        return _canonical_json_bytes(
            self.prepare(basename, record_type, status_text, body)
        )

    def cache_completion_result_candidate(
        self,
        result: Mapping[str, object],
        completion_ledger_record: Mapping[str, object],
    ) -> dict[str, object]:
        frozen_bytes = _canonical_json_bytes(dict(result))
        frozen = _strict_json(
            frozen_bytes,
            "prepared campaign completion result",
            _MAX_LEDGER_BYTES,
        )
        if type(frozen) is not dict:
            _fail(
                "FAIL_CAMPAIGN_LEDGER",
                "prepared campaign completion result differs",
                terminal=True,
            )
        completion_body = completion_ledger_record.get("body")
        if (
            completion_ledger_record.get("record_type")
            != "CAMPAIGN_COMPLETION"
            or completion_ledger_record.get("status")
            != "CAMPAIGN_COMPLETION_COMMITTED"
            or type(completion_body) is not dict
            or frozen.get("schema_version")
            != "hegel-phase3a-q05b-shadow-retry-campaign-result/2"
            or frozen.get("campaign_id") != self.contract["contract_id"]
            or frozen.get("contract_digest") != self.contract["contract_digest"]
            or frozen.get("capsule_digest") != self.capsule["capsule_digest"]
            or frozen.get("campaign_completion_digest")
            != completion_body.get("record_digest")
            or frozen.get("ledger_completion_record_digest")
            != completion_ledger_record.get("record_digest")
            or frozen.get("authoritative_completion_record_name")
            != "campaign.completion.json"
            or frozen.get("completion_commit_verified") is not True
            or frozen.get("status") != completion_body.get("terminal_status")
            or frozen.get("supervisor_invocation_count")
            != completion_body.get("supervisor_invocation_count")
            or frozen.get("last_slot_id_or_null")
            != completion_body.get("last_slot_id_or_null")
            or frozen.get("last_slot_outcome_digest_or_null")
            != completion_body.get("last_slot_outcome_digest_or_null")
        ):
            _fail(
                "FAIL_CAMPAIGN_LEDGER",
                "prepared result does not bind the exact completion record",
                terminal=True,
            )
        self.completion_result_candidate_or_null = deepcopy(frozen)
        return deepcopy(frozen)

    def publish_prepared(
        self,
        basename: str,
        prepared_record: Mapping[str, object] | bytes,
    ) -> dict[str, object]:
        if type(prepared_record) is bytes:
            frozen_bytes = bytes(prepared_record)
        elif type(prepared_record) is dict:
            frozen_bytes = _canonical_json_bytes(dict(prepared_record))
        else:
            _fail(
                "FAIL_CAMPAIGN_LEDGER",
                "prepared ledger record differs",
                terminal=True,
            )
        frozen_record = _strict_json(
            frozen_bytes,
            f"prepared ledger record {basename}",
            _MAX_LEDGER_BYTES,
        )
        if type(frozen_record) is not dict:
            _fail(
                "FAIL_CAMPAIGN_LEDGER",
                "prepared ledger record is not an object",
                terminal=True,
            )
        record_type = frozen_record.get("record_type")
        status_text = frozen_record.get("status")
        body = frozen_record.get("body")
        if (
            type(record_type) is not str
            or type(status_text) is not str
            or type(body) is not dict
        ):
            _fail(
                "FAIL_CAMPAIGN_LEDGER",
                "prepared ledger record fields differ",
                terminal=True,
            )
        expected = self.prepare(basename, record_type, status_text, body)
        if frozen_record != expected:
            _fail(
                "FAIL_CAMPAIGN_LEDGER",
                "prepared ledger record digest or chain differs",
                terminal=True,
            )
        name_index = self.allowed_names.index(basename)
        next_sequence = frozen_record["sequence"]
        next_record_digest = frozen_record["record_digest"]
        assert type(next_sequence) is int and type(next_record_digest) is str
        next_names = [*self.published_names, basename]
        next_records = [
            *self.published_records,
            deepcopy(frozen_record),
        ]
        if basename == "campaign.completion.json":
            candidate = self.completion_result_candidate_or_null
            if (
                type(candidate) is not dict
                or candidate.get("ledger_completion_record_digest")
                != next_record_digest
                or candidate.get("campaign_completion_digest")
                != body.get("record_digest")
            ):
                _fail(
                    "FAIL_CAMPAIGN_LEDGER",
                    "completion result candidate is absent or stale",
                    terminal=True,
                )
            _verify_campaign_completion_chain_v2(
                self.contract,
                self.capsule,
                self.authorization,
                list(zip(next_names, next_records)),
            )
        if (
            _verify_held_directory_path(
                self.root_path,
                self.root_fd,
                "campaign ledger publication",
            )
            != self.root_identity
        ):
            _fail("FAIL_CAMPAIGN_LEDGER", "ledger root identity drifted", terminal=True)
        authoritative_candidate = (
            deepcopy(self.completion_result_candidate_or_null)
            if basename == "campaign.completion.json"
            else None
        )

        def commit_verified_publication() -> None:
            _postpublication_directory_identity(
                self.root_path,
                self.root_fd,
                "campaign ledger publication replay",
                self.root_identity,
            )
            # For completion this is deliberately the first state store after
            # replay.  Every later asynchronous fault can therefore return the
            # already verified, prebuilt authoritative result.
            if basename == "campaign.completion.json":
                assert type(authoritative_candidate) is dict
                self.authoritative_completion_result_or_null = deepcopy(
                    authoritative_candidate
                )
            self.sequence = next_sequence
            self.previous_record_digest = next_record_digest
            self.published_names = next_names
            self.published_records = next_records
            self.last_name_index = name_index
            if basename == "campaign.completion.json":
                self.completion_commit_verified = True

        publication_attempted = False
        try:
            publication_attempted = True
            _publish_immutable_at(
                self.root_fd,
                basename,
                frozen_bytes,
                publication_label="campaign-ledger-record",
                commit_callback=commit_verified_publication,
            )
            return deepcopy(frozen_record)
        except BaseException as error:
            if publication_attempted:
                _guard_publication_attempt_failure(
                    "campaign ledger state commit",
                    basename,
                    "FAIL_CAMPAIGN_LEDGER",
                    error,
                )
            raise

    def publish(
        self,
        basename: str,
        record_type: str,
        status_text: str,
        body: Mapping[str, object],
    ) -> dict[str, object]:
        return self.publish_prepared(
            basename,
            self.prepare_bytes(basename, record_type, status_text, body),
        )

    def close(self) -> None:
        active_error = sys.exc_info()[1]
        close_errors: list[tuple[str, BaseException]] = []
        if getattr(self, "root_fd", None) is not None:
            root_fd = self.root_fd
            self.root_fd = None
            if getattr(self, "locked", False):
                try:
                    fcntl.flock(root_fd, fcntl.LOCK_UN)
                except BaseException as error:
                    close_errors.append(("ledger unlock", error))
                self.locked = False
            try:
                os.close(root_fd)
            except BaseException as error:
                close_errors.append(("ledger root descriptor close", error))
        if getattr(self, "parent_fd", None) is not None:
            parent_fd = self.parent_fd
            self.parent_fd = None
            try:
                os.close(parent_fd)
            except BaseException as error:
                close_errors.append(("ledger parent descriptor close", error))
        if close_errors:
            if (
                isinstance(active_error, ShadowRetryCampaignErrorV2)
                and active_error.code
                in {
                    "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN",
                    "FAIL_CAMPAIGN_SPAWN_UNCERTAIN",
                }
            ):
                raise active_error
            if (
                getattr(self, "completion_commit_verified", False)
                or type(
                    getattr(
                        self,
                        "authoritative_completion_result_or_null",
                        None,
                    )
                )
                is dict
            ):
                return
            label, close_error = close_errors[0]
            if getattr(self, "published_names", []):
                _raise_postpublication_uncertain(label, close_error)
            _fail(
                "FAIL_CAMPAIGN_LEDGER",
                f"{label} failed before any ledger publication: "
                f"{type(close_error).__name__}:{close_error}",
                terminal=True,
            )


def _ledger_progress_snapshot_v2(
    published_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if type(published_records) not in (list, tuple):
        _fail(
            "FAIL_CAMPAIGN_LEDGER",
            "published ledger record snapshot differs",
            terminal=True,
        )
    consumed_count = 0
    last_slot_id: str | None = None
    last_outcome_digest: str | None = None
    for record in published_records:
        if type(record) is not dict or type(record.get("body")) is not dict:
            _fail(
                "FAIL_CAMPAIGN_LEDGER",
                "published ledger progress record differs",
                terminal=True,
            )
        if record.get("record_type") == "SLOT_CONSUMED":
            consumed_count += 1
        elif record.get("record_type") == "SLOT_OUTCOME":
            body = record["body"]
            slot_id = body.get("slot_id")
            outcome_digest = body.get("record_digest")
            if (
                slot_id not in {"SLOT_01", "SLOT_02", "SLOT_03"}
                or type(outcome_digest) is not str
                or _TAGGED_SHA256.fullmatch(outcome_digest) is None
            ):
                _fail(
                    "FAIL_CAMPAIGN_LEDGER",
                    "published slot outcome progress differs",
                    terminal=True,
                )
            last_slot_id = str(slot_id)
            last_outcome_digest = outcome_digest
    if not 0 <= consumed_count <= 3:
        _fail(
            "FAIL_CAMPAIGN_LEDGER",
            "published invocation count differs",
            terminal=True,
        )
    return {
        "supervisor_invocation_count": consumed_count,
        "last_slot_id_or_null": last_slot_id,
        "last_slot_outcome_digest_or_null": last_outcome_digest,
    }


def _binding_record(
    contract: Mapping[str, object],
    capsule: Mapping[str, object],
    slot: Mapping[str, object],
    process_result: Mapping[str, object],
    stdout_facts: Mapping[str, object],
    trajectory: Mapping[str, object],
    supervisor_artifact: Mapping[str, object],
    external_artifact: Mapping[str, object],
    docker_absent: bool,
    docker_rows: Sequence[Mapping[str, str]],
    docker_error: str | None,
    source_replay_before: Mapping[str, object],
    source_replay_after: Mapping[str, object],
    future_slot_freshness: Mapping[str, object],
    cleanup_replay: Mapping[str, object],
) -> dict[str, object]:
    trace_root = Path(str(slot["trace_root"]))
    trace_fd, _trace_stat = _open_directory_nofollow(
        trace_root,
        "trajectory root external binding",
    )
    try:
        trace_identity = _verify_held_directory_path(
            trace_root,
            trace_fd,
            "trajectory root external binding",
        )
    finally:
        os.close(trace_fd)
    if (
        not stat.S_ISDIR(int(trace_identity["mode"]))
        or stat.S_IMODE(int(trace_identity["mode"])) != 0o700
        or trace_identity["uid"] != os.geteuid()
        or int(trace_identity["nlink"]) < 2
        or int(trace_identity["mount_id"]) <= 0
    ):
        _fail(
            "FAIL_CAMPAIGN_BINDING",
            "trajectory root external identity differs",
            terminal=True,
        )
    target = contract["target_commit_a_binding"]
    assert type(target) is dict
    required_hashes = target["required_raw_sha256_by_path"]
    assert type(required_hashes) is dict
    value: dict[str, object] = {
        "schema_version": "hegel-phase3a-q05b-shadow-retry-external-trajectory-binding/2",
        "campaign_id": contract["contract_id"],
        "contract_digest": contract["contract_digest"],
        "capsule_digest": capsule["capsule_digest"],
        "slot_id": slot["slot_id"],
        "slot_ordinal": slot["slot_ordinal"],
        "target_supervisor_commit": target["target_supervisor_commit"],
        "target_supervisor_tool_raw_sha256": required_hashes[
            "tools/phase3_q05b_dual_qualification_v2.py"
        ],
        "trajectory_core_raw_sha256": required_hashes[
            "src/hegel_machine/phase3_q05b_trajectory_v2.py"
        ],
        "trace_root_path": slot["trace_root"],
        "trace_root_device": trace_identity["dev"],
        "trace_root_inode": trace_identity["ino"],
        "trace_root_mount_id": trace_identity["mount_id"],
        "trace_root_mode": trace_identity["mode"],
        "trace_root_uid": trace_identity["uid"],
        "trace_root_nlink": trace_identity["nlink"],
        "trace_id": trajectory["trace_id_or_null"],
        "producer_id": trajectory["producer_id_or_null"],
        "clock_id": trajectory.get("clock_id_or_null"),
        "verification_status": trajectory["verification_status"],
        "complete_verification_passed": trajectory[
            "complete_verification_passed"
        ],
        "verification_error_or_null": trajectory[
            "verification_error_or_null"
        ],
        "terminal_manifest_outcome": trajectory["terminal_outcome_or_null"],
        "terminal_manifest_raw_sha256": trajectory[
            "terminal_manifest_raw_sha256_or_null"
        ],
        "terminal_manifest_digest": trajectory[
            "terminal_manifest_digest_or_null"
        ],
        "event_count": trajectory["event_count"],
        "first_event_record_digest": trajectory[
            "first_event_record_digest_or_null"
        ],
        "final_event_record_digest": trajectory[
            "final_event_record_digest_or_null"
        ],
        "terminal_event_type": trajectory["terminal_event_type_or_null"],
        "terminal_error_code_or_null": trajectory[
            "terminal_error_code_or_null"
        ],
        "terminal_error_detail_or_null": trajectory[
            "terminal_error_detail_or_null"
        ],
        "torn_tail_staging_basename_or_null": trajectory[
            "torn_tail_staging_basename_or_null"
        ],
        "open_span_ids_in_begin_order": deepcopy(
            list(trajectory["open_span_ids_in_begin_order"])
        ),
        "open_span_contexts_in_begin_order": deepcopy(
            list(trajectory["open_span_contexts_in_begin_order"])
        ),
        "closed_span_count": trajectory.get("closed_span_count_or_null"),
        "failure_rows_retained_in_sequence_order": deepcopy(
            list(trajectory["failure_rows_retained_in_sequence_order"])
        ),
        "cleanup_rows_retained_in_sequence_order": deepcopy(
            list(trajectory["cleanup_rows_retained_in_sequence_order"])
        ),
        "partition_certificate_or_null": deepcopy(
            trajectory["partition_certificate_or_null"]
        ),
        "last_durable_event_row_or_null": deepcopy(
            trajectory["last_durable_event_row_or_null"]
        ),
        "canonical_event_payloads_digest": trajectory[
            "canonical_event_payloads_digest"
        ],
        "success_trace_bindings_or_null": deepcopy(
            trajectory["success_trace_bindings_or_null"]
        ),
        "process_return_code": process_result["return_code"],
        "process_timed_out": process_result["timed_out"],
        "process_stdout_overflow": process_result["stdout_overflow"],
        "process_stderr_overflow": process_result["stderr_overflow"],
        "process_child_reaped": process_result["child_reaped"],
        "process_group_absent": process_result["process_group_absent"],
        "process_wrapper_resources_closed": process_result[
            "wrapper_resources_closed"
        ],
        "stdout_raw_sha256": sha256(process_result["stdout"]).hexdigest(),
        "stdout_byte_count": len(process_result["stdout"]),
        "stderr_raw_sha256": sha256(process_result["stderr"]).hexdigest(),
        "stderr_byte_count": len(process_result["stderr"]),
        "supervisor_artifact_present": supervisor_artifact["state"] != "ABSENT",
        "supervisor_artifact_raw_sha256_or_null": supervisor_artifact[
            "raw_sha256_or_null"
        ],
        "external_artifact_present": external_artifact["state"] != "ABSENT",
        "external_artifact_raw_sha256_or_null": external_artifact[
            "raw_sha256_or_null"
        ],
        "supervisor_artifact_evidence": deepcopy(dict(supervisor_artifact)),
        "external_artifact_evidence": deepcopy(dict(external_artifact)),
        "docker_inventory_evidence": {
            "absent": docker_absent,
            "rows": [deepcopy(dict(row)) for row in docker_rows],
            "error_or_null": docker_error,
        },
        "source_replay_before_evidence": deepcopy(
            dict(source_replay_before)
        ),
        "source_replay_after_evidence": deepcopy(dict(source_replay_after)),
        "future_slot_freshness_evidence": deepcopy(
            dict(future_slot_freshness)
        ),
        "cleanup_replay_evidence": deepcopy(dict(cleanup_replay)),
        "source_replay_before_root": source_replay_before["root"],
        "source_replay_after_root": source_replay_after["root"],
        "cleanup_replay_root": cleanup_replay["root"],
        "binding_digest": None,
    }
    expected_fields = contract["trajectory_binding_policy"][
        "external_binding_exact_fields"
    ]
    if (
        type(expected_fields) is not list
        or any(type(field) is not str for field in expected_fields)
        or len(expected_fields) != len(set(expected_fields))
        or len(expected_fields) != len(value)
        or set(value) != set(expected_fields)
    ):
        _fail(
            "FAIL_CAMPAIGN_BINDING",
            "external trajectory binding field registry differs",
            terminal=True,
        )
    if (
        docker_absent is not True
        or docker_rows
        or docker_error is not None
        or process_result["child_reaped"] is not True
        or process_result["process_group_absent"] is not True
        or process_result["wrapper_resources_closed"] is not True
    ):
        # The binding is still published as evidence, but cleanup certainty is
        # carried by cleanup_replay_root and cannot be elevated by this helper.
        pass
    unsigned = deepcopy(value)
    unsigned.pop("binding_digest")
    value["binding_digest"] = _domain_digest(
        contract["hash_domains"]["trajectory_binding"],
        unsigned,
    )
    return value


def _safe_docker_inventory(
    contract: Mapping[str, object],
) -> tuple[bool, list[dict[str, str]], str | None]:
    try:
        return _docker_inventory(contract)
    except BaseException as error:
        diagnostic = _bounded_process_exception_diagnostic(
            "Docker read-only inventory",
            _docker_inventory_argv_v2(contract),
            error,
        )
        return (
            False,
            [],
            _canonical_json_bytes(diagnostic).decode("ascii").rstrip("\n"),
        )


def _source_state_replay(
    contract: Mapping[str, object],
    context: Mapping[str, object],
    slot: Mapping[str, object],
    *,
    artifact_presence: str,
) -> dict[str, object]:
    execution = contract["execution_policy"]
    assert type(execution) is dict
    target_git = _verify_target_git_objects(contract)
    controller_git = _controller_git_identity(contract)
    clone = _verify_clone(contract, slot, artifact_state=artifact_presence)
    clone_closure_rows = clone.pop("clone_closure_rows_internal")
    cargo_source, source_rows = _scan_tree(
        Path(str(execution["cargo_copy_source"])),
        label=f"{slot['slot_id']} Cargo source replay",
        allowed_uids=(os.geteuid(), 0),
        reject_group_or_world_writable=True,
    )
    cargo_cache, cache_rows = _scan_tree(
        Path(str(slot["cargo_cache_root"])),
        label=f"{slot['slot_id']} Cargo cache replay",
    )
    if (
        target_git != context["target_git"]
        or controller_git != context["controller_git"]
        or source_rows != context["cargo_rows"]
        or cargo_source["manifest_digest"]
        != context["cargo_identity"]["manifest_digest"]
        or cargo_cache["content_manifest_digest"]
        != context["cargo_identity"]["content_manifest_digest"]
        or [
            {key: value for key, value in row.items() if key != "uid"}
            for row in cache_rows
        ]
        != [
            {key: value for key, value in row.items() if key != "uid"}
            for row in context["cargo_rows"]
        ]
        or any(row["uid"] != os.geteuid() for row in cache_rows)
    ):
        _fail(
            "FAIL_CAMPAIGN_SOURCE",
            f"{slot['slot_id']} source replay differs",
            terminal=True,
        )
    evidence = {
        "target_git": target_git,
        "controller_git": controller_git,
        "clone": clone,
        "cargo_source": cargo_source,
        "cargo_cache": cargo_cache,
    }
    return {
        "certain": True,
        "error_or_null": None,
        "evidence": evidence,
        "clone_closure_rows_internal": clone_closure_rows,
        "root": _domain_digest(
            "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R8/SOURCE_REPLAY/V2",
            evidence,
        ),
    }


def _safe_source_state_replay(
    contract: Mapping[str, object],
    context: Mapping[str, object],
    slot: Mapping[str, object],
    *,
    artifact_presence: str,
) -> dict[str, object]:
    try:
        return _source_state_replay(
            contract,
            context,
            slot,
            artifact_presence=artifact_presence,
        )
    except BaseException as error:
        detail = (
            f"{getattr(error, 'code', type(error).__name__)}:"
            f"{getattr(error, 'detail', str(error))}"
        )[:4096]
        receipt = getattr(error, "receipt", None)
        evidence = {
            "error": detail,
            "error_receipt_or_null": (
                deepcopy(dict(receipt)) if type(receipt) is dict else None
            ),
            "artifact_presence": artifact_presence,
        }
        return {
            "certain": False,
            "error_or_null": detail,
            "evidence": evidence,
            "root": _domain_digest(
                "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R8/SOURCE_REPLAY/V2",
                evidence,
            ),
        }


def _augment_source_replay_with_alias_rows(
    source_replay: Mapping[str, object],
    alias_rows: Mapping[str, object],
) -> dict[str, object]:
    result = deepcopy(dict(source_replay))
    evidence = deepcopy(dict(result["evidence"]))
    evidence["nonalias_directory_identities"] = deepcopy(dict(alias_rows))
    result["evidence"] = evidence
    result["root"] = _domain_digest(
        "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R8/SOURCE_REPLAY/V2",
        evidence,
    )
    return result


def _enforce_clone_closure(
    slot: Mapping[str, object],
    source_before: Mapping[str, object],
    source_after: Mapping[str, object],
) -> dict[str, object]:
    result = deepcopy(dict(source_after))
    after_rows = result.pop("clone_closure_rows_internal", None)
    before_rows = source_before.get("clone_closure_rows_internal")
    if source_before.get("certain") is not True or result.get("certain") is not True:
        return result
    before_clone = source_before["evidence"]["clone"]
    after_clone = result["evidence"]["clone"]
    before_closure = before_clone["clone_closure_without_artifact_digest"]
    after_closure = after_clone["clone_closure_without_artifact_digest"]
    if before_closure == after_closure:
        return result
    if type(before_rows) is not list or type(after_rows) is not list:
        before_rows = []
        after_rows = []
    before_by_path = {
        str(row["relative_path"]): deepcopy(dict(row))
        for row in before_rows
        if type(row) is dict and type(row.get("relative_path")) is str
    }
    after_by_path = {
        str(row["relative_path"]): deepcopy(dict(row))
        for row in after_rows
        if type(row) is dict and type(row.get("relative_path")) is str
    }
    added_paths = sorted(set(after_by_path) - set(before_by_path))
    removed_paths = sorted(set(before_by_path) - set(after_by_path))
    changed_paths = sorted(
        path
        for path in set(before_by_path) & set(after_by_path)
        if before_by_path[path] != after_by_path[path]
    )
    limit = 64
    difference = {
        "added_total": len(added_paths),
        "removed_total": len(removed_paths),
        "changed_total": len(changed_paths),
        "added_rows": [after_by_path[path] for path in added_paths[:limit]],
        "removed_rows": [before_by_path[path] for path in removed_paths[:limit]],
        "changed_rows": [
            {
                "relative_path": path,
                "before": before_by_path[path],
                "after": after_by_path[path],
            }
            for path in changed_paths[:limit]
        ],
        "truncated": any(
            len(paths) > limit
            for paths in (added_paths, removed_paths, changed_paths)
        ),
    }
    first_paths = (added_paths + removed_paths + changed_paths)[:8]
    detail = (
        f"{slot['slot_id']} clone closure changed outside the single allowed "
        f"artifact path: before={before_closure}; after={after_closure}; "
        f"before_entries={before_clone['clone_tree_entry_count']}; "
        f"after_entries={after_clone['clone_tree_entry_count']}; "
        f"first_changed_paths={first_paths}"
    )
    evidence = deepcopy(dict(result["evidence"]))
    evidence["clone_closure_comparison"] = {
        "before": before_closure,
        "after": after_closure,
        "equal": False,
        "error": detail,
        "difference": difference,
    }
    return {
        "certain": False,
        "error_or_null": detail,
        "evidence": evidence,
        "root": _domain_digest(
            "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R8/SOURCE_REPLAY/V2",
            evidence,
        ),
    }


def _invalid_trajectory(error: BaseException | str) -> dict[str, object]:
    detail = (
        error
        if type(error) is str
        else (
            f"{getattr(error, 'code', type(error).__name__)}:"
            f"{getattr(error, 'detail', str(error))}"
        )
    )
    return {
        "verification_status": "INVALID",
        "complete": False,
        "complete_verification_passed": False,
        "verification_error_or_null": str(detail)[:4096],
        "trace_id_or_null": None,
        "producer_id_or_null": None,
        "event_count": 0,
        "first_event_record_digest_or_null": None,
        "final_event_record_digest_or_null": None,
        "terminal_outcome_or_null": None,
        "terminal_manifest_digest_or_null": None,
        "terminal_manifest_raw_sha256_or_null": None,
        "clock_id_or_null": None,
        "closed_span_count_or_null": None,
        "terminal_event_type_or_null": None,
        "terminal_error_code_or_null": None,
        "terminal_error_detail_or_null": None,
        "torn_tail_staging_basename_or_null": None,
        "open_span_ids_in_begin_order": [],
        "open_span_contexts_in_begin_order": [],
        "failure_rows_retained_in_sequence_order": [],
        "cleanup_rows_retained_in_sequence_order": [],
        "partition_certificate_or_null": None,
        "last_durable_event_row_or_null": None,
        "canonical_event_payloads_digest": None,
        "success_trace_bindings_or_null": None,
    }


def _safe_trajectory_evidence(
    contract: Mapping[str, object],
    slot: Mapping[str, object],
    *,
    source_certain: bool,
) -> dict[str, object]:
    # The trajectory module performs its own exact target hash check before
    # execution.  Always attempt producer-free replay so a separate clone or
    # Cargo closure failure cannot erase the causal trace that explains it.
    _ = source_certain
    try:
        return _trajectory_evidence(contract, slot)
    except BaseException as error:
        return _invalid_trajectory(error)


def _trajectory_complete_and_bound(
    contract: Mapping[str, object],
    slot: Mapping[str, object],
    trajectory: Mapping[str, object],
) -> bool:
    policy = contract["trajectory_binding_policy"]
    assert type(policy) is dict
    status = trajectory.get("verification_status")
    expected_terminal = (
        "OK" if status == "COMPLETE_OK" else "FAIL" if status == "COMPLETE_FAIL" else None
    )
    event_count = trajectory.get("event_count")
    first_digest = trajectory.get("first_event_record_digest_or_null")
    final_digest = trajectory.get("final_event_record_digest_or_null")
    payload_digest = trajectory.get("canonical_event_payloads_digest")
    last_event = trajectory.get("last_durable_event_row_or_null")
    failures = trajectory.get("failure_rows_retained_in_sequence_order")
    cleanups = trajectory.get("cleanup_rows_retained_in_sequence_order")
    certificate = trajectory.get("partition_certificate_or_null")
    common = (
        trajectory.get("complete") is True
        and trajectory.get("complete_verification_passed") is True
        and trajectory.get("verification_error_or_null") is None
        and trajectory.get("trace_id_or_null") == slot["trajectory_trace_id"]
        and trajectory.get("producer_id_or_null") == policy["producer_id"]
        and trajectory.get("clock_id_or_null") == policy["clock_id"]
        and trajectory.get("torn_tail_staging_basename_or_null") is None
        and trajectory.get("open_span_ids_in_begin_order") == []
        and trajectory.get("open_span_contexts_in_begin_order") == []
        and trajectory.get("terminal_outcome_or_null") == expected_terminal
        and trajectory.get("terminal_event_type_or_null") == expected_terminal
        and type(event_count) is int
        and event_count >= 1
        and type(first_digest) is str
        and _TAGGED_SHA256.fullmatch(first_digest) is not None
        and type(final_digest) is str
        and _TAGGED_SHA256.fullmatch(final_digest) is not None
        and type(payload_digest) is str
        and _TAGGED_SHA256.fullmatch(payload_digest) is not None
        and type(last_event) is dict
        and set(last_event) == set(_TRAJECTORY_MINIMAL_EVENT_FIELDS)
        and last_event.get("sequence") == event_count
        and last_event.get("record_digest") == final_digest
        and last_event.get("event_type") == expected_terminal
        and type(failures) is list
        and len(failures) <= _MAX_RETAINED_FAILURE_ROWS
        and type(cleanups) is list
        and len(cleanups) <= _MAX_RETAINED_CLEANUP_ROWS
        and type(trajectory.get("terminal_manifest_raw_sha256_or_null")) is str
        and _RAW_SHA256.fullmatch(
            str(trajectory.get("terminal_manifest_raw_sha256_or_null"))
        )
        is not None
        and type(trajectory.get("terminal_manifest_digest_or_null")) is str
        and _TAGGED_SHA256.fullmatch(
            str(trajectory.get("terminal_manifest_digest_or_null"))
        )
        is not None
    )
    if not common:
        return False
    if status == "COMPLETE_OK":
        return (
            certificate is None
            and failures == []
            and cleanups == []
            and type(trajectory.get("success_trace_bindings_or_null")) is dict
        )
    if status != "COMPLETE_FAIL" or type(certificate) is not dict:
        return False
    required_certificate_fields = {
        "schema_version",
        "trace_id",
        "producer_id",
        "trajectory_status",
        "event_count",
        "first_event_record_digest",
        "final_event_record_digest",
        "canonical_event_payloads_digest",
        "origin_sequence",
        "origin_record_digest",
        "origin_actor_or_null",
        "suffix_event_count",
        "suffix_event_root",
        "failure_event_count",
        "failure_rows_digest",
        "failure_rows_retained_count",
        "failure_rows_truncated",
        "cleanup_nonfailure_event_count",
        "cleanup_rows_digest",
        "cleanup_rows_retained_count",
        "cleanup_rows_truncated",
        "other_nonfailure_event_count",
        "other_nonfailure_event_root",
        "partitioned_event_count",
        "unpartitioned_event_count",
        "certificate_digest",
    }
    if (
        set(certificate) != required_certificate_fields
        or certificate.get("schema_version") != _PARTITION_CERTIFICATE_SCHEMA_V2
        or certificate.get("trace_id") != trajectory.get("trace_id_or_null")
        or certificate.get("producer_id") != trajectory.get("producer_id_or_null")
        or certificate.get("trajectory_status") != "COMPLETE_FAIL"
        or certificate.get("event_count") != event_count
        or certificate.get("first_event_record_digest") != first_digest
        or certificate.get("final_event_record_digest") != final_digest
        or certificate.get("canonical_event_payloads_digest") != payload_digest
        or certificate.get("failure_rows_retained_count") != len(failures)
        or certificate.get("cleanup_rows_retained_count") != len(cleanups)
        or len(_canonical_json_bytes(certificate))
        > _MAX_PARTITION_CERTIFICATE_BYTES
    ):
        return False
    unsigned = deepcopy(certificate)
    observed_digest = unsigned.pop("certificate_digest", None)
    return observed_digest == _domain_digest(
        str(contract["hash_domains"]["trajectory_partition_certificate"]),
        unsigned,
    )


def _artifact_matches_success_wire(
    contract: Mapping[str, object],
    slot: Mapping[str, object],
    stdout_facts: Mapping[str, object],
    supervisor_artifact: Mapping[str, object],
    trajectory: Mapping[str, object],
) -> bool:
    core = _load_core(contract)
    wire = stdout_facts.get("stdout_wire_value_or_null")
    summary = supervisor_artifact.get("summary_or_null")
    trace_bindings = trajectory.get("success_trace_bindings_or_null")
    target = contract["target_commit_a_binding"]
    delivery = (
        wire.get("actual_final_delivery_identity")
        if type(wire) is dict
        else None
    )
    parent_identity = supervisor_artifact.get("parent_identity_or_null")
    file_identity = supervisor_artifact.get("file_identity_or_null")
    return (
        stdout_facts.get("stdout_wire_kind")
        == core.STDOUT_WIRE_SUCCESS_RESULT_V2
        and type(wire) is dict
        and supervisor_artifact.get("state") == "VALID"
        and supervisor_artifact.get("replay_passed") is True
        and type(summary) is dict
        and type(trace_bindings) is dict
        and wire.get("artifact_path") == slot["supervisor_artifact_path"]
        and wire.get("artifact_sha256")
        == supervisor_artifact.get("raw_sha256_or_null")
        and summary.get("canonical_sha256")
        == supervisor_artifact.get("raw_sha256_or_null")
        and wire.get("artifact_set_root") == summary.get("artifact_set_root")
        and wire.get("candidate_receipt_root")
        == summary.get("candidate_receipt_root")
        and wire.get("final_receipt_root") == summary.get("final_receipt_root")
        and wire.get("qualification_count") == summary.get("qualification_count")
        and wire.get("qualification_mask") == summary.get("qualification_mask")
        and type(summary.get("closed_q1_authority")) is dict
        and wire.get("q1_state")
        == summary["closed_q1_authority"].get("state")
        and wire.get("q1_gate_count")
        == summary["closed_q1_authority"].get("gate_count")
        and wire.get("q1_gate_mask")
        == summary["closed_q1_authority"].get("gate_mask")
        and wire.get("q1_formal_output_roots")
        == summary["closed_q1_authority"].get("formal_output_roots")
        and summary.get("actual_admission_source_commit")
        == target["target_supervisor_commit"]
        and summary.get("actual_admission_artifact_path")
        == slot["supervisor_artifact_path"]
        and wire.get("actual_admission_pre_artifact_live_replay_root")
        == summary.get("pre_artifact_live_marker_replay_root")
        and wire.get("stage_roots") == trace_bindings.get("stage_roots")
        and wire.get("artifact_sha256")
        == trace_bindings.get("adapter_artifact_sha256")
        and wire.get("artifact_sha256")
        == trace_bindings.get("publication_artifact_sha256")
        and wire.get("actual_admission_prepublication_live_replay_root")
        == trace_bindings.get("prepublication_live_marker_replay_root")
        and wire.get("publisher_handle_close_status") == "CLOSED"
        and type(delivery) is dict
        and type(parent_identity) is dict
        and type(file_identity) is dict
        and delivery.get("publisher_parent_device")
        == parent_identity.get("dev")
        and delivery.get("publisher_parent_inode")
        == parent_identity.get("ino")
        and delivery.get("publisher_file_device")
        == file_identity.get("dev")
        and delivery.get("publisher_file_inode")
        == file_identity.get("ino")
        and delivery.get("publisher_file_mode")
        == file_identity.get("mode")
        and delivery.get("publisher_file_nlink")
        == file_identity.get("nlink")
        and delivery.get("payload_length") == file_identity.get("size")
        and delivery.get("payload_sha256")
        == supervisor_artifact.get("raw_sha256_or_null")
        and delivery.get("first_anchored_replay_sha256")
        == supervisor_artifact.get("raw_sha256_or_null")
        and delivery.get(
            "actual_admission_evidence_root"
        )
        == summary.get("actual_admission_evidence_root")
        and delivery.get("source_commit")
        == target["target_supervisor_commit"]
    )


def _artifact_observation_facts(
    supervisor_artifact: Mapping[str, object],
    external_artifact: Mapping[str, object],
) -> tuple[str, str, str]:
    if (
        supervisor_artifact.get("state") == "VALID"
        and supervisor_artifact.get("replay_passed") is True
        and external_artifact.get("state") == "VALID"
        and external_artifact.get("replay_passed") is True
        and supervisor_artifact.get("raw_sha256_or_null")
        == external_artifact.get("raw_sha256_or_null")
    ):
        return "VALID", "VALID", "VALID"
    if (
        supervisor_artifact.get("state") == "ABSENT"
        and external_artifact.get("state") == "ABSENT"
    ):
        return "ABSENT", "ABSENCE_VERIFIED", "ABSENCE_REPLAY_VERIFIED"
    return "RESIDUAL_INVALID", "INVALID", "INVALID"


def _cleanup_replay_evidence(
    process_result: Mapping[str, object],
    trajectory: Mapping[str, object],
    docker_absent: bool,
    docker_rows: Sequence[Mapping[str, str]],
    docker_error: str | None,
) -> dict[str, object]:
    value = {
        "process": {
            key: process_result[key]
            for key in (
                "return_code",
                "stdout_overflow",
                "stderr_overflow",
                "timed_out",
                "child_reaped",
                "process_group_absent",
                "wrapper_resources_closed",
            )
        },
        "trajectory_status": trajectory["verification_status"],
        "trajectory_terminal_manifest_digest_or_null": trajectory[
            "terminal_manifest_digest_or_null"
        ],
        "trajectory_open_span_ids": deepcopy(
            list(trajectory["open_span_ids_in_begin_order"])
        ),
        "docker_absent": docker_absent,
        "docker_rows": [deepcopy(dict(row)) for row in docker_rows],
        "docker_error_or_null": docker_error,
    }
    return {
        "preimage": value,
        "root": _domain_digest(
            "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R8/CLEANUP_REPLAY/V2",
            value,
        ),
    }


def _build_slot_observation(
    contract: Mapping[str, object],
    slot: Mapping[str, object],
    process_result: Mapping[str, object],
    stdout_facts: Mapping[str, object],
    trajectory: Mapping[str, object],
    supervisor_artifact: Mapping[str, object],
    external_artifact: Mapping[str, object],
    *,
    source_before_certain: bool,
    source_after_certain: bool,
    docker_absent: bool,
    external_binding_certain: bool,
) -> dict[str, object]:
    core = _load_core(contract)
    artifact_state, artifact_validation, artifact_replay = (
        _artifact_observation_facts(supervisor_artifact, external_artifact)
    )
    trajectory_complete = _trajectory_complete_and_bound(
        contract, slot, trajectory
    )
    cleanup_certain = (
        process_result["child_reaped"] is True
        and process_result["process_group_absent"] is True
        and process_result["wrapper_resources_closed"] is True
        and process_result["timed_out"] is False
        and docker_absent is True
        and trajectory_complete
    )
    stderr = process_result["stderr"]
    stdout = process_result["stdout"]
    maximum_capture = int(core.CAPTURE_MAX_BYTES_V2)
    observation: dict[str, object] = {
        "schema_version": core.SLOT_OBSERVATION_SCHEMA_V2,
        "campaign_id": contract["contract_id"],
        "slot_number": slot["slot_ordinal"],
        "expected_trace_id": slot["trajectory_trace_id"],
        "expected_producer_id": contract["trajectory_binding_policy"][
            "producer_id"
        ],
        "invocation_count": 1,
        "supervisor_returned": type(process_result["return_code"]) is int,
        "supervisor_reaped": process_result["child_reaped"],
        "process_group_absent": process_result["process_group_absent"],
        "exit_code_or_null": process_result["return_code"],
        "stdout_capture_complete": (
            process_result["stdout_overflow"] is False
            and process_result["timed_out"] is False
        ),
        "stdout_capture_bounded": (
            process_result["stdout_overflow"] is False
            and len(stdout) <= maximum_capture
        ),
        **deepcopy(dict(stdout_facts)),
        "stderr_capture_complete": (
            process_result["stderr_overflow"] is False
            and process_result["timed_out"] is False
        ),
        "stderr_capture_bounded": (
            process_result["stderr_overflow"] is False
            and len(stderr) <= maximum_capture
        ),
        "stderr_raw_byte_count": len(stderr),
        "stderr_raw_sha256": sha256(stderr).hexdigest(),
        "stderr_empty": stderr == b"",
        "trajectory_status": trajectory["verification_status"],
        "trajectory_trace_id": trajectory["trace_id_or_null"],
        "trajectory_producer_id": trajectory["producer_id_or_null"],
        "trajectory_torn_tail_or_null": trajectory[
            "torn_tail_staging_basename_or_null"
        ],
        "trajectory_open_span_ids": deepcopy(
            list(trajectory["open_span_ids_in_begin_order"])
        ),
        "trajectory_terminal_error_code_or_null": trajectory[
            "terminal_error_code_or_null"
        ],
        "trajectory_terminal_error_detail_or_null": trajectory[
            "terminal_error_detail_or_null"
        ],
        "trajectory_terminal_event_type_or_null": trajectory[
            "terminal_event_type_or_null"
        ],
        "trajectory_event_count": trajectory["event_count"],
        "trajectory_first_event_record_digest": trajectory[
            "first_event_record_digest_or_null"
        ],
        "trajectory_final_event_record_digest": trajectory[
            "final_event_record_digest_or_null"
        ],
        "trajectory_canonical_event_payloads_digest": trajectory[
            "canonical_event_payloads_digest"
        ],
        "trajectory_partition_certificate_or_null": deepcopy(
            trajectory["partition_certificate_or_null"]
        ),
        "trajectory_failure_rows": deepcopy(
            list(trajectory["failure_rows_retained_in_sequence_order"])
        ),
        "trajectory_cleanup_rows": deepcopy(
            list(trajectory["cleanup_rows_retained_in_sequence_order"])
        ),
        "artifact_state": artifact_state,
        "artifact_validation_status": artifact_validation,
        "artifact_replay_status": artifact_replay,
        "cleanup_certain": cleanup_certain,
        "docker_absence_certain": docker_absent,
        "clone_clean_certain": source_before_certain and source_after_certain,
        "cargo_isolation_certain": source_before_certain and source_after_certain,
        "source_binding_certain": source_before_certain and source_after_certain,
        "external_binding_certain": external_binding_certain,
        "replay_certain": (
            trajectory_complete
            and source_before_certain
            and source_after_certain
            and external_binding_certain
        ),
        "record_digest": None,
    }
    if set(observation) != set(core.SLOT_OBSERVATION_FIELDS_V2):
        _fail(
            "FAIL_CAMPAIGN_OBSERVATION",
            "slot observation field registry differs",
            terminal=True,
        )
    observation["record_digest"] = core.slot_observation_record_digest_v2(
        observation
    )
    return observation


def _create_evidence_root(
    slot: Mapping[str, object],
) -> tuple[int, int]:
    evidence_root = Path(str(slot["evidence_root"]))
    return _create_private_root(evidence_root, f"{slot['slot_id']} evidence root")


def _publish_raw_streams(
    slot: Mapping[str, object],
    evidence_fd: int,
    process_result: Mapping[str, object],
    maximum_bytes: int,
) -> None:
    evidence_root = Path(str(slot["evidence_root"]))
    stdout_path = Path(str(slot["stdout_path"]))
    stderr_path = Path(str(slot["stderr_path"]))
    if stdout_path.parent != evidence_root or stderr_path.parent != evidence_root:
        _fail(
            "FAIL_CAMPAIGN_EVIDENCE",
            "raw stream evidence path relation differs",
            terminal=True,
        )
    _publish_immutable_at(
        evidence_fd,
        stdout_path.name,
        process_result["stdout"],
        publication_label="slot-supervisor-stdout",
        maximum_bytes=maximum_bytes,
        known_failure_code="FAIL_CAMPAIGN_EVIDENCE",
    )
    _publish_immutable_at(
        evidence_fd,
        stderr_path.name,
        process_result["stderr"],
        publication_label="slot-supervisor-stderr",
        maximum_bytes=maximum_bytes,
        known_failure_code="FAIL_CAMPAIGN_EVIDENCE",
    )


def _slot_outcome_record(
    contract: Mapping[str, object],
    capsule: Mapping[str, object],
    slot: Mapping[str, object],
    binding: Mapping[str, object],
    observation: Mapping[str, object],
    decision: str,
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "hegel-phase3a-q05b-shadow-retry-slot-outcome/2",
        "campaign_id": contract["contract_id"],
        "contract_digest": contract["contract_digest"],
        "capsule_digest": capsule["capsule_digest"],
        "slot_id": slot["slot_id"],
        "slot_ordinal": slot["slot_ordinal"],
        "external_trajectory_binding_digest": binding["binding_digest"],
        "observation": deepcopy(dict(observation)),
        "decision": decision,
        "record_digest": None,
    }
    return _record_digest(contract, "slot_outcome", value)


def _campaign_terminal_body(
    contract: Mapping[str, object],
    capsule: Mapping[str, object],
    *,
    status_text: str,
    invocation_count: int,
    last_slot_id_or_null: str | None,
    last_slot_outcome_digest_or_null: str | None,
    error_code_or_null: str | None = None,
    error_detail_or_null: str | None = None,
    error_receipt_or_null: Mapping[str, object] | None = None,
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "hegel-phase3a-q05b-shadow-retry-campaign-terminal/2",
        "campaign_id": contract["contract_id"],
        "contract_digest": contract["contract_digest"],
        "capsule_digest": capsule["capsule_digest"],
        "status": status_text,
        "supervisor_invocation_count": invocation_count,
        "last_slot_id_or_null": last_slot_id_or_null,
        "last_slot_outcome_digest_or_null": last_slot_outcome_digest_or_null,
        "error_code_or_null": error_code_or_null,
        "error_detail_or_null": error_detail_or_null,
        "error_receipt_or_null": _bounded_receipt_or_null(
            error_receipt_or_null
        ),
        "shadow_only": True,
        "nonadopting": True,
        "q1_state": "NOT_RUN",
        "active_transition": False,
        "adoption": False,
        "promotion": False,
        "current_pointer_written": False,
        "record_digest": None,
    }
    return _record_digest(contract, "campaign_terminal", value)


def _campaign_completion_body(
    contract: Mapping[str, object],
    capsule: Mapping[str, object],
    *,
    authorization_digest: str,
    terminal: Mapping[str, object],
    terminal_ledger_record: Mapping[str, object],
    preceding_ledger_names: Sequence[str],
) -> dict[str, object]:
    if (
        type(authorization_digest) is not str
        or _TAGGED_SHA256.fullmatch(authorization_digest) is None
        or type(terminal) is not dict
        or type(terminal_ledger_record) is not dict
        or type(preceding_ledger_names) not in (list, tuple)
        or any(type(name) is not str for name in preceding_ledger_names)
        or not preceding_ledger_names
        or preceding_ledger_names[-1] != "campaign.terminal.json"
    ):
        _fail(
            "FAIL_CAMPAIGN_LEDGER",
            "campaign completion preimage differs",
            terminal=True,
        )
    terminal_body_digest = terminal.get("record_digest")
    terminal_ledger_digest = terminal_ledger_record.get("record_digest")
    terminal_sequence = terminal_ledger_record.get("sequence")
    if (
        type(terminal_body_digest) is not str
        or _TAGGED_SHA256.fullmatch(terminal_body_digest) is None
        or type(terminal_ledger_digest) is not str
        or _TAGGED_SHA256.fullmatch(terminal_ledger_digest) is None
        or type(terminal_sequence) is not int
        or terminal_sequence != len(preceding_ledger_names)
        or terminal_ledger_record.get("record_type") != "CAMPAIGN_TERMINAL"
        or terminal_ledger_record.get("body") != terminal
    ):
        _fail(
            "FAIL_CAMPAIGN_LEDGER",
            "campaign terminal-to-completion link differs",
            terminal=True,
        )
    value: dict[str, object] = {
        "schema_version": "hegel-phase3a-q05b-shadow-retry-campaign-completion/2",
        "campaign_id": contract["contract_id"],
        "contract_digest": contract["contract_digest"],
        "campaign_source_binding_digest": contract["campaign_source_binding"][
            "source_binding_digest"
        ],
        "capsule_digest": capsule["capsule_digest"],
        "authorization_digest": authorization_digest,
        "terminal_record_name": "campaign.terminal.json",
        "terminal_body_digest": terminal_body_digest,
        "terminal_ledger_record_digest": terminal_ledger_digest,
        "terminal_ledger_sequence": terminal_sequence,
        "terminal_status": terminal["status"],
        "supervisor_invocation_count": terminal["supervisor_invocation_count"],
        "last_slot_id_or_null": terminal["last_slot_id_or_null"],
        "last_slot_outcome_digest_or_null": terminal[
            "last_slot_outcome_digest_or_null"
        ],
        "preceding_ledger_record_names_in_sequence_order": list(
            preceding_ledger_names
        ),
        "preceding_ledger_record_count": len(preceding_ledger_names),
        "completion_record_name": "campaign.completion.json",
        "completion_ledger_sequence": terminal_sequence + 1,
        "terminal_record_is_staged": True,
        "completion_is_authoritative_final_commit": True,
        "no_later_campaign_write_authorized": True,
        "record_digest": None,
    }
    return _record_digest(contract, "campaign_completion", value)


def _campaign_result_from_prepared_completion(
    contract: Mapping[str, object],
    capsule: Mapping[str, object],
    *,
    terminal: Mapping[str, object],
    terminal_ledger_record: Mapping[str, object],
    completion: Mapping[str, object],
    completion_ledger_record: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "hegel-phase3a-q05b-shadow-retry-campaign-result/2",
        "campaign_id": contract["contract_id"],
        "contract_digest": contract["contract_digest"],
        "capsule_digest": capsule["capsule_digest"],
        "status": terminal["status"],
        "supervisor_invocation_count": terminal["supervisor_invocation_count"],
        "last_slot_id_or_null": terminal["last_slot_id_or_null"],
        "last_slot_outcome_digest_or_null": terminal[
            "last_slot_outcome_digest_or_null"
        ],
        "error_code_or_null": terminal["error_code_or_null"],
        "error_detail_or_null": terminal["error_detail_or_null"],
        "error_receipt_or_null": deepcopy(terminal["error_receipt_or_null"]),
        "campaign_terminal_digest": terminal["record_digest"],
        "ledger_terminal_record_digest": terminal_ledger_record["record_digest"],
        "campaign_completion_digest": completion["record_digest"],
        "ledger_completion_record_digest": completion_ledger_record[
            "record_digest"
        ],
        "authoritative_completion_record_name": "campaign.completion.json",
        "completion_commit_verified": True,
        "ledger_root": contract["fixed_external_roots"]["ledger_root"],
        "shadow_only": True,
        "nonadopting": True,
    }


def _ledger_semantic_fail(detail: str) -> NoReturn:
    _fail(
        "FAIL_CAMPAIGN_LEDGER",
        f"ledger semantic state differs: {detail}",
        terminal=True,
    )


def _docker_inventory_argv_v2(contract: Mapping[str, object]) -> list[str]:
    execution = contract["execution_policy"]
    target = contract["target_commit_a_binding"]
    assert type(execution) is dict and type(target) is dict
    labels = execution["target_docker_label_policy"]
    assert type(labels) is dict
    commit = str(target["target_supervisor_commit"])
    return [
        str(execution["docker_executable"]),
        f"--host={execution['docker_host']}",
        "ps",
        "-a",
        "--no-trunc",
        "--filter",
        f"label={labels['source_commit_label_key']}={commit}",
        "--format",
        '{{.ID}}|{{.Label "org.hegel.q05b.execution_namespace"}}|'
        '{{.Label "org.hegel.q05b.slot"}}|'
        '{{.Label "org.hegel.q05b.source_commit"}}',
    ]


def _bound_command_slot_id_v2(
    contract: Mapping[str, object],
    label: object,
    argv: object,
) -> str | None:
    if type(label) is not str or type(argv) is not list:
        return None
    execution = contract.get("execution_policy")
    if type(execution) is not dict:
        return None
    git = execution.get("git_executable")
    for slot in _slot_registry(contract):
        slot_id = str(slot["slot_id"])
        supervisor_argv = _slot_argv(contract, slot)
        if argv == supervisor_argv and label in {
            f"{slot_id} bound supervisor",
            "bounded process wrapper resource closure",
        }:
            return slot_id
        clone = str(slot["clone_root"])
        if (
            argv
            and argv[0] == git
            and (
                any(
                    argv[index] == "-C" and argv[index + 1] == clone
                    for index in range(len(argv) - 1)
                )
                or (
                    label == "slot no-local clone"
                    and argv[-1] == clone
                )
            )
        ):
            return slot_id
    return None


def _bounded_command_binding_is_exact_v2(
    contract: Mapping[str, object],
    label: object,
    argv: object,
) -> bool:
    if (
        type(label) is not str
        or not label
        or len(label.encode("utf-8")) > 512
        or type(argv) is not list
        or not argv
        or len(argv) > 32
        or any(
            type(item) is not str
            or not item
            or "\x00" in item
            or len(item.encode("utf-8")) > 4096
            for item in argv
        )
    ):
        return False
    execution = contract["execution_policy"]
    target = contract["target_commit_a_binding"]
    assert type(execution) is dict and type(target) is dict
    git = str(execution["git_executable"])
    source = str(execution["clone_source_repository"])
    commit = str(target["target_supervisor_commit"])
    allowed: set[tuple[str, tuple[str, ...]]] = set()

    def allow(name: str, arguments: Sequence[str]) -> None:
        allowed.add((name, tuple([git, *arguments])))

    source_rows = (
        ("target Git common directory replay", ["-C", source, "rev-parse", "--git-common-dir"]),
        ("target Git worktree directory replay", ["-C", source, "rev-parse", "--absolute-git-dir"]),
        ("target Git object directory replay", ["-C", source, "rev-parse", "--git-path", "objects"]),
        ("target Git shallow-state replay", ["-C", source, "rev-parse", "--is-shallow-repository"]),
        ("target Git object-format replay", ["-C", source, "rev-parse", "--show-object-format"]),
        ("target Git local config replay", ["-C", source, "config", "--local", "--no-includes", "--null", "--list"]),
        ("target tree replay", ["-C", source, "rev-parse", f"{commit}^{{tree}}"]),
        ("target parent replay", ["-C", source, "rev-parse", f"{commit}^"]),
        ("target commit object replay", ["-C", source, "cat-file", "commit", commit]),
        ("target local object closure fsck", ["-C", source, "fsck", "--full", "--strict", "--no-dangling", "--no-reflogs", commit]),
        ("target tree special-entry replay", ["-C", source, "ls-tree", "-r", "-z", "--full-tree", commit]),
        ("controller HEAD replay", ["-C", source, "rev-parse", "HEAD"]),
        ("controller tree replay", ["-C", source, "rev-parse", "HEAD^{tree}"]),
        ("controller parent replay", ["-C", source, "rev-parse", "HEAD^"]),
        ("controller cleanliness replay", ["-C", source, "status", "--porcelain=v1", "-z", "--untracked-files=all"]),
        ("controller toplevel replay", ["-C", source, "rev-parse", "--show-toplevel"]),
    )
    for name, arguments in source_rows:
        allow(name, arguments)
    project_prefix = str(execution["clone_source_project_relative_path"])
    for relative in target["required_paths_in_exact_order"]:
        object_path = (PurePosixPath(project_prefix) / str(relative)).as_posix()
        allow(
            f"target object replay {relative}",
            ["-C", source, "show", f"{commit}:{object_path}"],
        )
    for slot in _slot_registry(contract):
        clone = str(slot["clone_root"])
        clone_rows = (
            ("slot clone HEAD replay", ["-C", clone, "rev-parse", "HEAD"]),
            ("slot clone tree replay", ["-C", clone, "rev-parse", "HEAD^{tree}"]),
            ("slot clone parent replay", ["-C", clone, "rev-parse", "HEAD^"]),
            ("slot clone tracked cleanliness replay", ["-C", clone, "status", "--porcelain=v1", "-z", "--untracked-files=no"]),
            ("slot clone untracked replay", ["-C", clone, "ls-files", "--others", "-z"]),
            ("slot clone remote replay", ["-C", clone, "remote"]),
            ("slot clone detached replay", ["-C", clone, "branch", "--show-current"]),
            ("slot clone ref replay", ["-C", clone, "for-each-ref", "--format=%(refname)%00%(objectname)"]),
            ("slot clone commit-graph verify", ["-C", clone, "commit-graph", "verify"]),
            ("slot clone graph-enabled strict fsck", ["-C", clone, "fsck", "--full", "--strict", "--unreachable", "--no-reflogs"]),
            ("slot clone graph-disabled strict fsck", ["-c", "core.commitGraph=false", "-C", clone, "fsck", "--full", "--strict", "--unreachable", "--no-reflogs"]),
            ("slot no-local clone", ["clone", "--no-checkout", "--no-hardlinks", "--no-tags", "--no-local", "--quiet", source, clone]),
            ("slot detached checkout", ["-C", clone, "checkout", "--detach", "--quiet", commit]),
            ("slot remote removal", ["-C", clone, "remote", "remove", "origin"]),
            ("slot cloned ref enumeration", ["-C", clone, "for-each-ref", "--format=%(refname)"]),
            ("slot clone reflog expiration", ["-C", clone, "reflog", "expire", "--expire=now", "--all"]),
            ("slot clone exact closure prune", ["-C", clone, "gc", "--prune=now", "--quiet"]),
            ("slot clone target graph ref creation", ["-C", clone, "update-ref", _CLONE_TARGET_REF_V2R8, commit]),
            ("slot clone reachable commit-graph write", ["-C", clone, "commit-graph", "write", "--reachable"]),
        )
        for name, arguments in clone_rows:
            allow(name, arguments)
        if label == "slot cloned ref deletion" and tuple(argv[:5]) == (
            git,
            "-C",
            clone,
            "update-ref",
            "-d",
        ):
            ref = argv[5] if len(argv) == 6 else None
            return (
                type(ref) is str
                and ref.startswith("refs/")
                and ".." not in ref
                and "\x00" not in ref
            )
        if label == "bounded process wrapper resource closure" and argv == _slot_argv(
            contract, slot
        ):
            return True
        if label == f"{slot['slot_id']} bound supervisor" and argv == _slot_argv(
            contract, slot
        ):
            return True
    if label == "Docker read-only inventory" and argv == _docker_inventory_argv_v2(
        contract
    ):
        return True
    return (label, tuple(argv)) in allowed


def _receipt_stream_semantics_v2(value: object) -> bool:
    if type(value) is not dict or set(value) != {
        "byte_count",
        "raw_sha256",
        "excerpt_hex",
        "excerpt_byte_count",
        "excerpt_truncated",
    }:
        return False
    raw_hex = value["excerpt_hex"]
    if (
        type(value["byte_count"]) is not int
        or not 0 <= value["byte_count"] <= _MAX_JSON_BYTES
        or type(value["raw_sha256"]) is not str
        or _RAW_SHA256.fullmatch(value["raw_sha256"]) is None
        or type(raw_hex) is not str
        or re.fullmatch(r"[0-9a-f]*", raw_hex) is None
        or len(raw_hex) > 2048
        or type(value["excerpt_byte_count"]) is not int
        or value["excerpt_byte_count"] != len(raw_hex) // 2
        or value["excerpt_byte_count"] != min(value["byte_count"], 1024)
        or type(value["excerpt_truncated"]) is not bool
        or value["excerpt_truncated"] is not (
            value["byte_count"] > value["excerpt_byte_count"]
        )
    ):
        return False
    return (
        value["raw_sha256"] == sha256(bytes.fromhex(raw_hex)).hexdigest()
        if value["excerpt_truncated"] is False
        else True
    )


def _error_snapshot_semantics_v2(
    contract: Mapping[str, object],
    value: object,
    *,
    depth: int,
    expected_active_slot_id_or_null: str | None = None,
    expected_command_argv_or_null: Sequence[str] | None = None,
) -> bool:
    return (
        type(value) is dict
        and set(value)
        == {
            "error_code",
            "error_detail",
            "error_terminal",
            "error_receipt_or_null",
        }
        and type(value["error_code"]) is str
        and 1 <= len(value["error_code"]) <= 256
        and type(value["error_detail"]) is str
        and 1 <= len(value["error_detail"]) <= 4096
        and type(value["error_terminal"]) is bool
        and _terminal_error_receipt_semantics_v2(
            contract,
            value["error_receipt_or_null"],
            terminal_error_code=value["error_code"],
            terminal_error_detail=value["error_detail"],
            top_level=False,
            depth=depth + 1,
            expected_active_slot_id_or_null=(
                expected_active_slot_id_or_null
            ),
            expected_command_argv_or_null=expected_command_argv_or_null,
        )
    )


def _terminal_error_receipt_semantics_v2(
    contract: Mapping[str, object],
    value: object,
    *,
    terminal_error_code: str,
    terminal_error_detail: str,
    top_level: bool,
    depth: int = 0,
    expected_prespawn_slot_id_or_null: str | None = None,
    expected_active_slot_id_or_null: str | None = None,
    expected_command_argv_or_null: Sequence[str] | None = None,
) -> bool:
    if value is None:
        return True
    if type(value) is not dict or depth > 3:
        return False
    try:
        if len(_canonical_json_bytes(value)) > _MAX_ERROR_RECEIPT_BYTES:
            return False
    except BaseException:
        return False
    schema = value.get("schema_version")
    registry = contract["terminal_failure_policy"].get(
        "authoritative_error_receipt_schema_registry_in_exact_order"
    )
    if registry != list(_AUTHORITATIVE_ERROR_RECEIPT_SCHEMAS_V2):
        return False
    if schema == _BOUNDED_RECEIPT_OMISSION_SCHEMA_V2:
        return set(value) == {"schema_version", "status"} and value.get(
            "status"
        ) == "RECEIPT_OMITTED_OVER_BOUND"
    if schema == _SCHEDULING_ADMISSION_EVIDENCE_SCHEMA_V6:
        expected_detail = _scheduling_admission_failure_detail_v6(value)
        return (
            top_level
            and terminal_error_code == _SCHEDULING_ADMISSION_FAILURE_V2
            and type(expected_prespawn_slot_id_or_null) is str
            and value.get("slot_id") == expected_prespawn_slot_id_or_null
            and expected_detail is not None
            and terminal_error_detail == expected_detail[:4096]
            and _scheduling_admission_evidence_semantics_v6(
                contract, value, require_passed=False
            )
        )
    if schema == _PUBLICATION_RECEIPT_SCHEMA_V2:
        if set(value) != {
            "schema_version",
            "publication_outcome",
            "publication_label",
            "publication_basename",
        }:
            return False
        label = value["publication_label"]
        basename = value["publication_basename"]
        outcome = value["publication_outcome"]
        allowed_top_level_pairs = {
            ("slot-supervisor-stdout", "supervisor.stdout"): "FAIL_CAMPAIGN_EVIDENCE",
            ("slot-supervisor-stderr", "supervisor.stderr"): "FAIL_CAMPAIGN_EVIDENCE",
            ("slot-external-trajectory-binding", "trajectory-binding.json"): "FAIL_CAMPAIGN_BINDING",
            ("slot-outcome", "slot-result.json"): "FAIL_CAMPAIGN_OBSERVATION",
        }
        return (
            type(label) is str
            and _SAFE_NAME.fullmatch(label) is not None
            and type(basename) is str
            and _SAFE_NAME.fullmatch(basename) is not None
            and outcome == "KNOWN_NO_PUBLICATION"
            and (
                not top_level
                or allowed_top_level_pairs.get((label, basename))
                == terminal_error_code
            )
            and f"publication_label={label}; basename={basename}"
            in terminal_error_detail
            and terminal_error_code
            not in {
                "FAIL_CAMPAIGN_LEDGER",
                "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN",
            }
        )
    diagnostic_schema = _BOUNDED_COMMAND_DIAGNOSTIC_SCHEMA_V2
    exception_schema = _BOUNDED_COMMAND_EXCEPTION_DIAGNOSTIC_SCHEMA_V2
    if schema == diagnostic_schema:
        if set(value) != {
            "schema_version",
            "label",
            "argv",
            "return_code",
            "timed_out",
            "stdout_overflow",
            "stderr_overflow",
            "child_reaped",
            "process_group_absent",
            "wrapper_resources_closed",
            "stdout",
            "stderr",
            "classification",
        } or not _bounded_command_binding_is_exact_v2(
            contract, value["label"], value["argv"]
        ):
            return False
        classification = value["classification"]
        if type(classification) is not dict or classification.get(
            "accepted"
        ) is not False:
            return False
        classification_fields = set(classification)
        if classification_fields == {
            "require_empty_stdout",
            "require_empty_stderr",
            "accepted",
        }:
            classification_ok = all(
                type(classification[key]) is bool
                for key in ("require_empty_stdout", "require_empty_stderr")
            )
        elif classification_fields == {"accepted", "reason"}:
            classification_ok = (
                type(classification["reason"]) is str
                and bool(classification["reason"])
            )
        elif classification_fields == {
            "accepted",
            "close_errors",
            "prior_error_or_null",
        }:
            classification_ok = (
                type(classification["close_errors"]) is list
                and bool(classification["close_errors"])
                and all(
                    type(item) is str and bool(item) and len(item) <= 4096
                    for item in classification["close_errors"]
                )
                and (
                    classification["prior_error_or_null"] is None
                    or (
                        type(classification["prior_error_or_null"]) is str
                        and bool(classification["prior_error_or_null"])
                    )
                )
            )
        else:
            classification_ok = False
        certain_git_source = _certain_registered_git_failure_diagnostic_v2(
            contract,
            value,
            terminal_error_detail,
        )
        terminal_code_ok = (
            terminal_error_code == "FAIL_CAMPAIGN_SOURCE"
            and certain_git_source
        ) or (
            terminal_error_code == "FAIL_CAMPAIGN_PROCESS"
            and not certain_git_source
        )
        command_slot_id = _bound_command_slot_id_v2(
            contract,
            value["label"],
            value["argv"],
        )
        if (
            not classification_ok
            or type(value["return_code"]) not in (int, type(None))
            or any(
                type(value[field]) is not bool
                for field in (
                    "timed_out",
                    "stdout_overflow",
                    "stderr_overflow",
                    "child_reaped",
                    "process_group_absent",
                    "wrapper_resources_closed",
                )
            )
            or not _receipt_stream_semantics_v2(value["stdout"])
            or not _receipt_stream_semantics_v2(value["stderr"])
            or not terminal_code_ok
            or (
                expected_command_argv_or_null is not None
                and value["argv"] != list(expected_command_argv_or_null)
            )
            or (
                command_slot_id is not None
                and command_slot_id != expected_active_slot_id_or_null
            )
            or (
                top_level
                and value["argv"][0]
                != contract["execution_policy"]["git_executable"]
            )
        ):
            return False
        if classification_fields == {
            "accepted",
            "close_errors",
            "prior_error_or_null",
        }:
            return (
                value["label"] == "bounded process wrapper resource closure"
                and terminal_error_detail
                == "wrapper resource closure failed: "
                + ";".join(classification["close_errors"])
            )
        return terminal_error_detail == (
            f"{value['label']} failed closed: "
            f"return_code={value['return_code']}; "
            f"timed_out={value['timed_out']}; "
            f"stdout_overflow={value['stdout_overflow']}; "
            f"stderr_overflow={value['stderr_overflow']}; "
            f"stderr_excerpt_hex={value['stderr']['excerpt_hex']}"
        )
    if schema == exception_schema:
        if set(value) != {
            "schema_version",
            "label",
            "argv",
            "phase",
            "error_code",
            "error_detail",
            "return_code",
            "timed_out",
            "stdout_overflow",
            "stderr_overflow",
            "child_reaped",
            "process_group_absent",
            "wrapper_resources_closed",
            "stdout",
            "stderr",
            "prior_receipt_or_null",
        } or not _bounded_command_binding_is_exact_v2(
            contract, value["label"], value["argv"]
        ):
            return False
        inner_code = value["error_code"]
        inner_detail = value["error_detail"]
        command_slot_id = _bound_command_slot_id_v2(
            contract,
            value["label"],
            value["argv"],
        )
        return (
            value["phase"] == "BOUNDED_PROCESS_RAISED_BEFORE_RESULT"
            and type(inner_code) is str
            and 1 <= len(inner_code) <= 256
            and type(inner_detail) is str
            and 1 <= len(inner_detail) <= 4096
            and all(
                value[field] is None
                for field in (
                    "return_code",
                    "timed_out",
                    "stdout_overflow",
                    "stderr_overflow",
                    "child_reaped",
                    "process_group_absent",
                    "wrapper_resources_closed",
                )
            )
            and value["stdout"] == {"available": False}
            and value["stderr"] == {"available": False}
            and _terminal_error_receipt_semantics_v2(
                contract,
                value["prior_receipt_or_null"],
                terminal_error_code=inner_code,
                terminal_error_detail=inner_detail,
                top_level=False,
                depth=depth + 1,
                expected_active_slot_id_or_null=(
                    expected_active_slot_id_or_null
                ),
                expected_command_argv_or_null=value["argv"],
            )
            and terminal_error_code in {inner_code, "FAIL_CAMPAIGN_PROCESS"}
            and (
                expected_command_argv_or_null is None
                or value["argv"] == list(expected_command_argv_or_null)
            )
            and (
                command_slot_id is None
                or command_slot_id == expected_active_slot_id_or_null
            )
            and (
                not top_level
                or value["label"].endswith(" bound supervisor")
                or value["argv"][0]
                == contract["execution_policy"]["git_executable"]
            )
            and terminal_error_detail
            == f"{value['label']} bounded process raised before a result: "
            f"{inner_code}:{inner_detail}"
        )
    if schema == _PRESPAWN_DOCKER_RECEIPT_SCHEMA_V2:
        expected_fields = {
            "schema_version",
            "docker_rows",
            "docker_error_or_null",
        }
        if value.get("docker_error_or_null") is not None:
            expected_fields.add("bounded_command_diagnostic_or_null")
        rows = value.get("docker_rows")
        error = value.get("docker_error_or_null")
        labels = contract["execution_policy"]["target_docker_label_policy"]
        assert type(labels) is dict
        allowed_docker_slots = labels[
            "supervisor_slot_label_values_in_exact_order"
        ]
        if (
            set(value) != expected_fields
            or terminal_error_code != "FAIL_CAMPAIGN_DOCKER"
            or type(rows) is not list
            or any(
                type(row) is not dict
                or set(row)
                != {
                    "container_id",
                    "execution_namespace",
                    "slot",
                    "source_commit",
                }
                or any(type(item) is not str for item in row.values())
                or _RAW_SHA256.fullmatch(str(row["container_id"])) is None
                or _RAW_SHA256.fullmatch(str(row["execution_namespace"])) is None
                or row["slot"] not in allowed_docker_slots
                or row["source_commit"]
                != contract["target_commit_a_binding"]["target_supervisor_commit"]
                for row in rows
            )
            or (error is not None and (type(error) is not str or not error))
            or (not rows and error is None)
            or len({row["container_id"] for row in rows}) != len(rows)
        ):
            return False
        nested = value.get("bounded_command_diagnostic_or_null")
        if error is not None:
            try:
                error_bytes = error.encode("ascii") + b"\n"
            except BaseException:
                return False
            if (
                type(nested) is dict
                and nested.get("schema_version")
                == _BOUNDED_RECEIPT_OMISSION_SCHEMA_V2
            ):
                return (
                    len(error_bytes) > 16 * 1024
                    and _terminal_error_receipt_semantics_v2(
                        contract,
                        nested,
                        terminal_error_code="FAIL_CAMPAIGN_PROCESS",
                        terminal_error_detail="bounded Docker diagnostic omitted",
                        top_level=False,
                        depth=depth + 1,
                        expected_active_slot_id_or_null=(
                            expected_active_slot_id_or_null
                        ),
                    )
                    and type(expected_prespawn_slot_id_or_null) is str
                    and terminal_error_detail
                    == (
                        f"{expected_prespawn_slot_id_or_null} pre-spawn Docker "
                        f"absence is uncertain: {error}"
                    )[:4096]
                )
            try:
                decoded = _strict_json(
                    error_bytes,
                    "authoritative pre-spawn Docker diagnostic",
                    16 * 1024,
                )
            except BaseException:
                return False
            if decoded != nested or not _terminal_error_receipt_semantics_v2(
                contract,
                nested,
                terminal_error_code="FAIL_CAMPAIGN_PROCESS",
                terminal_error_detail=(
                    f"{nested['label']} failed closed: "
                    f"return_code={nested['return_code']}; "
                    f"timed_out={nested['timed_out']}; "
                    f"stdout_overflow={nested['stdout_overflow']}; "
                    f"stderr_overflow={nested['stderr_overflow']}; "
                    f"stderr_excerpt_hex={nested['stderr']['excerpt_hex']}"
                    if type(nested) is dict
                    and nested.get("schema_version") == diagnostic_schema
                    else (
                        f"{nested['label']} bounded process raised before a result: "
                        f"{nested['error_code']}:{nested['error_detail']}"
                        if type(nested) is dict
                        and nested.get("schema_version") == exception_schema
                        else "invalid"
                    )
                ),
                top_level=False,
                depth=depth + 1,
                expected_active_slot_id_or_null=(
                    expected_active_slot_id_or_null
                ),
            ):
                return False
        return (
            type(expected_prespawn_slot_id_or_null) is str
            and terminal_error_detail
            == (
                f"{expected_prespawn_slot_id_or_null} pre-spawn Docker "
                f"absence is uncertain: {error or rows}"
            )[:4096]
        )
    if schema == _SPAWN_REGISTRATION_SCHEMA_V2:
        short_fields = {
            "schema_version",
            "phase",
            "registration_state",
            "later_ledger_write_permitted",
        }
        if set(value) == short_fields:
            return (
                not top_level
                and value["phase"] == "SPAWN_REGISTRATION"
                and value["registration_state"] == "PROCESS_OBJECT_MISSING"
                and value["later_ledger_write_permitted"] is False
            )
        fields = {
            "schema_version",
            "phase",
            "argv",
            "registration_state",
            "registration_raw_hex",
            "registered_pid_or_null",
            "popen_pid_or_null",
            "popen_child_created",
            "pid_match_or_null",
            "constructor_error_or_null",
            "cleanup_target_pid_or_null",
            "cleanup_kill_attempted",
            "cleanup_direct_child_reaped",
            "cleanup_process_group_absent",
            "cleanup_identity_consistent",
            "cleanup_errors",
            "cleanup_certain",
            "later_ledger_write_permitted",
        }
        if set(value) != fields or value["phase"] != "SPAWN_REGISTRATION":
            return False
        argv_ok = any(
            value["argv"] == _slot_argv(contract, slot)
            for slot in _slot_registry(contract)
        )
        registered = value["registered_pid_or_null"]
        popen = value["popen_pid_or_null"]
        target_pid = value["cleanup_target_pid_or_null"]
        raw_hex = value["registration_raw_hex"]
        state = value["registration_state"]
        cleanup_errors = value["cleanup_errors"]
        constructor_error = value["constructor_error_or_null"]
        if (
            not argv_ok
            or state not in {"VALID", "MISSING", "TORN", "MALFORMED", "READ_ERROR"}
            or type(raw_hex) is not str
            or re.fullmatch(r"[0-9a-f]*", raw_hex) is None
            or len(raw_hex) > 2 * (_SPAWN_RECEIPT_MAX_BYTES + 1)
            or any(
                item is not None
                and (type(item) is not int or not 2 <= item <= 2**31 - 1)
                for item in (registered, popen, target_pid)
            )
            or type(value["popen_child_created"]) is not bool
            or value["pid_match_or_null"]
            is not (
                popen == registered
                if popen is not None and registered is not None
                else None
            )
            or type(cleanup_errors) is not list
            or any(type(item) is not str or not item for item in cleanup_errors)
            or any(
                type(value[field]) is not bool
                for field in (
                    "cleanup_kill_attempted",
                    "cleanup_direct_child_reaped",
                    "cleanup_process_group_absent",
                    "cleanup_identity_consistent",
                    "cleanup_certain",
                    "later_ledger_write_permitted",
                )
            )
            or (
                constructor_error is not None
                and not _error_snapshot_semantics_v2(
                    contract,
                    constructor_error,
                    depth=depth,
                    expected_active_slot_id_or_null=(
                        expected_active_slot_id_or_null
                    ),
                    expected_command_argv_or_null=value["argv"],
                )
            )
            or (
                expected_command_argv_or_null is not None
                and value["argv"] != list(expected_command_argv_or_null)
            )
        ):
            return False
        cleanup_certain = (
            target_pid is not None
            and value["cleanup_direct_child_reaped"] is True
            and value["cleanup_process_group_absent"] is True
            and value["cleanup_identity_consistent"] is True
            and cleanup_errors == []
        )
        later = (
            state == "VALID"
            and value["pid_match_or_null"] is not False
            and cleanup_certain
        )
        if (
            value["cleanup_certain"] is not cleanup_certain
            or value["later_ledger_write_permitted"] is not later
            or target_pid != (popen if popen is not None else registered)
            or value["cleanup_kill_attempted"] is not (target_pid is not None)
            or value["cleanup_identity_consistent"]
            is not (value["pid_match_or_null"] is not False)
            or (
                state == "VALID"
                and (
                    registered is None
                    or raw_hex != (str(registered) + "\n").encode("ascii").hex()
                )
            )
            or (state == "MISSING" and raw_hex != "")
        ):
            return False
        return not top_level
    return False


def _verify_domain_root_v2(
    value: object,
    *,
    domain: str,
    root_field: str,
    preimage_field: str | None,
    label: str,
) -> dict[str, object]:
    if type(value) is not dict:
        _ledger_semantic_fail(f"{label} is not an object")
    observed = value.get(root_field)
    if type(observed) is not str or _TAGGED_SHA256.fullmatch(observed) is None:
        _ledger_semantic_fail(f"{label} root differs")
    if preimage_field is None:
        unsigned = deepcopy(value)
        unsigned.pop(root_field, None)
    else:
        unsigned = value.get(preimage_field)
    if observed != _domain_digest(domain, unsigned):
        _ledger_semantic_fail(f"{label} root digest differs")
    return value


def _verify_slot_absence_evidence_semantics_v2(
    capsule: Mapping[str, object],
    value: object,
    *,
    expected_label: str,
    expected_slots: Sequence[Mapping[str, object]],
    require_certain: bool,
) -> dict[str, object]:
    evidence = _verify_domain_root_v2(
        value,
        domain=(
            "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R8/"
            "SLOT_ROOT_ABSENCE/V2"
        ),
        root_field="root",
        preimage_field=None,
        label="slot absence evidence",
    )
    if set(evidence) != {"label", "rows", "error_or_null", "certain", "root"}:
        _ledger_semantic_fail("slot absence evidence fields differ")
    rows = evidence.get("rows")
    certain = evidence.get("certain")
    error_or_null = evidence.get("error_or_null")
    if (
        evidence.get("label") != expected_label
        or type(certain) is not bool
        or type(rows) is not list
        or (certain and error_or_null is not None)
        or (
            not certain
            and (type(error_or_null) is not str or not error_or_null)
        )
        or (require_certain and certain is not True)
    ):
        _ledger_semantic_fail("slot absence evidence disposition differs")
    expected_targets = [
        (slot["slot_id"], field, os.fspath(Path(str(slot[field]))))
        for slot in expected_slots
        for field in (
            "clone_root",
            "cargo_cache_root",
            "artifact_parent_root",
            "evidence_root",
            "trace_root",
        )
    ]
    if (
        (certain and len(rows) != len(expected_targets))
        or (not certain and len(rows) > len(expected_targets))
    ):
        _ledger_semantic_fail("slot absence row count differs")
    parent_registry = capsule.get("freshness_parent_identities")
    for row, (slot_id, field, target_path) in zip(rows, expected_targets):
        if type(row) is not dict or set(row) != {
            "slot_id",
            "field",
            "target_path",
            "state_parent_path",
            "expected_state_parent_stable_identity",
            "observed_state_parent_identity",
            "observed_state_parent_stable_identity",
            "final_state_parent_identity",
            "final_state_parent_stable_identity",
            "target_present_identity_or_null",
            "absent_and_anchored",
        }:
            _ledger_semantic_fail("slot absence row fields differ")
        parent_path = os.fspath(Path(target_path).parent)
        expected_parent = (
            parent_registry.get(parent_path)
            if type(parent_registry) is dict
            else row.get("expected_state_parent_stable_identity")
        )
        if (
            row.get("slot_id") != slot_id
            or row.get("field") != field
            or row.get("target_path") != target_path
            or row.get("state_parent_path") != parent_path
        ):
            _ledger_semantic_fail("slot absence row mapping differs")
        if certain and (
            type(expected_parent) is not dict
            or row.get("expected_state_parent_stable_identity")
            != expected_parent
            or row.get("observed_state_parent_stable_identity")
            != expected_parent
            or row.get("final_state_parent_stable_identity")
            != expected_parent
            or row.get("target_present_identity_or_null") is not None
            or row.get("absent_and_anchored") is not True
        ):
            _ledger_semantic_fail("slot absence anchoring differs")
    return evidence


def _trajectory_from_binding_v2(
    binding: Mapping[str, object],
) -> dict[str, object]:
    return {
        "verification_status": binding["verification_status"],
        "complete": binding["verification_status"]
        in {"COMPLETE_OK", "COMPLETE_FAIL"},
        "complete_verification_passed": binding[
            "complete_verification_passed"
        ],
        "verification_error_or_null": binding["verification_error_or_null"],
        "trace_id_or_null": binding["trace_id"],
        "producer_id_or_null": binding["producer_id"],
        "event_count": binding["event_count"],
        "first_event_record_digest_or_null": binding[
            "first_event_record_digest"
        ],
        "final_event_record_digest_or_null": binding[
            "final_event_record_digest"
        ],
        "terminal_outcome_or_null": binding["terminal_manifest_outcome"],
        "terminal_manifest_digest_or_null": binding[
            "terminal_manifest_digest"
        ],
        "terminal_manifest_raw_sha256_or_null": binding[
            "terminal_manifest_raw_sha256"
        ],
        "clock_id_or_null": binding["clock_id"],
        "closed_span_count_or_null": binding["closed_span_count"],
        "terminal_event_type_or_null": binding["terminal_event_type"],
        "terminal_error_code_or_null": binding[
            "terminal_error_code_or_null"
        ],
        "terminal_error_detail_or_null": binding[
            "terminal_error_detail_or_null"
        ],
        "torn_tail_staging_basename_or_null": binding[
            "torn_tail_staging_basename_or_null"
        ],
        "open_span_ids_in_begin_order": deepcopy(
            binding["open_span_ids_in_begin_order"]
        ),
        "open_span_contexts_in_begin_order": deepcopy(
            binding["open_span_contexts_in_begin_order"]
        ),
        "failure_rows_retained_in_sequence_order": deepcopy(
            binding["failure_rows_retained_in_sequence_order"]
        ),
        "cleanup_rows_retained_in_sequence_order": deepcopy(
            binding["cleanup_rows_retained_in_sequence_order"]
        ),
        "partition_certificate_or_null": deepcopy(
            binding["partition_certificate_or_null"]
        ),
        "last_durable_event_row_or_null": deepcopy(
            binding["last_durable_event_row_or_null"]
        ),
        "canonical_event_payloads_digest": binding[
            "canonical_event_payloads_digest"
        ],
        "success_trace_bindings_or_null": deepcopy(
            binding["success_trace_bindings_or_null"]
        ),
    }


def _verify_binding_observation_crosslinks_v2(
    contract: Mapping[str, object],
    slot: Mapping[str, object],
    binding: Mapping[str, object],
    observation: Mapping[str, object],
) -> None:
    trajectory = _trajectory_from_binding_v2(binding)
    trajectory_complete = _trajectory_complete_and_bound(
        contract,
        slot,
        trajectory,
    )
    before = binding["source_replay_before_evidence"]
    after = binding["source_replay_after_evidence"]
    future = binding["future_slot_freshness_evidence"]
    docker = binding["docker_inventory_evidence"]
    supervisor_artifact = binding["supervisor_artifact_evidence"]
    external_artifact = binding["external_artifact_evidence"]
    if not all(
        type(value) is dict
        for value in (
            before,
            after,
            future,
            docker,
            supervisor_artifact,
            external_artifact,
        )
    ):
        _ledger_semantic_fail("binding evidence object fields differ")
    artifact_state, artifact_validation, artifact_replay = (
        _artifact_observation_facts(supervisor_artifact, external_artifact)
    )
    source_certain = (
        before.get("certain") is True
        and after.get("certain") is True
        and future.get("certain") is True
    )
    docker_certain = (
        docker.get("absent") is True
        and docker.get("rows") == []
        and docker.get("error_or_null") is None
    )
    cleanup_certain = (
        binding["process_child_reaped"] is True
        and binding["process_group_absent"] is True
        and binding["process_wrapper_resources_closed"] is True
        and binding["process_timed_out"] is False
        and docker_certain
        and trajectory_complete
    )
    maximum_capture = int(_load_core(contract).CAPTURE_MAX_BYTES_V2)
    expected_pairs = {
        "campaign_id": contract["contract_id"],
        "slot_number": slot["slot_ordinal"],
        "expected_trace_id": slot["trajectory_trace_id"],
        "expected_producer_id": contract["trajectory_binding_policy"][
            "producer_id"
        ],
        "invocation_count": 1,
        "supervisor_returned": type(binding["process_return_code"]) is int,
        "supervisor_reaped": binding["process_child_reaped"],
        "process_group_absent": binding["process_group_absent"],
        "exit_code_or_null": binding["process_return_code"],
        "stdout_capture_complete": (
            binding["process_stdout_overflow"] is False
            and binding["process_timed_out"] is False
        ),
        "stdout_capture_bounded": (
            binding["process_stdout_overflow"] is False
            and type(binding["stdout_byte_count"]) is int
            and 0 <= binding["stdout_byte_count"] <= maximum_capture
        ),
        "stdout_raw_byte_count": binding["stdout_byte_count"],
        "stdout_raw_sha256": binding["stdout_raw_sha256"],
        "stderr_capture_complete": (
            binding["process_stderr_overflow"] is False
            and binding["process_timed_out"] is False
        ),
        "stderr_capture_bounded": (
            binding["process_stderr_overflow"] is False
            and type(binding["stderr_byte_count"]) is int
            and 0 <= binding["stderr_byte_count"] <= maximum_capture
        ),
        "stderr_raw_byte_count": binding["stderr_byte_count"],
        "stderr_raw_sha256": binding["stderr_raw_sha256"],
        "stderr_empty": binding["stderr_byte_count"] == 0,
        "trajectory_status": binding["verification_status"],
        "trajectory_trace_id": binding["trace_id"],
        "trajectory_producer_id": binding["producer_id"],
        "trajectory_torn_tail_or_null": binding[
            "torn_tail_staging_basename_or_null"
        ],
        "trajectory_open_span_ids": binding[
            "open_span_ids_in_begin_order"
        ],
        "trajectory_terminal_error_code_or_null": binding[
            "terminal_error_code_or_null"
        ],
        "trajectory_terminal_error_detail_or_null": binding[
            "terminal_error_detail_or_null"
        ],
        "trajectory_terminal_event_type_or_null": binding[
            "terminal_event_type"
        ],
        "trajectory_event_count": binding["event_count"],
        "trajectory_first_event_record_digest": binding[
            "first_event_record_digest"
        ],
        "trajectory_final_event_record_digest": binding[
            "final_event_record_digest"
        ],
        "trajectory_canonical_event_payloads_digest": binding[
            "canonical_event_payloads_digest"
        ],
        "trajectory_partition_certificate_or_null": binding[
            "partition_certificate_or_null"
        ],
        "trajectory_failure_rows": binding[
            "failure_rows_retained_in_sequence_order"
        ],
        "trajectory_cleanup_rows": binding[
            "cleanup_rows_retained_in_sequence_order"
        ],
        "artifact_state": artifact_state,
        "artifact_validation_status": artifact_validation,
        "artifact_replay_status": artifact_replay,
        "cleanup_certain": cleanup_certain,
        "docker_absence_certain": docker_certain,
        "clone_clean_certain": source_certain,
        "cargo_isolation_certain": source_certain,
        "source_binding_certain": source_certain,
        "external_binding_certain": True,
        "replay_certain": trajectory_complete and source_certain,
    }
    if any(observation.get(key) != value for key, value in expected_pairs.items()):
        _ledger_semantic_fail("binding-to-observation cross-link differs")


def _verify_ledger_state_semantics_v2(
    contract: Mapping[str, object],
    capsule: Mapping[str, object],
    authorization: Mapping[str, object],
    names: Sequence[str],
    records: Sequence[Mapping[str, object]],
) -> None:
    """Validate the exact finite ledger automaton, not only its hashes."""

    prefix_names = list(names[:-2])
    legal_prefixes = [
        list(_LEDGER_RECORD_NAMES_V2[:2]),
        list(_LEDGER_RECORD_NAMES_V2[:3]),
        list(_LEDGER_RECORD_NAMES_V2[:4]),
        list(_LEDGER_RECORD_NAMES_V2[:6]),
        list(_LEDGER_RECORD_NAMES_V2[:7]),
        list(_LEDGER_RECORD_NAMES_V2[:8]),
        list(_LEDGER_RECORD_NAMES_V2[:10]),
        list(_LEDGER_RECORD_NAMES_V2[:11]),
        list(_LEDGER_RECORD_NAMES_V2[:12]),
        list(_LEDGER_RECORD_NAMES_V2[:14]),
    ]
    if prefix_names not in legal_prefixes:
        _ledger_semantic_fail("terminal prefix is not one of ten legal states")
    by_name = {
        name: record
        for name, record in zip(names, records)
    }
    if (
        by_name["campaign.intent.json"]["status"]
        != "AUTHORIZED_CAMPAIGN_INTENT_STAGED"
        or by_name["campaign.intent.json"]["body"]
        != {
            "capsule": deepcopy(dict(capsule)),
            "authorization_digest": authorization["authorization_digest"],
        }
        or by_name["campaign.authorization-consumed.json"]["status"]
        != "CAMPAIGN_AUTHORIZATION_CONSUMED"
        or by_name["campaign.authorization-consumed.json"]["body"]
        != {
            "authorization": deepcopy(dict(authorization)),
            "consumed_once": True,
        }
        or authorization != _authorization_template(contract, capsule)
    ):
        _ledger_semantic_fail("campaign intent or authorization differs")

    slots = _slot_registry(contract)
    full_outcomes: list[tuple[Mapping[str, object], Mapping[str, object]]] = []
    consumed_count = 0
    for ordinal, slot in enumerate(slots, start=1):
        stem = f"slot-{ordinal:02d}"
        intent_name = f"{stem}.intent.json"
        consumed_name = f"{stem}.consumed.json"
        binding_name = f"{stem}.trajectory-binding.json"
        outcome_name = f"{stem}.outcome.json"
        if intent_name not in by_name:
            continue
        intent_record = by_name[intent_name]
        intent_body = intent_record["body"]
        if type(intent_body) is not dict or set(intent_body) != {
            "slot",
            "argv",
            "environment",
            "previous_slot_id_or_null",
            "slot_freshness_before_intent",
        }:
            _ledger_semantic_fail(f"{slot['slot_id']} intent fields differ")
        _verify_slot_absence_evidence_semantics_v2(
            capsule,
            intent_body["slot_freshness_before_intent"],
            expected_label=f"{slot['slot_id']} before intent",
            expected_slots=[slot],
            require_certain=True,
        )
        if (
            intent_record["status"] != f"{slot['slot_id']}_INTENT_STAGED"
            or intent_body["slot"] != slot
            or intent_body["argv"] != _slot_argv(contract, slot)
            or intent_body["environment"]
            != _runtime_environment(
                contract,
                git_worktree=str(slot["clone_root"]),
            )
            or intent_body["previous_slot_id_or_null"]
            != slot["previous_slot_id"]
        ):
            _ledger_semantic_fail(f"{slot['slot_id']} intent mapping differs")
        if consumed_name not in by_name:
            continue
        consumed_count += 1
        consumed_record = by_name[consumed_name]
        consumed_body = consumed_record["body"]
        if (
            type(consumed_body) is not dict
            or set(consumed_body) != {
                "slot_id",
                "slot_ordinal",
                "argv",
                "source_replay_before_root",
                "docker_absence_before_spawn",
                "scheduling_admission",
            }
            or consumed_record["status"]
            != f"{slot['slot_id']}_CONSUMED_OUTCOME_UNKNOWN"
            or consumed_body["slot_id"] != slot["slot_id"]
            or consumed_body["slot_ordinal"] != ordinal
            or consumed_body["argv"] != _slot_argv(contract, slot)
            or type(consumed_body["source_replay_before_root"]) is not str
            or _TAGGED_SHA256.fullmatch(
                consumed_body["source_replay_before_root"]
            )
            is None
            or consumed_body["docker_absence_before_spawn"] is not True
            or not _scheduling_admission_evidence_exact_v2(
                contract, consumed_body["scheduling_admission"]
            )
            or consumed_body["scheduling_admission"].get("slot_id")
            != consumed_body["slot_id"]
        ):
            _ledger_semantic_fail(f"{slot['slot_id']} consumed mapping differs")
        if binding_name not in by_name:
            continue
        binding_record = by_name[binding_name]
        binding = binding_record["body"]
        expected_binding_fields = contract["trajectory_binding_policy"][
            "external_binding_exact_fields"
        ]
        if (
            type(binding) is not dict
            or type(expected_binding_fields) is not list
            or set(binding) != set(expected_binding_fields)
        ):
            _ledger_semantic_fail(f"{slot['slot_id']} binding fields differ")
        binding_unsigned = deepcopy(binding)
        binding_digest = binding_unsigned.pop("binding_digest", None)
        target = contract["target_commit_a_binding"]
        required_hashes = target["required_raw_sha256_by_path"]
        if (
            binding_record["status"] != binding["verification_status"]
            or binding["schema_version"]
            != "hegel-phase3a-q05b-shadow-retry-external-trajectory-binding/2"
            or binding["campaign_id"] != contract["contract_id"]
            or binding["contract_digest"] != contract["contract_digest"]
            or binding["capsule_digest"] != capsule["capsule_digest"]
            or binding["slot_id"] != slot["slot_id"]
            or binding["slot_ordinal"] != ordinal
            or binding["target_supervisor_commit"]
            != target["target_supervisor_commit"]
            or binding["target_supervisor_tool_raw_sha256"]
            != required_hashes["tools/phase3_q05b_dual_qualification_v2.py"]
            or binding["trajectory_core_raw_sha256"]
            != required_hashes["src/hegel_machine/phase3_q05b_trajectory_v2.py"]
            or binding["trace_root_path"] != slot["trace_root"]
            or binding["trace_id"] != slot["trajectory_trace_id"]
            or binding["producer_id"]
            != contract["trajectory_binding_policy"]["producer_id"]
            or binding["source_replay_before_root"]
            != consumed_body["source_replay_before_root"]
            or binding_digest
            != _domain_digest(
                contract["hash_domains"]["trajectory_binding"],
                binding_unsigned,
            )
        ):
            _ledger_semantic_fail(f"{slot['slot_id']} binding mapping differs")
        before = _verify_domain_root_v2(
            binding["source_replay_before_evidence"],
            domain=(
                "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R8/"
                "SOURCE_REPLAY/V2"
            ),
            root_field="root",
            preimage_field="evidence",
            label="source replay before",
        )
        after = _verify_domain_root_v2(
            binding["source_replay_after_evidence"],
            domain=(
                "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R8/"
                "SOURCE_REPLAY/V2"
            ),
            root_field="root",
            preimage_field="evidence",
            label="source replay after",
        )
        if (
            set(before) != {"certain", "error_or_null", "evidence", "root"}
            or set(after) != {"certain", "error_or_null", "evidence", "root"}
            or binding["source_replay_before_root"] != before["root"]
            or binding["source_replay_after_root"] != after["root"]
        ):
            _ledger_semantic_fail(f"{slot['slot_id']} source roots differ")
        future_slots = slots[ordinal:]
        future = _verify_slot_absence_evidence_semantics_v2(
            capsule,
            binding["future_slot_freshness_evidence"],
            expected_label=f"{slot['slot_id']} future slots before outcome",
            expected_slots=future_slots,
            require_certain=False,
        )
        cleanup = _verify_domain_root_v2(
            binding["cleanup_replay_evidence"],
            domain=(
                "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R8/"
                "CLEANUP_REPLAY/V2"
            ),
            root_field="root",
            preimage_field="preimage",
            label="cleanup replay",
        )
        docker_evidence = binding["docker_inventory_evidence"]
        if (
            type(docker_evidence) is not dict
            or set(docker_evidence) != {"absent", "rows", "error_or_null"}
        ):
            _ledger_semantic_fail(
                f"{slot['slot_id']} Docker evidence differs"
            )
        expected_cleanup_preimage = {
            "process": {
                "return_code": binding["process_return_code"],
                "stdout_overflow": binding["process_stdout_overflow"],
                "stderr_overflow": binding["process_stderr_overflow"],
                "timed_out": binding["process_timed_out"],
                "child_reaped": binding["process_child_reaped"],
                "process_group_absent": binding["process_group_absent"],
                "wrapper_resources_closed": binding[
                    "process_wrapper_resources_closed"
                ],
            },
            "trajectory_status": binding["verification_status"],
            "trajectory_terminal_manifest_digest_or_null": binding[
                "terminal_manifest_digest"
            ],
            "trajectory_open_span_ids": binding[
                "open_span_ids_in_begin_order"
            ],
            "docker_absent": docker_evidence["absent"],
            "docker_rows": docker_evidence["rows"],
            "docker_error_or_null": docker_evidence["error_or_null"],
        }
        if (
            set(cleanup) != {"preimage", "root"}
            or cleanup["preimage"] != expected_cleanup_preimage
            or binding["cleanup_replay_root"] != cleanup["root"]
            or type(future) is not dict
            or type(binding["supervisor_artifact_evidence"]) is not dict
            or type(binding["external_artifact_evidence"]) is not dict
            or binding["supervisor_artifact_present"]
            != (
                binding["supervisor_artifact_evidence"].get("state")
                != "ABSENT"
            )
            or binding["external_artifact_present"]
            != (
                binding["external_artifact_evidence"].get("state")
                != "ABSENT"
            )
            or binding["supervisor_artifact_raw_sha256_or_null"]
            != binding["supervisor_artifact_evidence"].get(
                "raw_sha256_or_null"
            )
            or binding["external_artifact_raw_sha256_or_null"]
            != binding["external_artifact_evidence"].get(
                "raw_sha256_or_null"
            )
        ):
            _ledger_semantic_fail(f"{slot['slot_id']} binding evidence differs")
        if outcome_name not in by_name:
            _ledger_semantic_fail("binding-only terminal state is forbidden")
        outcome_record = by_name[outcome_name]
        outcome = outcome_record["body"]
        if type(outcome) is not dict or set(outcome) != {
            "schema_version",
            "campaign_id",
            "contract_digest",
            "capsule_digest",
            "slot_id",
            "slot_ordinal",
            "external_trajectory_binding_digest",
            "observation",
            "decision",
            "record_digest",
        }:
            _ledger_semantic_fail(f"{slot['slot_id']} outcome fields differ")
        outcome_unsigned = deepcopy(outcome)
        outcome_digest = outcome_unsigned.pop("record_digest", None)
        observation = outcome.get("observation")
        decision = outcome.get("decision")
        if (
            outcome_record["status"] != decision
            or outcome["schema_version"]
            != "hegel-phase3a-q05b-shadow-retry-slot-outcome/2"
            or outcome["campaign_id"] != contract["contract_id"]
            or outcome["contract_digest"] != contract["contract_digest"]
            or outcome["capsule_digest"] != capsule["capsule_digest"]
            or outcome["slot_id"] != slot["slot_id"]
            or outcome["slot_ordinal"] != ordinal
            or outcome["external_trajectory_binding_digest"]
            != binding_digest
            or type(observation) is not dict
            or type(decision) is not str
            or outcome_digest
            != _domain_digest(
                contract["hash_domains"]["slot_outcome"],
                outcome_unsigned,
            )
        ):
            _ledger_semantic_fail(f"{slot['slot_id']} outcome mapping differs")
        try:
            core_decision = _load_core(contract).evaluate_slot_observation_v2(
                observation
            )
        except BaseException as error:
            _ledger_semantic_fail(
                f"{slot['slot_id']} observation rejected by core: "
                f"{type(error).__name__}:{error}"
            )
        if decision != core_decision:
            _ledger_semantic_fail(f"{slot['slot_id']} core decision differs")
        if (
            ordinal < len(slots)
            and f"slot-{ordinal + 1:02d}.intent.json" in by_name
            and decision != _load_core(contract).RETRY_EXACT_GAP_V2
        ):
            _ledger_semantic_fail("later slot lacks exact-gap predecessor")
        _verify_binding_observation_crosslinks_v2(
            contract,
            slot,
            binding,
            observation,
        )
        if decision == _load_core(contract).SUCCESS_V2:
            stdout_facts = {
                key: deepcopy(observation[key])
                for key in (
                    "stdout_raw_byte_count",
                    "stdout_raw_sha256",
                    "stdout_canonical_single_lf",
                    "stdout_wire_kind",
                    "stdout_wire_value_or_null",
                    "stdout_status_or_null",
                    "stdout_artifact_written_or_null",
                )
            }
            if not _artifact_matches_success_wire(
                contract,
                slot,
                stdout_facts,
                binding["supervisor_artifact_evidence"],
                _trajectory_from_binding_v2(binding),
            ):
                _ledger_semantic_fail(
                    f"{slot['slot_id']} success artifact cross-link differs"
                )
        full_outcomes.append((outcome, observation))

    for outcome, _observation in full_outcomes[:-1]:
        if outcome["decision"] != _load_core(contract).RETRY_EXACT_GAP_V2:
            _ledger_semantic_fail("later slot lacks exact-gap predecessor")
    if consumed_count != sum(
        record["record_type"] == "SLOT_CONSUMED" for record in records
    ):
        _ledger_semantic_fail("consumed count cross-link differs")

    terminal = records[-2]["body"]
    error_code = terminal["error_code_or_null"]
    error_detail = terminal["error_detail_or_null"]
    error_receipt = terminal["error_receipt_or_null"]
    last_outcome = full_outcomes[-1][0] if full_outcomes else None
    expected_last_slot = (
        last_outcome["slot_id"] if type(last_outcome) is dict else None
    )
    expected_last_digest = (
        last_outcome["record_digest"] if type(last_outcome) is dict else None
    )
    if (
        terminal["supervisor_invocation_count"] != consumed_count
        or terminal["last_slot_id_or_null"] != expected_last_slot
        or terminal["last_slot_outcome_digest_or_null"] != expected_last_digest
    ):
        _ledger_semantic_fail("terminal progress differs from frozen records")
    active_prefix_match = (
        re.fullmatch(
            r"slot-([0-9]{2})\.(intent|consumed|trajectory-binding|outcome)\.json",
            prefix_names[-1],
        )
        if prefix_names
        else None
    )
    expected_active_slot_id_or_null = (
        "SLOT_" + active_prefix_match.group(1)
        if active_prefix_match is not None
        else None
    )
    expected_prespawn_slot_id_or_null = (
        expected_active_slot_id_or_null
        if active_prefix_match is not None
        and active_prefix_match.group(2) == "intent"
        else None
    )
    has_error = error_code is not None
    if has_error:
        if (
            type(error_code) is not str
            or not error_code
            or len(error_code) > 256
            or type(error_detail) is not str
            or not error_detail
            or len(error_detail) > 4096
            or error_code
            in {
                "FAIL_CAMPAIGN_LEDGER",
                "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN",
                "FAIL_CAMPAIGN_SPAWN_UNCERTAIN",
            }
            or (
                error_receipt is not None
                and (
                    type(error_receipt) is not dict
                    or len(_canonical_json_bytes(error_receipt))
                    > _MAX_ERROR_RECEIPT_BYTES
                )
            )
            or terminal["status"]
            != _error_code_terminal_disposition(contract, error_code)
            or not _terminal_error_receipt_semantics_v2(
                contract,
                error_receipt,
                terminal_error_code=error_code,
                terminal_error_detail=error_detail,
                top_level=True,
                expected_prespawn_slot_id_or_null=(
                    expected_prespawn_slot_id_or_null
                ),
                expected_active_slot_id_or_null=(
                    expected_active_slot_id_or_null
                ),
            )
        ):
            _ledger_semantic_fail("terminal error disposition differs")
    else:
        if error_detail is not None or error_receipt is not None or not full_outcomes:
            _ledger_semantic_fail("normal terminal error fields differ")
        decision = last_outcome["decision"]
        observation = full_outcomes[-1][1]
        core = _load_core(contract)
        if decision == core.SUCCESS_V2:
            expected_status = contract["success_policy"]["terminal_status"]
        elif decision == core.EXHAUSTED_RETRYABLE_GAP_V2:
            if last_outcome["slot_id"] != "SLOT_03":
                _ledger_semantic_fail("retry exhaustion is not slot three")
            expected_status = contract["terminal_failure_policy"][
                "slot_03_exact_gap_disposition"
            ]
        elif decision == core.TERMINAL_NONRETRYABLE_V2:
            expected_status = _observation_terminal_disposition(
                contract,
                observation,
            )
        else:
            _ledger_semantic_fail("retry decision cannot be a normal terminal")
        if terminal["status"] != expected_status:
            _ledger_semantic_fail("normal terminal status differs")


def _verify_campaign_completion_chain_v2(
    contract: Mapping[str, object],
    capsule: Mapping[str, object],
    authorization: Mapping[str, object],
    ordered_records: Sequence[tuple[str, Mapping[str, object]]],
) -> dict[str, object]:
    """Producer-free verification of an ordered, decoded immutable ledger."""

    if type(ordered_records) not in (list, tuple) or len(ordered_records) < 4:
        _fail(
            "FAIL_CAMPAIGN_LEDGER",
            "authoritative completion chain is absent",
            terminal=True,
        )
    policy = contract.get("ledger_policy")
    if type(policy) is not dict:
        _fail("FAIL_CAMPAIGN_CONTRACT", "ledger policy differs", terminal=True)
    allowed_names = policy.get("record_names_in_exact_order")
    if allowed_names != list(_LEDGER_RECORD_NAMES_V2):
        _fail(
            "FAIL_CAMPAIGN_CONTRACT",
            "ledger filename registry differs",
            terminal=True,
        )
    names: list[str] = []
    records: list[Mapping[str, object]] = []
    previous_digest: str | None = None
    last_name_index = -1
    expected_fields = {
        "schema_version",
        "campaign_id",
        "contract_digest",
        "campaign_source_binding_digest",
        "capsule_digest",
        "authorization_digest",
        "sequence",
        "record_type",
        "status",
        "previous_record_digest_or_null",
        "body",
        "record_digest",
    }
    for ordinal, row in enumerate(ordered_records, start=1):
        if type(row) not in (list, tuple) or len(row) != 2:
            _fail(
                "FAIL_CAMPAIGN_LEDGER",
                "offline ledger row shape differs",
                terminal=True,
            )
        basename, record = row
        if type(basename) is not str or type(record) is not dict:
            _fail(
                "FAIL_CAMPAIGN_LEDGER",
                "offline ledger row type differs",
                terminal=True,
            )
        if basename not in allowed_names or basename in names:
            _fail(
                "FAIL_CAMPAIGN_LEDGER",
                "offline ledger filename differs",
                terminal=True,
            )
        name_index = allowed_names.index(basename)
        if basename == "campaign.terminal.json":
            if name_index <= last_name_index:
                _fail(
                    "FAIL_CAMPAIGN_LEDGER",
                    "offline terminal order differs",
                    terminal=True,
                )
        elif name_index != last_name_index + 1:
            _fail(
                "FAIL_CAMPAIGN_LEDGER",
                "offline ledger state transition differs",
                terminal=True,
            )
        if set(record) != expected_fields or type(record.get("body")) is not dict:
            _fail(
                "FAIL_CAMPAIGN_LEDGER",
                "offline ledger record fields differ",
                terminal=True,
            )
        if (
            record["schema_version"]
            != "hegel-phase3a-q05b-shadow-retry-campaign-ledger-record/2"
            or record["campaign_id"] != contract["contract_id"]
            or record["contract_digest"] != contract["contract_digest"]
            or record["campaign_source_binding_digest"]
            != contract["campaign_source_binding"]["source_binding_digest"]
            or record["capsule_digest"] != capsule["capsule_digest"]
            or record["authorization_digest"]
            != authorization["authorization_digest"]
            or type(record["sequence"]) is not int
            or record["sequence"] != ordinal
            or record["record_type"]
            != _ledger_record_type_for_basename(basename)
            or type(record["status"]) is not str
            or not record["status"]
            or record["previous_record_digest_or_null"] != previous_digest
            or type(record["record_digest"]) is not str
            or _TAGGED_SHA256.fullmatch(record["record_digest"]) is None
        ):
            _fail(
                "FAIL_CAMPAIGN_LEDGER",
                "offline ledger identity or chain differs",
                terminal=True,
            )
        unsigned = deepcopy(dict(record))
        observed_digest = unsigned.pop("record_digest")
        expected_digest = _domain_digest(
            contract["hash_domains"]["ledger_record"],
            unsigned,
        )
        if observed_digest != expected_digest:
            _fail(
                "FAIL_CAMPAIGN_LEDGER",
                "offline ledger record digest differs",
                terminal=True,
            )
        names.append(basename)
        records.append(record)
        previous_digest = str(observed_digest)
        last_name_index = name_index
    if (
        names[:2]
        != ["campaign.intent.json", "campaign.authorization-consumed.json"]
        or names[-2:]
        != ["campaign.terminal.json", "campaign.completion.json"]
    ):
        _fail(
            "FAIL_CAMPAIGN_LEDGER",
            "terminal-only or misordered completion chain is non-authoritative",
            terminal=True,
        )
    terminal_record = records[-2]
    completion_record = records[-1]
    terminal_body = terminal_record["body"]
    completion_body = completion_record["body"]
    assert type(terminal_body) is dict and type(completion_body) is dict
    terminal_fields = {
        "schema_version",
        "campaign_id",
        "contract_digest",
        "capsule_digest",
        "status",
        "supervisor_invocation_count",
        "last_slot_id_or_null",
        "last_slot_outcome_digest_or_null",
        "error_code_or_null",
        "error_detail_or_null",
        "error_receipt_or_null",
        "shadow_only",
        "nonadopting",
        "q1_state",
        "active_transition",
        "adoption",
        "promotion",
        "current_pointer_written",
        "record_digest",
    }
    if set(terminal_body) != terminal_fields:
        _fail(
            "FAIL_CAMPAIGN_LEDGER",
            "offline campaign terminal body fields differ",
            terminal=True,
        )
    terminal_unsigned = deepcopy(terminal_body)
    terminal_digest = terminal_unsigned.pop("record_digest")
    if (
        terminal_body["schema_version"]
        != "hegel-phase3a-q05b-shadow-retry-campaign-terminal/2"
        or terminal_body["campaign_id"] != contract["contract_id"]
        or terminal_body["contract_digest"] != contract["contract_digest"]
        or terminal_body["capsule_digest"] != capsule["capsule_digest"]
        or terminal_body["status"] != terminal_record["status"]
        or type(terminal_body["status"]) is not str
        or not terminal_body["status"]
        or type(terminal_body["supervisor_invocation_count"]) is not int
        or not 0 <= terminal_body["supervisor_invocation_count"] <= 3
        or (
            terminal_body["last_slot_id_or_null"] is not None
            and terminal_body["last_slot_id_or_null"]
            not in {"SLOT_01", "SLOT_02", "SLOT_03"}
        )
        or (
            terminal_body["last_slot_outcome_digest_or_null"] is not None
            and (
                type(terminal_body["last_slot_outcome_digest_or_null"])
                is not str
                or _TAGGED_SHA256.fullmatch(
                    terminal_body["last_slot_outcome_digest_or_null"]
                )
                is None
            )
        )
        or (
            terminal_body["error_code_or_null"] is not None
            and type(terminal_body["error_code_or_null"]) is not str
        )
        or (
            terminal_body["error_detail_or_null"] is not None
            and type(terminal_body["error_detail_or_null"]) is not str
        )
        or (
            terminal_body["error_receipt_or_null"] is not None
            and type(terminal_body["error_receipt_or_null"]) is not dict
        )
        or terminal_body["shadow_only"] is not True
        or terminal_body["nonadopting"] is not True
        or terminal_body["q1_state"] != "NOT_RUN"
        or any(
            terminal_body[field] is not False
            for field in (
                "active_transition",
                "adoption",
                "promotion",
                "current_pointer_written",
            )
        )
        or terminal_digest
        != _domain_digest(
            contract["hash_domains"]["campaign_terminal"],
            terminal_unsigned,
        )
    ):
        _fail(
            "FAIL_CAMPAIGN_LEDGER",
            "offline campaign terminal body differs",
            terminal=True,
        )
    _verify_ledger_state_semantics_v2(
        contract,
        capsule,
        authorization,
        names,
        records,
    )
    expected_completion = _campaign_completion_body(
        contract,
        capsule,
        authorization_digest=str(authorization["authorization_digest"]),
        terminal=terminal_body,
        terminal_ledger_record=terminal_record,
        preceding_ledger_names=names[:-1],
    )
    if (
        completion_record["status"] != "CAMPAIGN_COMPLETION_COMMITTED"
        or completion_body != expected_completion
    ):
        _fail(
            "FAIL_CAMPAIGN_LEDGER",
            "offline campaign completion body differs",
            terminal=True,
        )
    return {
        "schema_version": (
            "hegel-phase3a-q05b-shadow-retry-campaign-completion-verification/2"
        ),
        "status": "AUTHORITATIVE_CAMPAIGN_COMPLETION_VERIFIED",
        "terminal_status": terminal_body["status"],
        "ledger_record_count": len(records),
        "campaign_completion_digest": completion_body["record_digest"],
        "ledger_completion_record_digest": completion_record["record_digest"],
    }


def _verify_campaign_completion_ledger_at_v2(
    contract: Mapping[str, object],
    capsule: Mapping[str, object],
    authorization: Mapping[str, object],
    ledger_root: Path,
) -> dict[str, object]:
    """Safely replay canonical immutable ledger files without the producer."""

    expected_root = Path(str(contract["fixed_external_roots"]["ledger_root"]))
    if ledger_root != expected_root:
        _fail(
            "FAIL_CAMPAIGN_LEDGER",
            "offline ledger root differs from the contract",
            terminal=True,
        )
    root_fd, _root_status = _open_directory_nofollow(
        ledger_root,
        "offline campaign ledger",
    )
    try:
        before = _verify_held_directory_path(
            ledger_root,
            root_fd,
            "offline campaign ledger",
        )
        if (
            not stat.S_ISDIR(int(before["mode"]))
            or stat.S_IMODE(int(before["mode"])) != 0o700
            or before["uid"] != os.geteuid()
            or int(before["nlink"]) < 2
            or int(before["mount_id"]) <= 0
        ):
            _fail(
                "FAIL_CAMPAIGN_LEDGER",
                "offline campaign ledger root identity differs",
                terminal=True,
            )
        policy = contract["ledger_policy"]
        assert type(policy) is dict
        allowed_names = policy["record_names_in_exact_order"]
        assert type(allowed_names) is list
        observed_names = os.listdir(root_fd)
        if (
            any(type(name) is not str for name in observed_names)
            or len(observed_names) != len(set(observed_names))
            or not set(observed_names).issubset(set(allowed_names))
        ):
            _fail(
                "FAIL_CAMPAIGN_LEDGER",
                "offline campaign ledger namespace differs",
                terminal=True,
            )
        ordered_names = [
            name for name in allowed_names if name in set(observed_names)
        ]
        ordered_records: list[tuple[str, Mapping[str, object]]] = []
        for basename in ordered_names:
            raw, _status = _read_regular_at_nofollow(
                root_fd,
                basename,
                f"offline ledger record {basename}",
                _MAX_LEDGER_BYTES,
                required_mode=0o444,
            )
            decoded = _strict_json(
                raw,
                f"offline ledger record {basename}",
                _MAX_LEDGER_BYTES,
            )
            if type(decoded) is not dict:
                _fail(
                    "FAIL_CAMPAIGN_LEDGER",
                    f"offline ledger record is not an object: {basename}",
                    terminal=True,
                )
            ordered_records.append((basename, decoded))
        after = _verify_held_directory_path(
            ledger_root,
            root_fd,
            "offline campaign ledger final replay",
        )
        if after != before:
            _fail(
                "FAIL_CAMPAIGN_LEDGER",
                "offline campaign ledger root changed during replay",
                terminal=True,
            )
        return _verify_campaign_completion_chain_v2(
            contract,
            capsule,
            authorization,
            ordered_records,
        )
    finally:
        os.close(root_fd)


def _publish_terminal(
    ledger: _LedgerV2,
    contract: Mapping[str, object],
    capsule: Mapping[str, object],
    *,
    status_text: str,
    invocation_count: int,
    last_slot_id_or_null: str | None,
    last_slot_outcome_digest_or_null: str | None,
    error_code_or_null: str | None = None,
    error_detail_or_null: str | None = None,
    error_receipt_or_null: Mapping[str, object] | None = None,
) -> dict[str, object]:
    cached = _authoritative_completion_result_v2(ledger)
    if cached is not None:
        return cached
    progress = _ledger_progress_snapshot_v2(ledger.published_records)
    # These legacy call-site values are intentionally non-authoritative.  A
    # signal may arrive after a frozen consumed/outcome record is committed but
    # before the corresponding Python local is updated.
    _ = (
        invocation_count,
        last_slot_id_or_null,
        last_slot_outcome_digest_or_null,
    )
    terminal = _campaign_terminal_body(
        contract,
        capsule,
        status_text=status_text,
        invocation_count=int(progress["supervisor_invocation_count"]),
        last_slot_id_or_null=progress["last_slot_id_or_null"],
        last_slot_outcome_digest_or_null=progress[
            "last_slot_outcome_digest_or_null"
        ],
        error_code_or_null=error_code_or_null,
        error_detail_or_null=error_detail_or_null,
        error_receipt_or_null=error_receipt_or_null,
    )
    result: dict[str, object] | None = None
    try:
        ledger_record = ledger.publish(
            "campaign.terminal.json",
            "CAMPAIGN_TERMINAL",
            status_text,
            terminal,
        )
        completion = _campaign_completion_body(
            contract,
            capsule,
            authorization_digest=str(
                ledger.authorization["authorization_digest"]
            ),
            terminal=terminal,
            terminal_ledger_record=ledger_record,
            preceding_ledger_names=ledger.published_names,
        )
        prepared_completion_bytes = ledger.prepare_bytes(
            "campaign.completion.json",
            "CAMPAIGN_COMPLETION",
            "CAMPAIGN_COMPLETION_COMMITTED",
            completion,
        )
        prepared_completion_record = _strict_json(
            prepared_completion_bytes,
            "prepared campaign completion ledger record",
            _MAX_LEDGER_BYTES,
        )
        if type(prepared_completion_record) is not dict:
            _fail(
                "FAIL_CAMPAIGN_LEDGER",
                "prepared campaign completion record differs",
                terminal=True,
            )
        result = _campaign_result_from_prepared_completion(
            contract,
            capsule,
            terminal=terminal,
            terminal_ledger_record=ledger_record,
            completion=completion,
            completion_ledger_record=prepared_completion_record,
        )
        ledger.cache_completion_result_candidate(
            result,
            prepared_completion_record,
        )
        ledger.publish_prepared(
            "campaign.completion.json",
            prepared_completion_bytes,
        )
        cached = _authoritative_completion_result_v2(ledger)
        if cached is None:
            _raise_postpublication_uncertain(
                "campaign completion authoritative result cache",
                publication_basename_or_null="campaign.completion.json",
            )
        return cached
    except BaseException as error:
        cached = _authoritative_completion_result_v2(ledger)
        if cached is not None:
            return cached
        recovered = _recover_completion_candidate_from_disk_v2(
            ledger,
            contract,
            capsule,
            result,
        )
        if recovered is not None:
            return recovered
        if "campaign.terminal.json" in ledger.published_names:
            _raise_postpublication_uncertain(
                "campaign completion continuation",
                error,
                publication_basename_or_null="campaign.completion.json",
            )
        raise


def _authoritative_completion_result_v2(
    ledger: object,
) -> dict[str, object] | None:
    result = getattr(
        ledger,
        "authoritative_completion_result_or_null",
        None,
    )
    if type(result) is dict:
        return deepcopy(result)
    candidate = getattr(
        ledger,
        "completion_result_candidate_or_null",
        None,
    )
    if (
        getattr(ledger, "completion_commit_verified", False) is True
        and type(candidate) is dict
    ):
        return deepcopy(candidate)
    return None


def _recover_completion_candidate_from_disk_v2(
    ledger: object,
    contract: Mapping[str, object],
    capsule: Mapping[str, object],
    expected_result_or_null: Mapping[str, object] | None = None,
) -> dict[str, object] | None:
    candidate_value: object = expected_result_or_null
    if type(candidate_value) is not dict:
        candidate_value = getattr(
            ledger,
            "completion_result_candidate_or_null",
            None,
        )
    if type(candidate_value) is not dict:
        return None
    candidate = deepcopy(candidate_value)
    authorization = getattr(ledger, "authorization", None)
    if type(authorization) is not dict:
        return None
    try:
        verification = _verify_campaign_completion_ledger_at_v2(
            contract,
            capsule,
            authorization,
            Path(str(contract["fixed_external_roots"]["ledger_root"])),
        )
    except BaseException:
        return None
    if (
        verification.get("status")
        != "AUTHORITATIVE_CAMPAIGN_COMPLETION_VERIFIED"
        or verification.get("terminal_status") != candidate.get("status")
        or verification.get("campaign_completion_digest")
        != candidate.get("campaign_completion_digest")
        or verification.get("ledger_completion_record_digest")
        != candidate.get("ledger_completion_record_digest")
        or candidate.get("completion_commit_verified") is not True
    ):
        return None
    setattr(
        ledger,
        "authoritative_completion_result_or_null",
        deepcopy(candidate),
    )
    setattr(ledger, "completion_commit_verified", True)
    return candidate


def _observation_terminal_disposition(
    contract: Mapping[str, object],
    observation: object,
) -> str:
    policy = contract["terminal_failure_policy"]
    assert type(policy) is dict
    if type(observation) is not dict:
        return str(policy["generic_nonretry_disposition"])
    trajectory_uncertain = (
        observation.get("trajectory_status")
        not in {"COMPLETE_OK", "COMPLETE_FAIL"}
        or observation.get("trajectory_torn_tail_or_null") is not None
        or observation.get("trajectory_open_span_ids") != []
        or observation.get("trajectory_trace_id")
        != observation.get("expected_trace_id")
        or observation.get("trajectory_producer_id")
        != observation.get("expected_producer_id")
    )
    if trajectory_uncertain:
        return str(
            policy["trajectory_incomplete_or_unverifiable_disposition"]
        )
    if (
        observation.get("cleanup_certain") is not True
        or observation.get("docker_absence_certain") is not True
        or observation.get("supervisor_reaped") is not True
        or observation.get("process_group_absent") is not True
    ):
        return str(policy["cleanup_or_survivor_uncertainty_disposition"])
    if any(
        observation.get(field) is not True
        for field in (
            "clone_clean_certain",
            "cargo_isolation_certain",
            "source_binding_certain",
            "external_binding_certain",
            "replay_certain",
        )
    ):
        return str(policy["source_or_binding_uncertainty_disposition"])
    return str(policy["generic_nonretry_disposition"])


def _error_code_terminal_disposition(
    contract: Mapping[str, object],
    code: object,
) -> str:
    policy = contract["terminal_failure_policy"]
    assert type(policy) is dict
    if code == _SCHEDULING_ADMISSION_FAILURE_V2:
        return str(policy["generic_nonretry_disposition"])
    if code in {"FAIL_CAMPAIGN_PROCESS", "FAIL_CAMPAIGN_DOCKER"}:
        return str(policy["cleanup_or_survivor_uncertainty_disposition"])
    if code == "FAIL_CAMPAIGN_TRAJECTORY":
        return str(
            policy["trajectory_incomplete_or_unverifiable_disposition"]
        )
    if code in {
        "FAIL_CAMPAIGN_SOURCE",
        "FAIL_CAMPAIGN_BINDING",
        "FAIL_CAMPAIGN_FRESHNESS",
        "FAIL_CAMPAIGN_FILESYSTEM",
        "FAIL_CAMPAIGN_ARTIFACT",
        "FAIL_CAMPAIGN_EVIDENCE",
        "FAIL_CAMPAIGN_OBSERVATION",
    }:
        return str(policy["source_or_binding_uncertainty_disposition"])
    return str(policy["crash_disposition"])


def _exception_terminal_disposition(
    contract: Mapping[str, object],
    error: BaseException,
) -> str:
    return _error_code_terminal_disposition(
        contract,
        getattr(error, "code", type(error).__name__),
    )


def _run_one_slot(
    contract: Mapping[str, object],
    context: Mapping[str, object],
    capsule: Mapping[str, object],
    ledger: _LedgerV2,
    slot: Mapping[str, object],
    invocation_counter: list[int],
) -> tuple[dict[str, object], str]:
    core = _load_core(contract)
    execution = contract["execution_policy"]
    assert type(execution) is dict
    ordinal = int(slot["slot_ordinal"])
    stem = f"slot-{ordinal:02d}"
    argv = _slot_argv(contract, slot)
    environment = _runtime_environment(
        contract,
        git_worktree=str(slot["clone_root"]),
    )
    slot_freshness_before_intent = _slot_roots_absence_evidence(
        context,
        [slot],
        label=f"{slot['slot_id']} before intent",
    )
    if slot_freshness_before_intent["certain"] is not True:
        _fail(
            "FAIL_CAMPAIGN_FRESHNESS",
            str(slot_freshness_before_intent["error_or_null"]),
            terminal=True,
        )
    ledger.publish(
        f"{stem}.intent.json",
        "SLOT_INTENT",
        f"{slot['slot_id']}_INTENT_STAGED",
        {
            "slot": deepcopy(dict(slot)),
            "argv": argv,
            "environment": environment,
            "previous_slot_id_or_null": slot["previous_slot_id"],
            "slot_freshness_before_intent": slot_freshness_before_intent,
        },
    )

    _prepare_clone(contract, slot)
    cache_identity = _copy_tree_exact(
        Path(str(execution["cargo_copy_source"])),
        Path(str(slot["cargo_cache_root"])),
        context["cargo_rows"],
    )
    if (
        cache_identity["content_manifest_digest"]
        != context["cargo_identity"]["content_manifest_digest"]
    ):
        _fail(
            "FAIL_CAMPAIGN_SOURCE",
            f"{slot['slot_id']} copied Cargo cache differs",
            terminal=True,
        )
    source_before = _source_state_replay(
        contract,
        context,
        slot,
        artifact_presence="ABSENT",
    )
    docker_before_absent, docker_before_rows, docker_before_error = (
        _safe_docker_inventory(contract)
    )
    if (
        docker_before_absent is not True
        or docker_before_rows
        or docker_before_error is not None
    ):
        decoded_diagnostic: object | None = None
        if docker_before_error is not None:
            try:
                docker_error_bytes = (
                    docker_before_error.encode("ascii") + b"\n"
                )
                if len(docker_error_bytes) > 16 * 1024:
                    decoded_diagnostic = {
                        "schema_version": _BOUNDED_RECEIPT_OMISSION_SCHEMA_V2,
                        "status": "RECEIPT_OMITTED_OVER_BOUND",
                    }
                else:
                    decoded_diagnostic = _strict_json(
                        docker_error_bytes,
                        "pre-spawn Docker diagnostic",
                        16 * 1024,
                    )
            except BaseException as error:
                decoded_diagnostic = _bounded_process_exception_diagnostic(
                    "Docker read-only inventory",
                    _docker_inventory_argv_v2(contract),
                    error,
                )
                docker_before_error = _canonical_json_bytes(
                    decoded_diagnostic
                ).decode("ascii").rstrip("\n")
        docker_receipt: dict[str, object] = {
            "schema_version": _PRESPAWN_DOCKER_RECEIPT_SCHEMA_V2,
            "docker_rows": [deepcopy(dict(row)) for row in docker_before_rows],
            "docker_error_or_null": docker_before_error,
        }
        if docker_before_error is not None:
            docker_receipt["bounded_command_diagnostic_or_null"] = (
                decoded_diagnostic
            )
        _fail(
            "FAIL_CAMPAIGN_DOCKER",
            f"{slot['slot_id']} pre-spawn Docker absence is uncertain: "
            f"{docker_before_error or docker_before_rows}",
            terminal=True,
            receipt=docker_receipt,
        )

    evidence_parent_fd, evidence_fd = _create_evidence_root(slot)
    os.close(evidence_parent_fd)
    evidence_identity = _verify_held_directory_path(
        Path(str(slot["evidence_root"])),
        evidence_fd,
        f"{slot['slot_id']} evidence root",
    )
    alias_base_paths = [
        Path(str(execution["clone_source_repository"])),
        Path(str(execution["cargo_copy_source"])),
        Path(str(contract["fixed_external_roots"]["authorization_root"])),
        Path(str(contract["fixed_external_roots"]["ledger_root"])),
        Path(str(slot["clone_root"])),
        Path(str(slot["cargo_cache_root"])),
        Path(str(slot["evidence_root"])),
    ]
    alias_before = _assert_existing_directory_roots_nonaliased(
        alias_base_paths,
        f"{slot['slot_id']} pre-spawn",
    )
    source_before = _augment_source_replay_with_alias_rows(
        source_before,
        alias_before,
    )
    consumed_record: list[dict[str, object]] = []
    scheduling_admission: dict[str, object] | None = None

    def consume_before_spawn() -> None:
        if (
            consumed_record
            or invocation_counter[0] >= 3
            or not _scheduling_admission_evidence_exact_v2(
                contract, scheduling_admission
            )
            or scheduling_admission.get("slot_id") != slot["slot_id"]
        ):
            _fail(
                "FAIL_CAMPAIGN_INVOCATION",
                "slot consumption or invocation count differs",
                terminal=True,
            )
        record = ledger.publish(
            f"{stem}.consumed.json",
            "SLOT_CONSUMED",
            f"{slot['slot_id']}_CONSUMED_OUTCOME_UNKNOWN",
            {
                "slot_id": slot["slot_id"],
                "slot_ordinal": ordinal,
                "argv": argv,
                "source_replay_before_root": source_before["root"],
                "docker_absence_before_spawn": True,
                "scheduling_admission": deepcopy(scheduling_admission),
            },
        )
        consumed_record.append(record)
        invocation_counter[0] += 1

    external_binding_attempted = False
    try:
        scheduling_admission = _run_scheduling_admission_gate_v2(
            contract, evidence_fd, str(slot["slot_id"])
        )
        try:
            process_result = _bounded_process(
                argv,
                cwd=str(slot["clone_project_root"]),
                env=environment,
                timeout_seconds=float(execution["supervisor_timeout_seconds"]),
                stdout_max_bytes=int(execution["stdout_max_bytes"]),
                stderr_max_bytes=int(execution["stderr_max_bytes"]),
                first_signal=signal.SIGINT,
                grace_seconds=float(execution["sigint_grace_seconds"]),
                reap_seconds=float(execution["sigkill_reap_seconds"]),
                before_popen=consume_before_spawn,
            )
        except BaseException as error:
            _raise_bounded_process_exception(
                f"{slot['slot_id']} bound supervisor",
                argv,
                error,
            )
        if len(consumed_record) != 1:
            _fail(
                "FAIL_CAMPAIGN_INVOCATION",
                "slot process launched without one consumed record",
                terminal=True,
            )
        _publish_raw_streams(
            slot,
            evidence_fd,
            process_result,
            max(
                int(execution["stdout_max_bytes"]),
                int(execution["stderr_max_bytes"]),
            ),
        )
        _postpublication_directory_identity(
            Path(str(slot["evidence_root"])),
            evidence_fd,
            f"{slot['slot_id']} raw stream replay",
            evidence_identity,
        )

        artifact_presence = (
            "PRESENT"
            if os.path.lexists(Path(str(slot["supervisor_artifact_path"])))
            else "ABSENT"
        )
        source_after = _safe_source_state_replay(
            contract,
            context,
            slot,
            artifact_presence=artifact_presence,
        )
        try:
            alias_after = _assert_existing_directory_roots_nonaliased(
                [
                    *alias_base_paths,
                    Path(str(slot["trace_root"])),
                    Path(str(slot["artifact_parent_root"])),
                ],
                f"{slot['slot_id']} post-reap",
            )
            source_after = _augment_source_replay_with_alias_rows(
                source_after,
                alias_after,
            )
        except BaseException as error:
            detail = (
                f"{getattr(error, 'code', type(error).__name__)}:"
                f"{getattr(error, 'detail', str(error))}"
            )[:4096]
            source_after = {
                "certain": False,
                "error_or_null": detail,
                "evidence": {"error": detail},
                "root": _domain_digest(
                    "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R8/SOURCE_REPLAY/V2",
                    {"error": detail},
                ),
            }
        source_after = _enforce_clone_closure(
            slot,
            source_before,
            source_after,
        )
        trajectory = _safe_trajectory_evidence(
            contract,
            slot,
            source_certain=bool(source_after["certain"]),
        )
        if source_after["certain"] is True:
            supervisor_artifact = _artifact_replay(contract, slot)
        elif artifact_presence == "ABSENT":
            supervisor_artifact = {
                "state": "ABSENT",
                "raw_sha256_or_null": None,
                "size_or_null": None,
                "replay_passed": False,
                "summary_or_null": None,
                "error_or_null": source_after["error_or_null"],
                "parent_identity_or_null": None,
                "file_identity_or_null": None,
            }
        else:
            supervisor_artifact = {
                "state": "RESIDUAL_INVALID",
                "raw_sha256_or_null": None,
                "size_or_null": None,
                "replay_passed": False,
                "summary_or_null": None,
                "error_or_null": source_after["error_or_null"],
                "parent_identity_or_null": None,
                "file_identity_or_null": None,
            }
        stdout_facts = core.classify_supervisor_stdout_v2(
            process_result["stdout"]
        )
        docker_absent, docker_rows, docker_error = _safe_docker_inventory(
            contract
        )
        trajectory_complete = _trajectory_complete_and_bound(
            contract, slot, trajectory
        )
        success_candidate = (
            process_result["return_code"] == 0
            and process_result["timed_out"] is False
            and process_result["stdout_overflow"] is False
            and process_result["stderr_overflow"] is False
            and process_result["child_reaped"] is True
            and process_result["process_group_absent"] is True
            and process_result["wrapper_resources_closed"] is True
            and process_result["stderr"] == b""
            and source_before["certain"] is True
            and source_after["certain"] is True
            and docker_absent is True
            and not docker_rows
            and docker_error is None
            and trajectory_complete
            and trajectory["verification_status"] == "COMPLETE_OK"
            and _artifact_matches_success_wire(
                contract,
                slot,
                stdout_facts,
                supervisor_artifact,
                trajectory,
            )
        )
        if success_candidate:
            external_artifact_publication_attempted = False
            committed_external_artifact: list[dict[str, object]] = []

            def cache_committed_external_artifact(
                value: dict[str, object],
            ) -> None:
                committed_external_artifact.append(deepcopy(value))

            try:
                external_artifact_publication_attempted = True
                external_artifact = _publish_external_artifact(
                    slot,
                    supervisor_artifact,
                    commit_callback=cache_committed_external_artifact,
                )
            except ShadowRetryCampaignErrorV2 as error:
                if committed_external_artifact:
                    _raise_postpublication_uncertain(
                        f"{slot['slot_id']} external artifact caller continuation",
                        error,
                        publication_basename_or_null=Path(
                            str(slot["artifact_path"])
                        ).name,
                    )
                if error.code == "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN":
                    # Once O_CREAT may have succeeded, the contract forbids
                    # every later evidence or terminal write.
                    raise
                if error.code != "FAIL_CAMPAIGN_ARTIFACT":
                    _guard_publication_attempt_failure(
                        f"{slot['slot_id']} external artifact caller continuation",
                        Path(str(slot["artifact_path"])).name,
                        "FAIL_CAMPAIGN_ARTIFACT",
                        error,
                    )
                external_artifact = {
                    "state": "RESIDUAL_INVALID",
                    "raw_sha256_or_null": None,
                    "size_or_null": None,
                    "replay_passed": False,
                    "error_or_null": f"{error.code}:{error.detail}"[:4096],
                    "parent_identity_or_null": None,
                    "file_identity_or_null": None,
                }
            except BaseException as error:
                if external_artifact_publication_attempted:
                    _raise_postpublication_uncertain(
                        f"{slot['slot_id']} external artifact caller continuation",
                        error,
                        publication_basename_or_null=Path(
                            str(slot["artifact_path"])
                        ).name,
                    )
                raise
        else:
            try:
                external_artifact = _absent_external_artifact(context, slot)
            except BaseException as error:
                external_artifact = {
                    "state": "RESIDUAL_INVALID",
                    "raw_sha256_or_null": None,
                    "size_or_null": None,
                    "replay_passed": False,
                    "error_or_null": (
                        f"{getattr(error, 'code', type(error).__name__)}:"
                        f"{getattr(error, 'detail', str(error))}"
                    )[:4096],
                    "parent_identity_or_null": None,
                    "file_identity_or_null": None,
                }

        try:
            alias_final = _assert_existing_directory_roots_nonaliased(
                [
                    *alias_base_paths,
                    Path(str(slot["trace_root"])),
                    Path(str(slot["artifact_parent_root"])),
                ],
                f"{slot['slot_id']} final evidence",
            )
            source_after = _augment_source_replay_with_alias_rows(
                source_after,
                alias_final,
            )
        except BaseException as error:
            detail = (
                f"{getattr(error, 'code', type(error).__name__)}:"
                f"{getattr(error, 'detail', str(error))}"
            )[:4096]
            source_after = {
                "certain": False,
                "error_or_null": detail,
                "evidence": {"error": detail},
                "root": _domain_digest(
                    "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R8/SOURCE_REPLAY/V2",
                    {"error": detail},
                ),
            }

        future_slots = list(context["slots"])[ordinal:]
        future_slot_freshness = _slot_roots_absence_evidence(
            context,
            future_slots,
            label=f"{slot['slot_id']} future slots before outcome",
        )
        source_after_certain = (
            source_after["certain"] is True
            and future_slot_freshness["certain"] is True
        )
        cleanup_replay = _cleanup_replay_evidence(
            process_result,
            trajectory,
            docker_absent,
            docker_rows,
            docker_error,
        )
        source_before_public = deepcopy(dict(source_before))
        source_before_public.pop("clone_closure_rows_internal", None)
        binding = _binding_record(
            contract,
            capsule,
            slot,
            process_result,
            stdout_facts,
            trajectory,
            supervisor_artifact,
            external_artifact,
            docker_absent,
            docker_rows,
            docker_error,
            source_before_public,
            source_after,
            future_slot_freshness,
            cleanup_replay,
        )
        binding_path = Path(str(slot["external_trajectory_binding_path"]))
        if binding_path.parent != Path(str(slot["evidence_root"])):
            _fail(
                "FAIL_CAMPAIGN_BINDING",
                "external binding path relation differs",
                terminal=True,
            )
        external_binding_attempted = True
        _publish_immutable_at(
            evidence_fd,
            binding_path.name,
            binding,
            publication_label="slot-external-trajectory-binding",
            known_failure_code="FAIL_CAMPAIGN_BINDING",
        )
        _postpublication_directory_identity(
            Path(str(slot["evidence_root"])),
            evidence_fd,
            f"{slot['slot_id']} binding replay",
            evidence_identity,
        )
        final_future_slot_freshness = _slot_roots_absence_evidence(
            context,
            future_slots,
            label=f"{slot['slot_id']} future slots before outcome",
        )
        if final_future_slot_freshness != future_slot_freshness:
            _fail(
                "FAIL_CAMPAIGN_FRESHNESS",
                "future-slot freshness changed before outcome publication",
                terminal=True,
            )
        ledger.publish(
            f"{stem}.trajectory-binding.json",
            "EXTERNAL_TRAJECTORY_BINDING",
            str(trajectory["verification_status"]),
            binding,
        )
        if external_artifact["state"] == "VALID":
            try:
                final_external_artifact = _replay_external_artifact(slot)
            except BaseException as error:
                _raise_postpublication_uncertain(
                    f"{slot['slot_id']} external artifact final replay",
                    error,
                )
        elif external_artifact["state"] == "ABSENT":
            final_external_artifact = _absent_external_artifact(context, slot)
        else:
            final_external_artifact = external_artifact
        if final_external_artifact != external_artifact:
            _raise_postpublication_uncertain(
                f"{slot['slot_id']} external artifact final replay "
                "differs from durable binding"
            )
        observation = _build_slot_observation(
            contract,
            slot,
            process_result,
            stdout_facts,
            trajectory,
            supervisor_artifact,
            external_artifact,
            source_before_certain=bool(source_before["certain"]),
            source_after_certain=source_after_certain,
            docker_absent=(
                docker_absent is True
                and not docker_rows
                and docker_error is None
            ),
            external_binding_certain=True,
        )
        decision = core.evaluate_slot_observation_v2(observation)
        slot_outcome = _slot_outcome_record(
            contract,
            capsule,
            slot,
            binding,
            observation,
            decision,
        )
        result_path = Path(str(slot["slot_result_path"]))
        if result_path.parent != Path(str(slot["evidence_root"])):
            _fail(
                "FAIL_CAMPAIGN_OBSERVATION",
                "slot result path relation differs",
                terminal=True,
            )
        _publish_immutable_at(
            evidence_fd,
            result_path.name,
            slot_outcome,
            publication_label="slot-outcome",
            known_failure_code="FAIL_CAMPAIGN_OBSERVATION",
        )
        _postpublication_directory_identity(
            Path(str(slot["evidence_root"])),
            evidence_fd,
            f"{slot['slot_id']} outcome replay",
            evidence_identity,
        )
        if (
            _slot_roots_absence_evidence(
                context,
                future_slots,
                label=f"{slot['slot_id']} future slots before outcome",
            )
            != future_slot_freshness
        ):
            _fail(
                "FAIL_CAMPAIGN_FRESHNESS",
                "future-slot freshness changed before final outcome record",
                terminal=True,
            )
        ledger.publish(
            f"{stem}.outcome.json",
            "SLOT_OUTCOME",
            decision,
            slot_outcome,
        )
        return slot_outcome, decision
    except BaseException as error:
        if external_binding_attempted:
            _guard_publication_attempt_failure(
                f"{slot['slot_id']} evidence-chain continuation",
                binding_path.name,
                "FAIL_CAMPAIGN_BINDING",
                error,
            )
        raise
    finally:
        active_error = sys.exc_info()[1]
        descriptor = evidence_fd
        evidence_fd = -1
        try:
            os.close(descriptor)
        except BaseException as close_error:
            if (
                isinstance(active_error, ShadowRetryCampaignErrorV2)
                and active_error.code
                in {
                    "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN",
                    "FAIL_CAMPAIGN_SPAWN_UNCERTAIN",
                    _SCHEDULING_ADMISSION_FAILURE_V2,
                }
            ):
                raise active_error
            if external_binding_attempted:
                _raise_postpublication_uncertain(
                    f"{slot['slot_id']} evidence descriptor close",
                    close_error,
                    publication_basename_or_null=binding_path.name,
                )
            raise


def create_phase3_q05b_shadow_retry_authorization_v2(
    expected_capsule_digest: str,
) -> dict[str, object]:
    """Publish the exact immutable owner record after an external direction."""

    if type(expected_capsule_digest) is not str or _TAGGED_SHA256.fullmatch(
        expected_capsule_digest
    ) is None:
        _fail("FAIL_CAMPAIGN_AUTHORIZATION", "expected capsule digest differs")
    contract = load_phase3_q05b_shadow_retry_campaign_contract_v2()
    context = _preflight_context(contract)
    capsule = context["capsule"]
    if capsule["capsule_digest"] != expected_capsule_digest:
        _fail(
            "FAIL_CAMPAIGN_AUTHORIZATION",
            "owner-directed capsule digest differs from current preflight",
        )
    authorization = context["authorization_template"]
    root = Path(str(contract["fixed_external_roots"]["authorization_root"]))
    parent_fd, root_fd = _create_private_root(root, "owner authorization root")
    publication_attempted = False
    basename = str(contract["authorization_policy"]["authorization_record_name"])
    try:
        root_identity = _verify_held_directory_path(
            root,
            root_fd,
            "owner authorization root",
        )
        publication_attempted = True
        _publish_immutable_at(
            root_fd,
            basename,
            authorization,
            publication_label="owner-authorization-record",
            known_failure_code="FAIL_CAMPAIGN_AUTHORIZATION",
        )
        _postpublication_directory_identity(
            root,
            root_fd,
            "owner authorization root publication replay",
            root_identity,
        )
        result = {
            "schema_version": "hegel-phase3a-q05b-shadow-retry-authorization-receipt/2",
            "status": "AUTHORIZED_EXACT_CAPSULE_ONCE",
            "capsule_digest": capsule["capsule_digest"],
            "owner_confirmation_digest": authorization[
                "owner_confirmation_digest"
            ],
            "authorization_digest": authorization["authorization_digest"],
            "authorization_path": os.fspath(root / basename),
            "supervisor_invocation_count": 0,
        }
    except BaseException as error:
        if publication_attempted:
            _guard_publication_attempt_failure(
                "owner authorization publication continuation",
                basename,
                "FAIL_CAMPAIGN_AUTHORIZATION",
                error,
            )
        raise
    finally:
        active_error = sys.exc_info()[1]
        close_errors: list[tuple[str, BaseException]] = []
        for label, descriptor in (
            ("owner authorization root", root_fd),
            ("owner authorization parent", parent_fd),
        ):
            try:
                os.close(descriptor)
            except BaseException as close_error:
                close_errors.append((label, close_error))
        if close_errors:
            if (
                isinstance(active_error, ShadowRetryCampaignErrorV2)
                and active_error.code == "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN"
            ):
                raise active_error
            close_label, close_error = close_errors[0]
            if publication_attempted:
                _raise_postpublication_uncertain(
                    f"{close_label} descriptor close",
                    close_error,
                    publication_basename_or_null=basename,
                )
            raise close_error
    return result


def run_phase3_q05b_shadow_retry_campaign_v2(
    expected_capsule_digest: str,
) -> dict[str, object]:
    """Consume one exact authorization and execute at most three slots."""

    if type(expected_capsule_digest) is not str or _TAGGED_SHA256.fullmatch(
        expected_capsule_digest
    ) is None:
        _fail("FAIL_CAMPAIGN_AUTHORIZATION", "expected capsule digest differs")
    contract = load_phase3_q05b_shadow_retry_campaign_contract_v2()
    context = _preflight_context(contract)
    capsule = context["capsule"]
    if capsule["capsule_digest"] != expected_capsule_digest:
        _fail(
            "FAIL_CAMPAIGN_AUTHORIZATION",
            "execution capsule differs from exact owner direction",
        )
    authorization = _load_authorization(contract, capsule)
    ledger: _LedgerV2 | None = None
    invocation_counter = [0]
    last_slot_id: str | None = None
    last_outcome_digest: str | None = None
    try:
        ledger = _LedgerV2(contract, capsule, authorization)
        ledger.publish(
            "campaign.intent.json",
            "CAMPAIGN_INTENT",
            "AUTHORIZED_CAMPAIGN_INTENT_STAGED",
            {
                "capsule": deepcopy(dict(capsule)),
                "authorization_digest": authorization["authorization_digest"],
            },
        )
        ledger.publish(
            "campaign.authorization-consumed.json",
            "CAMPAIGN_AUTHORIZATION_CONSUMED",
            "CAMPAIGN_AUTHORIZATION_CONSUMED",
            {
                "authorization": deepcopy(dict(authorization)),
                "consumed_once": True,
            },
        )
        for slot in context["slots"]:
            slot_outcome, decision = _run_one_slot(
                contract,
                context,
                capsule,
                ledger,
                slot,
                invocation_counter,
            )
            last_slot_id = str(slot["slot_id"])
            last_outcome_digest = str(slot_outcome["record_digest"])
            core = _load_core(contract)
            if decision == core.SUCCESS_V2:
                result = _publish_terminal(
                    ledger,
                    contract,
                    capsule,
                    status_text=contract["success_policy"]["terminal_status"],
                    invocation_count=invocation_counter[0],
                    last_slot_id_or_null=last_slot_id,
                    last_slot_outcome_digest_or_null=last_outcome_digest,
                )
                return result
            if decision == core.RETRY_EXACT_GAP_V2:
                continue
            if decision == core.EXHAUSTED_RETRYABLE_GAP_V2:
                status_text = contract["terminal_failure_policy"][
                    "slot_03_exact_gap_disposition"
                ]
            else:
                status_text = _observation_terminal_disposition(
                    contract,
                    slot_outcome.get("observation"),
                )
            result = _publish_terminal(
                ledger,
                contract,
                capsule,
                status_text=status_text,
                invocation_count=invocation_counter[0],
                last_slot_id_or_null=last_slot_id,
                last_slot_outcome_digest_or_null=last_outcome_digest,
            )
            return result
        _fail(
            "FAIL_CAMPAIGN_STATE",
            "campaign slot loop ended without a terminal decision",
            terminal=True,
        )
    except BaseException as error:
        if ledger is not None:
            cached = _authoritative_completion_result_v2(ledger)
            if cached is not None:
                return cached
            recovered = _recover_completion_candidate_from_disk_v2(
                ledger,
                contract,
                capsule,
            )
            if recovered is not None:
                return recovered
        if (
            ledger is not None
            and "campaign.authorization-consumed.json"
            in ledger.published_names
            and "campaign.terminal.json" not in ledger.published_names
            and getattr(error, "code", None)
            not in {
                "FAIL_CAMPAIGN_LEDGER",
                "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN",
                "FAIL_CAMPAIGN_SPAWN_UNCERTAIN",
            }
        ):
            try:
                result = _publish_terminal(
                    ledger,
                    contract,
                    capsule,
                    status_text=_exception_terminal_disposition(
                        contract,
                        error,
                    ),
                    invocation_count=invocation_counter[0],
                    last_slot_id_or_null=last_slot_id,
                    last_slot_outcome_digest_or_null=last_outcome_digest,
                    error_code_or_null=str(
                        getattr(error, "code", type(error).__name__)
                    )[:256],
                    error_detail_or_null=str(
                        getattr(error, "detail", str(error))
                    )[:4096],
                    error_receipt_or_null=getattr(error, "receipt", None),
                )
                return result
            except BaseException as terminal_error:
                terminal_code = getattr(
                    terminal_error,
                    "code",
                    "FAIL_CAMPAIGN_LEDGER",
                )
                if terminal_code not in {
                    "FAIL_CAMPAIGN_LEDGER",
                    "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN",
                }:
                    terminal_code = "FAIL_CAMPAIGN_LEDGER"
                _fail(
                    str(terminal_code),
                    "terminal/completion publication failed while closing "
                    f"{getattr(error, 'code', type(error).__name__)}: "
                    f"{getattr(terminal_error, 'detail', str(terminal_error))}",
                    terminal=True,
                    receipt=_terminalization_failure_receipt_v2(
                        error,
                        terminal_error,
                    ),
                )
        raise
    finally:
        if ledger is not None:
            ledger.close()


def _require_exact_cli_runtime(contract: Mapping[str, object]) -> None:
    expected = str(contract["execution_policy"]["supervisor_interpreter"])
    if (
        os.path.realpath(sys.executable) != os.path.realpath(expected)
        or sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or sys.flags.dont_write_bytecode != 1
        or sys.flags.utf8_mode != 1
        or Path(__file__).resolve() != TOOL_PATH
    ):
        _fail(
            "FAIL_CAMPAIGN_RUNTIME",
            "CLI requires /usr/bin/python3.10 -I -S -B -X utf8",
        )


class _CampaignArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        _fail("FAIL_CAMPAIGN_ARGUMENT", message)


def _parser() -> argparse.ArgumentParser:
    parser = _CampaignArgumentParser(allow_abbrev=False)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--record-authorization", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--capsule-digest")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    try:
        options = _parser().parse_args(arguments)
        contract = load_phase3_q05b_shadow_retry_campaign_contract_v2()
        _require_exact_cli_runtime(contract)
        if options.preflight:
            if options.capsule_digest is not None:
                _fail(
                    "FAIL_CAMPAIGN_ARGUMENT",
                    "--preflight admits no capsule digest",
                )
            result = preflight_phase3_q05b_shadow_retry_campaign_v2()
            exit_code = 0
        elif options.record_authorization:
            if options.capsule_digest is None:
                _fail(
                    "FAIL_CAMPAIGN_ARGUMENT",
                    "--record-authorization requires --capsule-digest",
                )
            result = create_phase3_q05b_shadow_retry_authorization_v2(
                options.capsule_digest
            )
            exit_code = 0
        else:
            if options.capsule_digest is None:
                _fail(
                    "FAIL_CAMPAIGN_ARGUMENT",
                    "--run requires --capsule-digest",
                )
            result = run_phase3_q05b_shadow_retry_campaign_v2(
                options.capsule_digest
            )
            exit_code = (
                0
                if result["status"]
                == contract["success_policy"]["terminal_status"]
                else 1
            )
    except BaseException as error:
        error_code = str(
            getattr(error, "code", type(error).__name__)
        )[:256]
        result = {
            "schema_version": "hegel-phase3a-q05b-shadow-retry-campaign-cli-error/2",
            "status": (
                "CAMPAIGN_PUBLICATION_UNCERTAIN_TERMINAL_NO_RESUME"
                if error_code == "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN"
                else "CAMPAIGN_COMMAND_FAILED_CLOSED"
            ),
            "error_code": error_code,
            "error_detail": str(
                getattr(error, "detail", str(error))
            )[:4096],
            "error_receipt_or_null": (
                deepcopy(dict(getattr(error, "receipt")))
                if type(getattr(error, "receipt", None)) is dict
                else None
            ),
            "supervisor_invocation_count_or_null": None,
            "resumable": False,
        }
        exit_code = 1
    sys.stdout.buffer.write(_canonical_json_bytes(result))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
