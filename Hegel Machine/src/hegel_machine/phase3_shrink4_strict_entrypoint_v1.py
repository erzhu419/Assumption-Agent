"""Direct target-free single-vector endpoint for shrink-4 qualification."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import ModuleType


if __package__ not in {None, ""}:
    raise RuntimeError("shrink-4 strict replay requires its direct file entrypoint")
if sys.flags.isolated != 1 or sys.flags.no_site != 1 or not sys.dont_write_bytecode:
    raise RuntimeError("shrink-4 strict replay requires python -I -S -B")
if len(sys.argv) != 3 or sys.argv[1] not in {
    "--source-json",
    "--formal-cbor-hex",
}:
    raise RuntimeError(
        "usage: phase3_shrink4_strict_entrypoint_v1.py "
        "(--source-json JSON | --formal-cbor-hex HEX)"
    )
_MODE, _PAYLOAD = sys.argv[1], sys.argv[2]

package = ModuleType("hegel_machine")
package.__path__ = [str(Path(__file__).resolve().parent)]  # type: ignore[attr-defined]
package.__package__ = "hegel_machine"
sys.modules["hegel_machine"] = package
__package__ = "hegel_machine"

from .phase3_m3_shrink4_core_v1 import (  # noqa: E402
    DSL_VERSION,
    FREEZE_VERSION,
    MAX_TOP_LEVEL_CLAUSES,
)
from .strict_ast_shrink4_v1 import (  # noqa: E402
    canonicalize_shrink4_source_ast,
    decode_shrink4_canonical_ast,
)
from .strict_ast_v1 import StrictAstError  # noqa: E402
from .strict_cbor_v1 import StrictCborError  # noqa: E402


_ALLOWED_PROJECT_MODULES = {
    "hegel_machine.phase3_m3_dsl_core_v1",
    "hegel_machine.phase3_m3_shrink1_core_v1",
    "hegel_machine.phase3_m3_shrink2_core_v1",
    "hegel_machine.phase3_m3_shrink3_core_v1",
    "hegel_machine.phase3_m3_shrink4_core_v1",
    "hegel_machine.strict_ast_shrink1_v1",
    "hegel_machine.strict_ast_shrink2_v1",
    "hegel_machine.strict_ast_shrink3_v1",
    "hegel_machine.strict_ast_shrink4_v1",
    "hegel_machine.strict_ast_v1",
    "hegel_machine.strict_cbor_v1",
}


def _loaded() -> tuple[str, ...]:
    return tuple(sorted(name for name in sys.modules if name.startswith("hegel_machine.")))


def _assert_target_free() -> tuple[str, ...]:
    loaded = _loaded()
    unexpected = set(loaded) - _ALLOWED_PROJECT_MODULES
    if unexpected or set(loaded) != _ALLOWED_PROJECT_MODULES:
        raise RuntimeError(
            "target-free module closure violation: "
            f"missing={sorted(_ALLOWED_PROJECT_MODULES - set(loaded))!r}; "
            f"unexpected={sorted(unexpected)!r}"
        )
    return loaded


def replay_one() -> dict[str, object]:
    _assert_target_free()
    boundary = "SOURCE_JSON" if _MODE == "--source-json" else "FORMAL_CBOR"
    try:
        if _MODE == "--source-json":
            program = canonicalize_shrink4_source_ast(json.loads(_PAYLOAD))
        else:
            try:
                payload = bytes.fromhex(_PAYLOAD)
            except ValueError as error:
                raise RuntimeError("formal CBOR input must be even-length hex") from error
            program = decode_shrink4_canonical_ast(payload)
    except (StrictAstError, StrictCborError) as error:
        result: dict[str, object] = {
            "status": "REJECTED",
            "error_code": error.code,
            "error_detail": error.detail,
        }
    else:
        result = {
            "status": "ACCEPTED",
            "canonical_cbor_hex": program.cbor_bytes.hex(),
            "canonical_ast_hash": program.hash_id,
            "root_operator_id": program.root_operator_id,
            "output_sort": program.metrics.output_sort,
            "depth": program.metrics.depth,
            "node_count": program.metrics.node_count,
        }
    loaded = _assert_target_free()
    return {
        "schema_version": "hegel-strict-canonicalizer-shrink4-replay/1",
        "implementation": "python",
        "dsl_version": DSL_VERSION,
        "freeze_version": FREEZE_VERSION,
        "boundary": boundary,
        "maximum_top_level_clauses": MAX_TOP_LEVEL_CLAUSES,
        **result,
        "loaded_hegel_modules": list(loaded),
        "target_or_split_modules_loaded": False,
    }


if __name__ == "__main__":
    sys.stdout.write(
        json.dumps(replay_one(), sort_keys=True, separators=(",", ":")) + "\n"
    )
