#!/usr/bin/python3.10
"""Pure A8-context validator for the fixed attempt-3 recovery inputs.

The parent launches this file with ``/usr/bin/python3.10 -I -S -B -X
pycache_prefix=/nonexistent/hegel-r3-pycache``.  The validator then exposes only
the frozen main-worktree A8 package, the vendored ``tomli`` source, and a
hashed Ubuntu cryptography runtime closure.  It validates the two diagnostic
JSON reports and replays the transaction-local actor protocol bundle.  It
never opens custody, seed, key-volume, or M3 state.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Final, Mapping, NoReturn


A8_BASIS_COMMIT: Final = "0af65964235390ce2bebefea7379eaa9c50eda24"
FIXED_RUN_ID_HEX: Final = "e4af9f57c38fb298462ec628c4ed8a03"
FIXED_LEDGER_ID_HEX: Final = "ec849e2f1e2e1163cfc450370b25b484"
FORMAL_REPOSITORY_ROOT: Final = Path(
    "/home/erzhu419/mine_code/Asumption Agent"
)
FORMAL_PROJECT_ROOT: Final = FORMAL_REPOSITORY_ROOT / "Hegel Machine"
FORMAL_SOURCE_ROOT: Final = FORMAL_PROJECT_ROOT / "src"
FIXED_PYTHON_EXECUTABLE: Final = Path("/usr/bin/python3.10")
FIXED_PYTHON_EXECUTABLE_SHA256: Final = (
    "7d51cd6b48b521277f5caa4610a82126e315fa2be4df069823a8b1eeb5bd4a86"
)
EXPECTED_A8_IMPORT_CLOSURE_SHA256_ROOT: Final = (
    "d071923c4f926104d78ef082f36aec66ff33221a530ad68a9bdf7cfe3f644d77"
)
VENDORED_DEPENDENCY_PARENT: Final = (
    Path(__file__).resolve().parents[1] / "vendor/phase3_m25_a8_r3"
)
VENDORED_TOMLI_ROOT: Final = VENDORED_DEPENDENCY_PARENT / "tomli"
VENDORED_TOMLI_VERSION: Final = "2.4.1"
VENDORED_TOMLI_SHA256: Final = {
    "LICENSE": "b80816b0d530b8accb4c2211783790984a6e3b61922c2b5ee92f3372ab2742fe",
    "__init__.py": "9eb042d7c0db5d14c2168ec4946e410de5a91c9cce86892f5e4db5e4633c6762",
    "_parser.py": "a412234c86bf710b361e0943276961f0e25fa6d7c36ba7a0e7eec87a3e018c7b",
    "_re.py": "a12359fe294523a72112e434d58452a14c9d050affa2417f9927474e4166bfdd",
    "_types.py": "f864c6d9552a929c7032ace654ee05ef26ca75d21b027b801d77e65907138b74",
    "py.typed": "f0f8f2675695a10a5156fb7bd66bafbaae6a13e8d315990af862c792175e6e67",
}
VENDORED_TOMLI_SHA256_ROOT: Final = (
    "09cf56112923cfb69bd47cafc72627dafebf2c479a223e4ce4e7b7eaf53e8bed"
)
FIXED_PYCACHE_PREFIX: Final = "/nonexistent/hegel-r3-pycache"
SYSTEM_DIST_PACKAGES_ROOT: Final = Path("/usr/lib/python3/dist-packages")
SYSTEM_CRYPTOGRAPHY_ROOT: Final = SYSTEM_DIST_PACKAGES_ROOT / "cryptography"
SYSTEM_BCRYPT_ROOT: Final = SYSTEM_DIST_PACKAGES_ROOT / "bcrypt"
EXPECTED_SYSTEM_CRYPTOGRAPHY_TREE_ROOT: Final = (
    "d31d17dd632b5e9462a08d7080385eccd3aae6e20189731a172cff31fd91ff2f"
)
EXPECTED_SYSTEM_BCRYPT_TREE_ROOT: Final = (
    "7d39c1a1f70fa30fd78f8428e6f5f8157ec13009ebc000f9171224bc932bedc0"
)
SYSTEM_DEPENDENCY_FILES: Final = {
    "/usr/lib/python3/dist-packages/_cffi_backend.cpython-310-x86_64-linux-gnu.so": (
        "1e3dcc3e5f0e3f2d9a897c1dff1ddfbac26d50e7fdd7f88b16bdf57a0101a214"
    ),
    "/usr/lib/python3/dist-packages/six.py": (
        "4ce39f422ee71467ccac8bed76beb05f8c321c7f0ceda9279ae2dfa3670106b3"
    ),
    "/usr/lib/x86_64-linux-gnu/libssl.so.3": (
        "03e9019df86b0d66d8f1686c9472b992c0c9196d3f2e2e8f282b37fa55f6ee7b"
    ),
    "/usr/lib/x86_64-linux-gnu/libcrypto.so.3": (
        "56801abd67bd45ca0473a71a22479dc6b044934c99ee1418fccb48e08bd183ea"
    ),
    "/usr/lib/x86_64-linux-gnu/libc.so.6": (
        "e01b1ce7be2987f3b8560e26d0df2623f9dd5cec17be923ae28a785bc0d32d50"
    ),
    "/usr/lib/x86_64-linux-gnu/libffi.so.8.1.0": (
        "247da4d5d34a91cadcdd6282be4c4644fcb8af001334d2b8a82ecda435418cbf"
    ),
    "/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2": (
        "8d06f393f4a93bcf9b81145a259524d66a95522a646bf8d7e05b6ffdf2e63dcc"
    ),
}
EXPECTED_VALIDATOR_DEPENDENCY_CLOSURE_ROOT: Final = (
    "f39b2f922af5723ee50374b4f04be5c6525a58a87e19de9376d2525a108d1dc7"
)
SCHEMA: Final = "hegel-phase3-m25-a8-r3-validation-request/1"
RECEIPT_SCHEMA: Final = "hegel-phase3-m25-a8-r3-a8-validation-receipt/1"
MAX_REQUEST_BYTES: Final = 4 * 1024 * 1024
HEX_64 = re.compile(r"[0-9a-f]{64}")


class A8RecoveryValidationError(RuntimeError):
    pass


def _fail(detail: str) -> NoReturn:
    raise A8RecoveryValidationError(detail)


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _read_request() -> dict[str, object]:
    payload = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    if not payload or len(payload) > MAX_REQUEST_BYTES:
        _fail("validation request size differs")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"validation request is not canonical JSON: {type(exc).__name__}")
    if type(value) is not dict or payload != _canonical_json(value):
        _fail("validation request is not one canonical JSON object")
    return value


def _require_formal_a8_context() -> None:
    executable = Path("/proc/self/exe").resolve(strict=True)
    executable_metadata = executable.stat()
    forbidden_startup_modules = {
        "site",
        "sitecustomize",
        "usercustomize",
        "apport_python_hook",
        "zope",
    }
    if (
        executable != FIXED_PYTHON_EXECUTABLE
        or not stat.S_ISREG(executable_metadata.st_mode)
        or stat.S_IMODE(executable_metadata.st_mode) != 0o755
        or executable_metadata.st_uid != 0
        or executable_metadata.st_gid != 0
        or hashlib.sha256(executable.read_bytes()).hexdigest()
        != FIXED_PYTHON_EXECUTABLE_SHA256
        or sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or sys.flags.dont_write_bytecode != 1
        or sys.pycache_prefix != FIXED_PYCACHE_PREFIX
        or sys.path[0:1] == [""]
        or forbidden_startup_modules.intersection(sys.modules)
        or any(
            "site-packages" in entry or "dist-packages" in entry
            for entry in sys.path
        )
        or FORMAL_REPOSITORY_ROOT.is_symlink()
        or FORMAL_PROJECT_ROOT.is_symlink()
        or FORMAL_SOURCE_ROOT.is_symlink()
        or not FORMAL_SOURCE_ROOT.is_dir()
        or FORMAL_REPOSITORY_ROOT.resolve(strict=True) != FORMAL_REPOSITORY_ROOT
    ):
        _fail("formal A8 source path identity differs")
    completed = subprocess.run(
        ["/usr/bin/git", "rev-parse", "--verify", "HEAD^{commit}"],
        cwd=FORMAL_REPOSITORY_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
        env={
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_PROTOCOL_FROM_USER": "0",
            "GIT_TERMINAL_PROMPT": "0",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
        },
    )
    if (
        completed.returncode != 0
        or completed.stderr
        or completed.stdout.decode("ascii", "strict").strip()
        != A8_BASIS_COMMIT
    ):
        _fail("formal main worktree HEAD is not the fixed A8 commit")


def _git(arguments: tuple[str, ...]) -> bytes:
    completed = subprocess.run(
        ["/usr/bin/git", *arguments],
        cwd=FORMAL_REPOSITORY_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=60,
        env={
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_PROTOCOL_FROM_USER": "0",
            "GIT_TERMINAL_PROMPT": "0",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
        },
    )
    if completed.returncode != 0:
        _fail("formal A8 Git blob verification failed")
    return completed.stdout


def _verify_a8_import_closure() -> str:
    """Bind every possible package import and direct admission input to A8.

    This check runs before and after imports/validation.  It prevents a dirty
    main worktree module from validating itself merely because HEAD still
    names A8.  The package's top-level Python path set is also exact, so an
    untracked shadow module cannot enter the isolated interpreter.
    """

    prefixes = (
        "Hegel Machine/src/hegel_machine",
        "Hegel Machine/config",
        "Hegel Machine/tools",
        "Hegel Machine/rust/formal_bridge_m25",
        "Hegel Machine/rust/m25_bridge_dag_replay",
    )
    listing = _git(
        (
            "ls-tree", "-r", "-z", "--full-tree", A8_BASIS_COMMIT,
            "--", *prefixes,
        )
    )
    rows: list[tuple[str, str]] = []
    expected_package_python: set[str] = set()
    aggregate = hashlib.sha256(b"hegel-m25-a8-r3-import-closure-v1\0")
    for raw_row in listing.split(b"\0"):
        if not raw_row:
            continue
        try:
            metadata, path_raw = raw_row.split(b"\t", 1)
            mode, kind, object_id = metadata.decode("ascii", "strict").split(" ")
            relative = path_raw.decode("utf-8", "strict")
        except (UnicodeDecodeError, ValueError):
            _fail("formal A8 Git tree row is malformed")
        if mode not in {"100644", "100755"} or kind != "blob":
            _fail("formal A8 validation closure contains a non-regular blob")
        path = FORMAL_REPOSITORY_ROOT / relative
        if path.is_symlink() or not path.is_file():
            _fail(f"formal A8 validation input is absent or linked: {relative}")
        current = path.read_bytes()
        frozen = _git(("show", f"{A8_BASIS_COMMIT}:{relative}"))
        if current != frozen:
            _fail(f"formal A8 validation input differs from Git: {relative}")
        digest = hashlib.sha256(current).hexdigest()
        rows.append((relative, digest))
        encoded = relative.encode("utf-8")
        aggregate.update(len(encoded).to_bytes(4, "big"))
        aggregate.update(encoded)
        aggregate.update(bytes.fromhex(digest))
        pure = Path(relative)
        if (
            pure.parent.as_posix() == "Hegel Machine/src/hegel_machine"
            and pure.suffix == ".py"
        ):
            expected_package_python.add(relative)
    package_root = FORMAL_SOURCE_ROOT / "hegel_machine"
    observed_package_python = {
        path.relative_to(FORMAL_REPOSITORY_ROOT).as_posix()
        for path in package_root.glob("*.py")
        if path.is_file() or path.is_symlink()
    }
    expected_entry_names = {
        Path(relative).name for relative in expected_package_python
    }
    observed_entry_names = {path.name for path in package_root.iterdir()}
    cache_path = package_root / "__pycache__"
    extra_entry_names = observed_entry_names - expected_entry_names
    if (
        not rows
        or observed_package_python != expected_package_python
        or extra_entry_names not in (set(), {"__pycache__"})
        or (
            "__pycache__" in extra_entry_names
            and (cache_path.is_symlink() or not cache_path.is_dir())
        )
    ):
        _fail("formal A8 top-level package path set differs")
    root = aggregate.hexdigest()
    if root != EXPECTED_A8_IMPORT_CLOSURE_SHA256_ROOT:
        _fail("formal A8 import closure root differs")
    return root


def _verify_vendored_tomli_v1() -> str:
    """Verify the vendored Python-3.10 TOML parser made visible to ``-S``."""

    if (
        VENDORED_DEPENDENCY_PARENT.is_symlink()
        or {path.name for path in VENDORED_DEPENDENCY_PARENT.iterdir()}
        != {"tomli"}
        or VENDORED_TOMLI_ROOT.is_symlink()
        or not VENDORED_TOMLI_ROOT.is_dir()
        or VENDORED_TOMLI_ROOT.resolve(strict=True) != VENDORED_TOMLI_ROOT
        or {path.name for path in VENDORED_TOMLI_ROOT.iterdir()}
        != set(VENDORED_TOMLI_SHA256)
    ):
        _fail("vendored tomli path set differs")
    observed: dict[str, str] = {}
    for name, expected_sha256 in VENDORED_TOMLI_SHA256.items():
        path = VENDORED_TOMLI_ROOT / name
        if path.is_symlink():
            _fail(f"vendored tomli input is linked: {name}")
        metadata = path.stat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o644
            or metadata.st_uid != os.getuid()
            or metadata.st_gid != os.getgid()
        ):
            _fail(f"vendored tomli metadata differs: {name}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_sha256:
            _fail(f"vendored tomli content differs: {name}")
        observed[name] = digest
    root = hashlib.sha256(_canonical_json(observed)).hexdigest()
    if root != VENDORED_TOMLI_SHA256_ROOT:
        _fail("vendored tomli closure root differs")
    return root


def _system_package_tree_root_v1(
    root: Path, *, domain: bytes, expected_count: int
) -> str:
    if (
        root.is_symlink()
        or not root.is_dir()
        or root.resolve(strict=True) != root
        or root.stat().st_uid != 0
        or root.stat().st_gid != 0
        or stat.S_IMODE(root.stat().st_mode) & 0o022
    ):
        _fail(f"system dependency root metadata differs: {root}")
    aggregate = hashlib.sha256(domain + b"\0")
    count = 0
    for path in sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    ):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts:
            continue
        if path.is_symlink():
            _fail(f"system dependency path is linked: {path}")
        if path.is_dir():
            metadata = path.stat()
            if (
                metadata.st_uid != 0
                or metadata.st_gid != 0
                or stat.S_IMODE(metadata.st_mode) & 0o022
            ):
                _fail(f"system dependency directory metadata differs: {path}")
            continue
        metadata = path.stat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != 0
            or metadata.st_gid != 0
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            _fail(f"system dependency file metadata differs: {path}")
        digest = hashlib.sha256(path.read_bytes()).digest()
        encoded = relative.as_posix().encode("utf-8")
        aggregate.update(len(encoded).to_bytes(4, "big"))
        aggregate.update(encoded)
        aggregate.update(digest)
        count += 1
    if count != expected_count:
        _fail(f"system dependency file count differs: {root}")
    return aggregate.hexdigest()


def _verify_system_crypto_dependency_closure_v1() -> str:
    if (
        SYSTEM_DIST_PACKAGES_ROOT.is_symlink()
        or not SYSTEM_DIST_PACKAGES_ROOT.is_dir()
        or SYSTEM_DIST_PACKAGES_ROOT.resolve(strict=True)
        != SYSTEM_DIST_PACKAGES_ROOT
    ):
        _fail("system dist-packages identity differs")
    cryptography_root = _system_package_tree_root_v1(
        SYSTEM_CRYPTOGRAPHY_ROOT,
        domain=b"hegel-r3-system-cryptography-tree-v1",
        expected_count=90,
    )
    bcrypt_root = _system_package_tree_root_v1(
        SYSTEM_BCRYPT_ROOT,
        domain=b"hegel-r3-system-bcrypt-tree-v1",
        expected_count=4,
    )
    if (
        cryptography_root != EXPECTED_SYSTEM_CRYPTOGRAPHY_TREE_ROOT
        or bcrypt_root != EXPECTED_SYSTEM_BCRYPT_TREE_ROOT
    ):
        _fail("system cryptography package tree root differs")
    records = {
        "vendored_tomli_tree": VENDORED_TOMLI_SHA256_ROOT,
        "system_cryptography_tree": cryptography_root,
        "system_bcrypt_tree": bcrypt_root,
    }
    for raw_path, expected_digest in SYSTEM_DEPENDENCY_FILES.items():
        path = Path(raw_path)
        if path.is_symlink():
            _fail(f"system dependency leaf is linked: {raw_path}")
        metadata = path.stat()
        if (
            path.resolve(strict=True) != path
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != 0
            or metadata.st_gid != 0
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            _fail(f"system dependency leaf metadata differs: {raw_path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_digest:
            _fail(f"system dependency leaf content differs: {raw_path}")
        records[raw_path] = digest
    root = hashlib.sha256(_canonical_json(records)).hexdigest()
    if root != EXPECTED_VALIDATOR_DEPENDENCY_CLOSURE_ROOT:
        _fail("validator dependency closure root differs")
    return root


def _verify_loaded_dependency_modules_v1() -> None:
    required_modules = {"tomli", "cryptography", "bcrypt", "_cffi_backend", "six"}
    if not required_modules.issubset(sys.modules):
        _fail("validator dependency module set is incomplete")
    for name, module in tuple(sys.modules.items()):
        module_file = getattr(module, "__file__", None)
        if not module_file:
            continue
        try:
            path = Path(str(module_file)).resolve(strict=True)
        except OSError:
            _fail(f"loaded module path cannot be resolved: {name}")
        if SYSTEM_DIST_PACKAGES_ROOT in path.parents:
            allowed = (
                SYSTEM_CRYPTOGRAPHY_ROOT in path.parents
                or SYSTEM_BCRYPT_ROOT in path.parents
                or path
                in {
                    Path(
                        "/usr/lib/python3/dist-packages/"
                        "_cffi_backend.cpython-310-x86_64-linux-gnu.so"
                    ),
                    Path("/usr/lib/python3/dist-packages/six.py"),
                }
            )
            if not allowed:
                _fail(f"unfrozen dist-packages module was loaded: {name}")
        elif VENDORED_DEPENDENCY_PARENT in path.parents:
            if VENDORED_TOMLI_ROOT not in path.parents:
                _fail(f"unfrozen vendored module was loaded: {name}")
        elif FORMAL_SOURCE_ROOT in path.parents:
            continue
    if {"site", "sitecustomize", "usercustomize", "apport_python_hook", "zope"}.intersection(
        sys.modules
    ):
        _fail("automatic site startup module entered the isolated validator")


def _verify_crypto_runtime_mappings_v1() -> None:
    mappings = Path("/proc/self/maps").read_text(encoding="utf-8")
    required = (
        "/usr/lib/x86_64-linux-gnu/libssl.so.3",
        "/usr/lib/x86_64-linux-gnu/libcrypto.so.3",
        "/usr/lib/x86_64-linux-gnu/libc.so.6",
        "/usr/lib/x86_64-linux-gnu/libffi.so.8.1.0",
        "/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2",
    )
    if any(path not in mappings for path in required):
        _fail("validator native cryptography runtime mapping differs")


def _string_keyed_json_object(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict or not all(type(key) is str for key in value):
        _fail(f"{label} is not a strict JSON object")
    return dict(value)


def _sha256_field(value: object, label: str) -> str:
    if type(value) is not str or HEX_64.fullmatch(value) is None:
        _fail(f"{label} is not lowercase SHA-256")
    return value


def _validate() -> dict[str, object]:
    request = _read_request()
    expected_keys = {
        "schema",
        "basis_commit",
        "run_id_hex",
        "ledger_id_hex",
        "actor_qualification_report",
        "actor_report_sha256",
        "errata_qualification_report",
        "errata_report_sha256",
        "live_actor_protocol_qualification_bundle",
        "live_bundle_sha256",
        "expected_live_bundle_content_id_hex",
        "expected_qualification_key_id_rows",
        "contains_raw_seed",
        "contains_private_key",
        "m3_start_allowed",
    }
    if set(request) != expected_keys:
        _fail("validation request field set differs")
    if (
        request.get("schema") != SCHEMA
        or request.get("basis_commit") != A8_BASIS_COMMIT
        or request.get("run_id_hex") != FIXED_RUN_ID_HEX
        or request.get("ledger_id_hex") != FIXED_LEDGER_ID_HEX
        or request.get("contains_raw_seed") is not False
        or request.get("contains_private_key") is not False
        or request.get("m3_start_allowed") is not False
    ):
        _fail("validation request identity or authority differs")
    actor = _string_keyed_json_object(
        request.get("actor_qualification_report"), "actor report"
    )
    errata = _string_keyed_json_object(
        request.get("errata_qualification_report"), "errata report"
    )
    bundle = _string_keyed_json_object(
        request.get("live_actor_protocol_qualification_bundle"),
        "live actor protocol bundle",
    )
    actor_sha256 = _sha256_field(
        request.get("actor_report_sha256"), "actor report digest"
    )
    errata_sha256 = _sha256_field(
        request.get("errata_report_sha256"), "errata report digest"
    )
    bundle_sha256 = _sha256_field(
        request.get("live_bundle_sha256"), "live bundle digest"
    )
    if (
        hashlib.sha256(_canonical_json(actor)).hexdigest() != actor_sha256
        or hashlib.sha256(_canonical_json(errata)).hexdigest() != errata_sha256
        or hashlib.sha256(_canonical_json(bundle)).hexdigest() != bundle_sha256
    ):
        _fail("validation request report or bundle digest differs")
    content_id_hex = request.get("expected_live_bundle_content_id_hex")
    rows = request.get("expected_qualification_key_id_rows")
    if (
        type(content_id_hex) is not str
        or HEX_64.fullmatch(content_id_hex) is None
        or type(rows) is not list
        or len(rows) != 4
    ):
        _fail("expected actor-protocol identity is malformed")
    expected_key_ids: dict[int, str] = {}
    for ordinal, row in enumerate(rows, start=1):
        if (
            type(row) is not dict
            or set(row) != {"purpose_id", "qualification_only_key_id_hex"}
            or row.get("purpose_id") != ordinal
            or type(row.get("qualification_only_key_id_hex")) is not str
            or re.fullmatch(
                r"[0-9a-f]{32}", str(row.get("qualification_only_key_id_hex"))
            )
            is None
        ):
            _fail("expected qualification key row differs")
        expected_key_ids[ordinal] = str(row["qualification_only_key_id_hex"])

    _require_formal_a8_context()
    import_closure_root = _verify_a8_import_closure()
    _verify_vendored_tomli_v1()
    dependency_closure_root = _verify_system_crypto_dependency_closure_v1()
    # -I -S excludes the caller path, all site initialization, and every
    # site/dist-packages path.  Expose only the three frozen roots below after
    # their source/native closures have been checked.
    # Preserve the fixed interpreter's stdlib paths ahead of all added roots,
    # then place vendored and system dependencies ahead of the A8 package
    # parent so an untracked top-level file under ``src`` cannot shadow them.
    sys.path.append(VENDORED_DEPENDENCY_PARENT.as_posix())
    sys.path.append(SYSTEM_DIST_PACKAGES_ROOT.as_posix())
    sys.path.append(FORMAL_SOURCE_ROOT.as_posix())
    import tomli as tomli_module  # noqa: PLC0415
    import hegel_machine as package_module  # noqa: PLC0415
    from hegel_machine import phase3_m25_container_ceremony_v1 as ceremony_module  # noqa: PLC0415
    from hegel_machine import phase3_m25_formal_container_executor_v1 as executor_module  # noqa: PLC0415
    for module in (package_module, ceremony_module, executor_module):
        module_path = Path(str(module.__file__)).resolve(strict=True)
        if FORMAL_SOURCE_ROOT not in module_path.parents:
            _fail("isolated validator imported a module outside fixed main A8")
    if (
        Path(str(tomli_module.__file__)).resolve(strict=True)
        != VENDORED_TOMLI_ROOT / "__init__.py"
        or tomli_module.__version__ != VENDORED_TOMLI_VERSION
    ):
        _fail("isolated validator imported a foreign tomli package")

    admission = ceremony_module.validate_ceremony_admission_v1(
        actor_qualification_report=actor,
        errata_qualification_report=errata,
        basis_commit=A8_BASIS_COMMIT,
        committed_input_paths=executor_module.REQUIRED_COMMIT_A_INPUTS,
    )
    replayed = executor_module.replay_transaction_local_actor_protocol_bundle_v1(
        basis_commit=A8_BASIS_COMMIT,
        bundle=bundle,
    )
    replayed_key_ids = {
        int(purpose): value.hex()
        for purpose, value in replayed.qualification_key_ids.items()
    }
    if (
        replayed.bundle_content_id.hex() != content_id_hex
        or replayed_key_ids != expected_key_ids
        or admission.get("basis_commit") != A8_BASIS_COMMIT
        or actor.get("technical_actor_eligible") is not True
        or actor.get("basis_commit") != A8_BASIS_COMMIT
        or errata.get("implementation_basis_commit") != A8_BASIS_COMMIT
    ):
        _fail("A8 report or transaction-bundle identity differs")
    input_sha256 = admission.get("input_sha256")
    if type(input_sha256) is not dict or not input_sha256:
        _fail("A8 ceremony admission input binding is absent")
    if _verify_a8_import_closure() != import_closure_root:
        _fail("formal A8 import closure changed during isolated validation")
    _verify_loaded_dependency_modules_v1()
    _verify_crypto_runtime_mappings_v1()
    if _verify_vendored_tomli_v1() != VENDORED_TOMLI_SHA256_ROOT:
        _fail("vendored tomli closure changed during isolated validation")
    if _verify_system_crypto_dependency_closure_v1() != dependency_closure_root:
        _fail("system cryptography closure changed during isolated validation")
    receipt: dict[str, object] = {
        "schema": RECEIPT_SCHEMA,
        "formal_repository_root": FORMAL_REPOSITORY_ROOT.as_posix(),
        "formal_repository_path_sha256": hashlib.sha256(
            FORMAL_REPOSITORY_ROOT.as_posix().encode("utf-8")
        ).hexdigest(),
        "formal_repository_commit": A8_BASIS_COMMIT,
        "python_executable": FIXED_PYTHON_EXECUTABLE.as_posix(),
        "python_executable_sha256": FIXED_PYTHON_EXECUTABLE_SHA256,
        "python_isolated": True,
        "python_no_site": True,
        "python_bytecode_disabled": True,
        "python_pycache_prefix": FIXED_PYCACHE_PREFIX,
        "a8_import_closure_sha256_root": import_closure_root,
        "a8_validator_dependency_closure_sha256_root": dependency_closure_root,
        "run_id_hex": FIXED_RUN_ID_HEX,
        "ledger_id_hex": FIXED_LEDGER_ID_HEX,
        "actor_report_sha256": actor_sha256,
        "errata_report_sha256": errata_sha256,
        "live_bundle_sha256": bundle_sha256,
        "live_bundle_content_id_hex": content_id_hex,
        "qualification_key_id_rows": rows,
        "commit_a_input_sha256": input_sha256,
        "commit_a_input_sha256_root": hashlib.sha256(
            _canonical_json(input_sha256)
        ).hexdigest(),
        "commit_a_input_count": len(input_sha256),
        "actor_technical_eligible": True,
        "errata_status": errata.get("status"),
        "transaction_bundle_replay_passed": True,
        "formal_identity_entropy_draw_count": 0,
        "contains_raw_seed": False,
        "contains_private_key": False,
        "raw_seed_bytes_read": False,
        "raw_seed_sha256_computed": False,
        "m3_start_invoked": False,
    }
    receipt["receipt_sha256"] = hashlib.sha256(_canonical_json(receipt)).hexdigest()
    return receipt


def main() -> int:
    try:
        receipt = _validate()
    except Exception as exc:
        sys.stderr.buffer.write(
            _canonical_json(
                {
                    "ok": False,
                    "error_code": "FAIL_M25_A8_R3_A8_VALIDATION",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )
        )
        return 2
    sys.stdout.buffer.write(_canonical_json(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
