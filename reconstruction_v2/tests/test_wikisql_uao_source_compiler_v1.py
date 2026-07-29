from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import stat
import tarfile
import tempfile
from typing import Any, Mapping

import pytest

from assumption_agent.benchmarks import wikisql_uao_reality_v1 as reality
from assumption_agent.benchmarks import wikisql_uao_source_compiler_v1 as compiler


SECRET = b"k" * reality.HMAC_SECRET_BYTES
FAMILY_OPERATOR = {"EQ": 0, "GT": 1, "LT": 2}
FAMILY_THRESHOLD = {"EQ": 5, "GT": 8, "LT": 2}


@dataclass(frozen=True)
class SyntheticArchive:
    path: Path
    sha256: str
    table_ids: tuple[str, ...]
    train_table_ids: tuple[str, ...]
    test_table_ids: tuple[str, ...]


def _rows(count: int = 11) -> list[list[object]]:
    return [[f"name-{index}", index] for index in range(count)]


def _table(table_id: str, *, row_count: int = 11) -> dict[str, object]:
    return {
        "id": table_id,
        "header": ["Name", "Score"],
        "types": ["text", "real"],
        "rows": _rows(row_count),
    }


def _question(
    table_id: str,
    *,
    family: str,
    phase: int = 1,
    question: str | None = None,
    conditions: list[list[object]] | None = None,
) -> dict[str, object]:
    threshold = FAMILY_THRESHOLD[family]
    return {
        "phase": phase,
        "question": question
        or f"Which Name has Score {family} {threshold} for example {phase}?",
        "sql": {
            "sel": 0,
            "agg": (phase - 1) % 6,
            "conds": conditions
            if conditions is not None
            else [[1, FAMILY_OPERATOR[family], threshold]],
        },
        "table_id": table_id,
    }


def _jsonl(rows: list[Mapping[str, object]]) -> bytes:
    return (
        "\n".join(
            json.dumps(
                row,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            for row in rows
        )
        + "\n"
    ).encode("ascii")


def _sqlite_bytes(
    path: Path,
    tables: list[Mapping[str, object]],
    *,
    mismatch_first_eq: bool,
    permute_first_table: bool = False,
) -> bytes:
    connection = sqlite3.connect(path)
    try:
        for table_index, table in enumerate(tables):
            table_id = table["id"]
            assert isinstance(table_id, str)
            name = "table_" + table_id.replace("-", "_")
            connection.execute(
                f'CREATE TABLE "{name}" (col0 TEXT, col1 REAL)'
            )
            rows = [list(row) for row in table["rows"]]  # type: ignore[index]
            if mismatch_first_eq and table_index == 0:
                rows[5][1] = 500
            if permute_first_table and table_index == 0:
                rows = list(reversed(rows))
            connection.executemany(
                f'INSERT INTO "{name}" (col0, col1) VALUES (?, ?)',
                [(str(row[0]).lower(), row[1]) for row in rows],
            )
        connection.commit()
    finally:
        connection.close()
    raw = path.read_bytes()
    assert raw.startswith(b"SQLite format 3\x00")
    return raw


def _add_member(
    bundle: tarfile.TarFile,
    name: str,
    raw: bytes,
) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(raw)
    member.mode = 0o600
    bundle.addfile(member, io.BytesIO(raw))


def _build_archive(
    root: Path,
    *,
    name: str = "source",
    unsafe_member: bool = False,
    item_schema_extra: bool = False,
    sqlite_mismatch: bool = False,
    sqlite_permuted: bool = False,
) -> SyntheticArchive:
    split_tables: dict[str, list[dict[str, object]]] = {
        "train": [],
        "test": [],
    }
    split_questions: dict[str, list[dict[str, object]]] = {
        "train": [],
        "test": [],
    }
    quotas = {"train": 4, "test": 2}
    for split in ("train", "test"):
        for family in reality.FAMILY_ORDER:
            for index in range(quotas[split]):
                table_id = f"{split}-{family}-{index}"
                split_tables[split].append(_table(table_id))
                split_questions[split].append(
                    _question(
                        table_id,
                        family=family,
                        phase=index + 1,
                    )
                )

    # Exact public README example: valid, but never eligible for selection.
    readme_table_id = "1-10007452-3"
    split_tables["train"].append(_table(readme_table_id))
    split_questions["train"].append(
        _question(
            readme_table_id,
            family="EQ",
            question="Who is the manufacturer for the order year 1998?",
        )
    )

    # Legitimate release rows that are ineligible under the frozen study.
    first_train = "train-EQ-0"
    split_questions["train"].extend(
        (
            _question(
                first_train,
                family="EQ",
                conditions=[[1, 0, 5], [0, 0, "name-5"]],
            ),
            _question(
                first_train,
                family="EQ",
                conditions=[[1, 3, 5]],
            ),
            _question(
                first_train,
                family="GT",
                conditions=[[1, 1, 3]],
            ),
        )
    )
    short_id = "train-short-0"
    split_tables["train"].append(_table(short_id, row_count=10))
    split_questions["train"].append(
        _question(short_id, family="EQ")
    )

    # Structurally valid release-schema rows that must be dispositioned before
    # any SQLite derivation or HMAC selection.
    wide_id = "train-wide-0"
    wide_header = [f"Column {index}" for index in range(65)]
    wide_types = ["text", "real", *(["text"] * 63)]
    wide_rows = [
        [f"name-{index}", index, *([f"value-{index}"] * 63)]
        for index in range(11)
    ]
    split_tables["train"].append(
        {
            "id": wide_id,
            "header": wide_header,
            "types": wide_types,
            "rows": wide_rows,
        }
    )
    split_questions["train"].append(_question(wide_id, family="EQ"))

    split_questions["train"].append(
        _question(
            first_train,
            family="EQ",
            question="Q" * (compiler.MAX_QUESTION_CHARACTERS + 1),
        )
    )

    long_header_id = "train-long-header-0"
    long_header_table = _table(long_header_id)
    long_header_table["header"] = [
        "H" * (compiler.MAX_HEADER_OR_CELL_CHARACTERS + 1),
        "Score",
    ]
    split_tables["train"].append(long_header_table)
    split_questions["train"].append(
        _question(long_header_id, family="EQ")
    )

    long_cell_id = "train-long-cell-0"
    long_cell_table = _table(long_cell_id)
    long_cell_rows = _rows()
    long_cell_rows[0][0] = "C" * (
        compiler.MAX_HEADER_OR_CELL_CHARACTERS + 1
    )
    long_cell_table["rows"] = long_cell_rows
    split_tables["train"].append(long_cell_table)
    split_questions["train"].append(
        _question(long_cell_id, family="EQ")
    )

    duplicate_id = "train-duplicate-0"
    duplicate_table = _table(duplicate_id)
    duplicate_rows = _rows()
    duplicate_rows[1] = list(duplicate_rows[0])
    duplicate_table["rows"] = duplicate_rows
    split_tables["train"].append(duplicate_table)
    split_questions["train"].append(
        _question(duplicate_id, family="EQ")
    )

    if item_schema_extra:
        split_questions["test"][0]["answer"] = "forbidden schema field"

    db_raw = {
        split: _sqlite_bytes(
            root / f"{name}-{split}.db",
            split_tables[split],
            mismatch_first_eq=sqlite_mismatch and split == "train",
            permute_first_table=sqlite_permuted and split == "train",
        )
        for split in ("train", "test")
    }
    members = {
        compiler.JSONL_MEMBERS[split]: _jsonl(split_questions[split])
        for split in ("train", "test")
    }
    members.update(
        {
            compiler.TABLE_MEMBERS[split]: _jsonl(split_tables[split])
            for split in ("train", "test")
        }
    )
    members.update(
        {
            compiler.DB_MEMBERS[split]: db_raw[split]
            for split in ("train", "test")
        }
    )
    members["data/version.txt"] = b"1.1\n"

    archive_path = root / f"{name}.tar.bz2"
    with tarfile.open(archive_path, "w:bz2") as bundle:
        directory = tarfile.TarInfo("data")
        directory.type = tarfile.DIRTYPE
        directory.mode = 0o700
        bundle.addfile(directory)
        for member_name in compiler.REQUIRED_MEMBERS:
            _add_member(bundle, member_name, members[member_name])
        _add_member(bundle, "data/dev.unused", b"must never be parsed")
        if unsafe_member:
            _add_member(bundle, "../../wikisql-escape", b"not extracted")
    archive_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    train_ids = tuple(
        table["id"] for table in split_tables["train"]
    )
    test_ids = tuple(
        table["id"] for table in split_tables["test"]
    )
    assert all(isinstance(value, str) for value in (*train_ids, *test_ids))
    return SyntheticArchive(
        path=archive_path,
        sha256=archive_sha256,
        table_ids=(*train_ids, *test_ids),  # type: ignore[arg-type]
        train_table_ids=train_ids,  # type: ignore[arg-type]
        test_table_ids=test_ids,  # type: ignore[arg-type]
    )


@pytest.fixture()
def compiled(
    tmp_path: Path,
) -> tuple[SyntheticArchive, compiler.CompilationBundle]:
    source = _build_archive(tmp_path)
    bundle = compiler.compile_archive(
        source.path,
        expected_archive_sha256=source.sha256,
        config=compiler.CompilerConfig.synthetic_test(),
        secret_factory=lambda width: SECRET[:width],
    )
    return source, bundle


def _recursive_keys(value: object) -> set[str]:
    result: set[str] = set()
    if isinstance(value, Mapping):
        result.update(value)
        for child in value.values():
            result.update(_recursive_keys(child))
    elif isinstance(value, list):
        for child in value:
            result.update(_recursive_keys(child))
    return result


def _without_self(value: Mapping[str, object]) -> dict[str, object]:
    return {key: row for key, row in value.items() if key != "self_sha256"}


def test_compiler_selects_exact_split_quotas_and_balanced_four_folds(
    compiled: tuple[SyntheticArchive, compiler.CompilationBundle],
) -> None:
    _source, bundle = compiled
    receipt = bundle.safe_receipt
    assert receipt["family_counts"] == {
        "A_form": {"EQ": 4, "GT": 4, "LT": 4},
        "A_hold": {"EQ": 2, "GT": 2, "LT": 2},
    }
    assert receipt["A_form_fold_family_counts"] == {
        str(fold): {"EQ": 1, "GT": 1, "LT": 1}
        for fold in range(4)
    }
    assert receipt["selected_item_count"] == 18
    assert receipt["selected_table_count"] == 18
    assert receipt["sqlite_rowid_derivation_candidate_count"] == 19
    assert receipt["sqlite_rowid_eligible_count"] == 18
    assert receipt["selected_sqlite_consistency_assert_count"] == 18
    assert receipt["sqlite_runtime_version"] == sqlite3.sqlite_version
    assert receipt["babel_runtime_version"] == compiler.babel.__version__
    assert receipt["babel_required_production_version"] == "2.10.3"
    assert receipt["babel_locale"] == "zh_CN"
    assert receipt["eligibility_contract"] == {
        "condition_count": 1,
        "condition_operator_indices": [0, 1, 2],
        "table_physical_row_count_minimum": 11,
        "table_physical_row_count_maximum": 80,
        "column_count_minimum": 1,
        "column_count_maximum": 64,
        "question_character_count_maximum": 16_000,
        "header_or_cell_character_count_maximum": 16_000,
        "canonical_serialized_row_character_count_maximum": 16_000,
        "canonical_serialized_rows_must_round_trip": True,
        "canonical_serialized_rows_must_be_unique": True,
        "sqlite_schema_rowid_order_and_normalized_cells_must_match_json_before_gold_derivation": True,
        "sqlite_gold_row_count_minimum": 1,
        "sqlite_gold_row_count_maximum": 5,
        "sqlite_gold_authoritative_before_HMAC": True,
    }
    assert receipt["train_test_table_overlap_count"] == 0
    assert receipt["authorized_member_open_count"] == 7
    assert receipt["ignored_regular_member_count"] == 1
    assert receipt["source_dispositions"]["train"] == {
        "README_example_denied": 1,
        "column_count_outside_1_64": 1,
        "condition_count_not_one": 1,
        "condition_operator_not_EQ_GT_LT": 1,
        "duplicate_canonical_serialized_rows": 1,
        "eligible": 12,
        "header_or_cell_characters_over_16000": 2,
        "question_characters_over_16000": 1,
        "sqlite_gold_row_count_outside_1_5": 1,
        "structurally_eligible_for_SQLite_derivation": 13,
        "table_row_count_outside_11_80": 1,
    }
    assert receipt["source_dispositions"]["test"] == {
        "eligible": 6,
        "structurally_eligible_for_SQLite_derivation": 6,
    }
    assert set(receipt["pack_commitments"]) == {
        "A_form_action_view",
        "A_form_label",
        "A_hold_action_view",
        "A_hold_label",
        "controller_only_provenance",
    }
    assert receipt["self_sha256"] == reality.canonical_sha256(
        _without_self(receipt)
    )


def test_action_views_are_exactly_label_free_and_source_opaque(
    compiled: tuple[SyntheticArchive, compiler.CompilationBundle],
) -> None:
    source, bundle = compiled
    for block, expected_count in (("A_form", 12), ("A_hold", 6)):
        action_pack = bundle.action_pack(block)
        label_pack = bundle.label_pack(block)
        assert action_pack["item_count"] == expected_count
        assert label_pack["item_count"] == expected_count
        assert action_pack["contains_labels"] is False
        assert action_pack["self_sha256"] == reality.canonical_sha256(
            _without_self(action_pack)
        )
        assert label_pack["self_sha256"] == reality.canonical_sha256(
            _without_self(label_pack)
        )
        action_items = action_pack["items"]
        label_items = label_pack["items"]
        assert isinstance(action_items, list)
        assert isinstance(label_items, list)
        assert all(
            set(item) == compiler.ACTION_VIEW_FIELDS
            for item in action_items
        )
        assert not (
            _recursive_keys(action_items)
            & compiler.ACTION_FORBIDDEN_FIELDS
        )
        serialized = reality.canonical_json_bytes(action_pack).decode("ascii")
        assert all(table_id not in serialized for table_id in source.table_ids)
        action_ids = {item["opaque_item_id"] for item in action_items}
        label_ids = {item["opaque_item_id"] for item in label_items}
        assert action_ids == label_ids
        expected_label_fields = set(compiler.LABEL_VIEW_FIELDS)
        if block == "A_form":
            expected_label_fields.add("fold_index")
        assert all(
            set(item) == expected_label_fields for item in label_items
        )
        assert not (
            {"split", "source_line_number", "source_table_id", "sql"}
            & _recursive_keys(label_items)
        )
        for action in action_items:
            assert len(action["opaque_item_id"]) == 64
            assert len(action["physical_rows"]) == 11
        for label in label_items:
            assert label["family"] in reality.FAMILY_ORDER
            assert 1 <= len(label["gold_row_ids"]) <= 5
            assert label["sqlite_rowid_cross_checked"] is True
            matching = next(
                action
                for action in action_items
                if action["opaque_item_id"] == label["opaque_item_id"]
            )
            assert label["action_view_sha256"] == reality.canonical_sha256(
                matching
            )


def test_controller_only_provenance_retains_lineage_outside_minimal_labels(
    compiled: tuple[SyntheticArchive, compiler.CompilationBundle],
) -> None:
    _source, bundle = compiled
    provenance = bundle.controller_provenance_pack
    assert provenance["access_policy"] == (
        "controller_only_never_Agent_or_scorer_input"
    )
    assert provenance["item_count"] == 18
    assert provenance["self_sha256"] == reality.canonical_sha256(
        _without_self(provenance)
    )
    rows = provenance["items"]
    assert isinstance(rows, list)
    assert all(set(row) == compiler.PROVENANCE_FIELDS for row in rows)
    form_rows = [row for row in rows if row["block"] == "A_form"]
    hold_rows = [row for row in rows if row["block"] == "A_hold"]
    assert len(form_rows) == 12
    assert len(hold_rows) == 6
    form_tables = {row["source_table_id"] for row in form_rows}
    hold_tables = {row["source_table_id"] for row in hold_rows}
    assert form_tables.isdisjoint(hold_tables)
    assert "1-10007452-3" not in form_tables | hold_tables
    assert all(row["split"] == "train" for row in form_rows)
    assert all(row["split"] == "test" for row in hold_rows)
    assert all(set(row["sql"]) == reality.OFFICIAL_SQL_FIELDS for row in rows)

    form_labels = bundle.a_form_label_pack["items"]
    hold_labels = bundle.a_hold_label_pack["items"]
    assert isinstance(form_labels, list)
    assert isinstance(hold_labels, list)
    assert {
        (row["family"], row["fold_index"]) for row in form_labels
    } == {
        (family, fold)
        for family in reality.FAMILY_ORDER
        for fold in range(4)
    }
    assert all("fold_index" not in row for row in hold_labels)
    assert {
        row["opaque_item_id"] for row in rows
    } == {
        row["opaque_item_id"] for row in (*form_labels, *hold_labels)
    }


def test_private_packs_and_safe_receipt_write_exclusively_with_mode_0600(
    compiled: tuple[SyntheticArchive, compiler.CompilationBundle],
) -> None:
    _source, bundle = compiled
    with tempfile.TemporaryDirectory(
        prefix="wikisql-compiler-test-",
        dir="/tmp",
    ) as raw_root:
        output = Path(raw_root) / "compiled"
        hashes = compiler.write_compilation(output, bundle)
        assert set(hashes) == {
            "private/selection_secret.bin",
            "private/A_form.action_views.json",
            "private/A_form.labels.json",
            "private/A_hold.action_views.json",
            "private/A_hold.labels.json",
            "private/controller_only.provenance.json",
            "safe/source_compiler_receipt.json",
        }
        assert stat.S_IMODE(output.stat().st_mode) == 0o700
        assert stat.S_IMODE((output / "private").stat().st_mode) == 0o700
        assert stat.S_IMODE((output / "safe").stat().st_mode) == 0o700
        for relative in hashes:
            path = output / relative
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
            assert hashlib.sha256(path.read_bytes()).hexdigest() == hashes[relative]
        action = json.loads(
            (output / "private/A_hold.action_views.json").read_text("ascii")
        )
        labels = json.loads(
            (output / "private/A_hold.labels.json").read_text("ascii")
        )
        provenance = json.loads(
            (
                output / "private/controller_only.provenance.json"
            ).read_text("ascii")
        )
        assert "family" not in _recursive_keys(action["items"])
        assert "family" in _recursive_keys(labels["items"])
        assert "sql" not in _recursive_keys(labels["items"])
        assert "sql" in _recursive_keys(provenance["items"])
        with pytest.raises(
            compiler.WikiSQLSourceCompilerError, match="cannot be created"
        ):
            compiler.write_compilation(output, bundle)


def test_source_capacity_failure_is_terminal_source_ineligible(
    tmp_path: Path,
) -> None:
    source = _build_archive(tmp_path, name="capacity")
    with pytest.raises(
        compiler.WikiSQLSourceIneligibleError, match="quota"
    ):
        compiler.compile_archive(
            source.path,
            expected_archive_sha256=source.sha256,
            config=compiler.CompilerConfig.synthetic_test(
                a_form_quota_per_family=8,
                a_hold_quota_per_family=2,
            ),
            secret_factory=lambda width: SECRET[:width],
        )


def test_strict_item_schema_and_sqlite_rowid_mismatch_fail_closed(
    tmp_path: Path,
) -> None:
    schema = _build_archive(
        tmp_path, name="schema", item_schema_extra=True
    )
    with pytest.raises(
        compiler.WikiSQLSourceCompilerError, match="item schema"
    ):
        compiler.compile_archive(
            schema.path,
            expected_archive_sha256=schema.sha256,
            config=compiler.CompilerConfig.synthetic_test(),
            secret_factory=lambda width: SECRET[:width],
        )

    mismatch = _build_archive(
        tmp_path, name="mismatch", sqlite_mismatch=True
    )
    with pytest.raises(
        compiler.WikiSQLSourceCompilerError,
        match="normalized cell disagrees with JSON table rows",
    ):
        compiler.compile_archive(
            mismatch.path,
            expected_archive_sha256=mismatch.sha256,
            config=compiler.CompilerConfig.synthetic_test(),
            secret_factory=lambda width: SECRET[:width],
        )

    permuted = _build_archive(
        tmp_path, name="permuted", sqlite_permuted=True
    )
    with pytest.raises(
        compiler.WikiSQLSourceCompilerError,
        match="normalized cell disagrees with JSON table rows",
    ):
        compiler.compile_archive(
            permuted.path,
            expected_archive_sha256=permuted.sha256,
            config=compiler.CompilerConfig.synthetic_test(),
            secret_factory=lambda width: SECRET[:width],
        )


def test_shared_row_document_round_trip_and_whole_document_bound() -> None:
    marker = 'legitimate " (text) = payload'
    rows = _rows()
    rows[0][0] = marker
    table = reality.WikiSQLTable(
        table_id="round-trip",
        header=('Header " (text) = literal', "Score"),
        types=("text", "real"),
        rows=tuple(tuple(row) for row in rows),
    )
    assert compiler._structural_eligibility_reason(  # noqa: SLF001
        question="Which row is equal to five?",
        table=table,
    ) is None
    documents = reality.validated_retrieval_documents(table)
    assert reality.parse_serialized_table_row_values(
        documents[0],
        table.header,
        table.types,
    )[0] == marker

    long_rows = tuple(
        (
            f"{row_index}-" + ("a" * 8_100),
            f"{row_index}-" + ("b" * 8_100),
        )
        for row_index in range(11)
    )
    long_table = reality.WikiSQLTable(
        table_id="whole-document-bound",
        header=("First", "Second"),
        types=("text", "text"),
        rows=long_rows,
    )
    assert all(
        len(cell) <= compiler.MAX_HEADER_OR_CELL_CHARACTERS
        for row in long_rows
        for cell in row
    )
    assert compiler._structural_eligibility_reason(  # noqa: SLF001
        question="Which row matches?",
        table=long_table,
    ) == "canonical_serialized_row_characters_over_16000"


def test_selected_items_receive_an_independent_sqlite_consistency_assert(
    tmp_path: Path,
) -> None:
    source = _build_archive(tmp_path, name="selected-consistency")

    def corrupt_one_derived_rowid(
        db_members: Mapping[str, bytes],
        requests: tuple[compiler.SQLiteDerivationRequest, ...],
    ) -> Mapping[str, tuple[int, ...]]:
        result = dict(compiler.sqlite_rowid_derive(db_members, requests))
        target = requests[0].item_commitment_sha256
        result[target] = (0,) if result[target] != (0,) else (1,)
        return result

    with pytest.raises(
        compiler.WikiSQLSourceCompilerError,
        match="selected SQLite consistency assert failed",
    ):
        compiler.compile_archive(
            source.path,
            expected_archive_sha256=source.sha256,
            config=compiler.CompilerConfig.synthetic_test(),
            secret_factory=lambda width: SECRET[:width],
            sqlite_rowid_deriver=corrupt_one_derived_rowid,
        )


def test_sqlite_derivation_precedes_secret_creation_and_hmac(
    tmp_path: Path,
) -> None:
    source = _build_archive(tmp_path, name="order")
    events: list[str] = []

    def derive(
        db_members: Mapping[str, bytes],
        requests: tuple[compiler.SQLiteDerivationRequest, ...],
    ) -> Mapping[str, tuple[int, ...]]:
        events.append("SQLite_authoritative_derivation")
        return compiler.sqlite_rowid_derive(db_members, requests)

    def secret_factory(width: int) -> bytes:
        assert events == ["SQLite_authoritative_derivation"]
        events.append("HMAC_secret_creation")
        return SECRET[:width]

    compiler.compile_archive(
        source.path,
        expected_archive_sha256=source.sha256,
        config=compiler.CompilerConfig.synthetic_test(),
        secret_factory=secret_factory,
        sqlite_rowid_deriver=derive,
    )
    assert events == [
        "SQLite_authoritative_derivation",
        "HMAC_secret_creation",
    ]


def test_tar_traversal_header_is_rejected_without_extraction(
    tmp_path: Path,
) -> None:
    source = _build_archive(tmp_path, name="unsafe", unsafe_member=True)
    outside = tmp_path / "wikisql-escape"
    assert not outside.exists()
    with pytest.raises(
        compiler.WikiSQLSourceCompilerError, match="unsafe member"
    ):
        compiler.read_authorized_archive(
            source.path,
            expected_archive_sha256=source.sha256,
            config=compiler.CompilerConfig.synthetic_test(),
        )
    assert not outside.exists()


def test_archive_hash_version_and_production_config_cannot_be_relaxed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _build_archive(tmp_path, name="identity")
    with pytest.raises(
        compiler.WikiSQLSourceCompilerError, match="SHA-256 drifted"
    ):
        compiler.read_authorized_archive(
            source.path,
            expected_archive_sha256="0" * 64,
            config=compiler.CompilerConfig.synthetic_test(),
        )
    with pytest.raises(
        compiler.WikiSQLSourceCompilerError, match="production compiler"
    ):
        compiler.CompilerConfig(
            mode="production",
            a_form_quota_per_family=4,
            a_hold_quota_per_family=2,
            expected_archive_size_bytes=None,
            expected_archive_git_blob_sha1=None,
        )
    production = compiler.CompilerConfig.production()
    if compiler.babel.__version__ != compiler.PRODUCTION_BABEL_VERSION:
        with pytest.raises(
            compiler.WikiSQLSourceCompilerError,
            match="production compile requires Babel 2.10.3",
        ):
            compiler.compile_archive(
                source.path,
                expected_archive_sha256=source.sha256,
                config=production,
            )
    monkeypatch.setattr(
        compiler.babel,
        "__version__",
        compiler.PRODUCTION_BABEL_VERSION,
    )
    assert production.a_form_quota_per_family == 64
    assert production.a_hold_quota_per_family == 24
    assert production.expected_archive_size_bytes == 26_164_664
    assert (
        production.expected_archive_git_blob_sha1
        == "941de4cb2ad5fa7aeb2e37d314468636ce070af7"
    )


def test_archive_hashing_binds_the_official_git_blob_domain(
    tmp_path: Path,
) -> None:
    payload = b"WikiSQL custody probe"
    path = tmp_path / "probe.bin"
    path.write_bytes(payload)
    sha256, git_blob_sha1, size = compiler._hash_file(path)  # noqa: SLF001
    assert sha256 == hashlib.sha256(payload).hexdigest()
    assert size == len(payload)
    expected = hashlib.sha1(  # noqa: S324 - official Git identity
        f"blob {len(payload)}\0".encode("ascii") + payload
    ).hexdigest()
    assert git_blob_sha1 == expected


def test_sqlite_numeric_fallback_uses_official_first_match_semantics() -> None:
    assert compiler._sqlite_condition_value(  # noqa: SLF001
        "-12.5 kg", "real"
    ) == pytest.approx(-12.5)
    assert compiler._sqlite_condition_value(  # noqa: SLF001
        "abc 12 x 34", "real"
    ) == pytest.approx(12.0)
    assert compiler._sqlite_condition_value(  # noqa: SLF001
        "1,234.5", "real"
    ) == pytest.approx(1234.5)
    assert compiler._sqlite_condition_value(  # noqa: SLF001
        "SOUTH AUSTRALIA", "text"
    ) == "south australia"


def test_sqlite_numeric_primary_parse_explicitly_uses_zh_cn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_parse_decimal(value: str, *, locale: str) -> str:
        calls.append((value, locale))
        return "1234.5"

    monkeypatch.setattr(compiler, "parse_decimal", fake_parse_decimal)
    assert compiler._sqlite_condition_value(  # noqa: SLF001
        "1,234.5", "real"
    ) == pytest.approx(1234.5)
    assert calls == [("1,234.5", "zh_CN")]
