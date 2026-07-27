from __future__ import annotations

import io
import json
import struct
import unittest
from unittest.mock import patch
import zipfile

from assumption_agent.benchmarks import (
    bird_p0_public_source_qualification_v1 as bird,
)
from assumption_agent.benchmarks.spider_p0_public_source_qualification_v1 import (
    RowContractError,
)


def _schema(database_id: str = "synthetic_db") -> dict[str, object]:
    original_columns = [
        [-1, "*"],
        [0, "id"],
        [0, "name"],
        [1, "id"],
        [1, "a_id"],
        [1, "value"],
        [2, "id"],
        [2, "b_id"],
        [2, "flag"],
    ]
    return {
        "column_names": original_columns,
        "column_names_original": original_columns,
        "column_types": [
            "text",
            "number",
            "text",
            "number",
            "number",
            "number",
            "number",
            "number",
            "text",
        ],
        "db_id": database_id,
        "foreign_keys": [[4, 1], [7, 3]],
        "primary_keys": [1, 3, 6],
        "table_names": ["a", "b", "c"],
        "table_names_original": ["A", "B", "C"],
    }


class BirdQualificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schemas = bird._parse_schemas([_schema()])

    def test_one_foreign_key_edge(self) -> None:
        row = {
            "SQL": (
                "SELECT a.name, b.value FROM A AS a "
                "JOIN B AS b ON a.id = b.a_id"
            ),
            "db_id": "synthetic_db",
            "question": "synthetic",
        }
        family, database_id, evidence_count, table_count = bird._classify_row(
            row, self.schemas
        )
        self.assertEqual(family, "ONE_FOREIGN_KEY_EDGE")
        self.assertEqual(database_id, "synthetic_db")
        self.assertEqual(evidence_count, 4)
        self.assertEqual(table_count, 2)

    def test_multi_foreign_key_path(self) -> None:
        row = {
            "SQL": (
                "SELECT a.name FROM A a "
                "JOIN B b ON a.id = b.a_id "
                "JOIN C c ON b.id = c.b_id"
            ),
            "db_id": "synthetic_db",
            "question": "synthetic",
        }
        family, _, evidence_count, table_count = bird._classify_row(
            row, self.schemas
        )
        self.assertEqual(family, "MULTI_FOREIGN_KEY_PATH")
        self.assertEqual(evidence_count, 5)
        self.assertEqual(table_count, 3)

    def test_nested_relation_precedes_one_edge(self) -> None:
        row = {
            "SQL": (
                "SELECT a.name FROM A a WHERE a.id IN "
                "(SELECT b.a_id FROM B b WHERE b.value > 1)"
            ),
            "db_id": "synthetic_db",
            "question": "synthetic",
        }
        family, _, evidence_count, table_count = bird._classify_row(
            row, self.schemas
        )
        self.assertEqual(family, "NESTED_OR_SET_RELATION")
        self.assertEqual(evidence_count, 4)
        self.assertEqual(table_count, 2)

    def test_unknown_table_fails_closed(self) -> None:
        row = {
            "SQL": "SELECT x.value, x.id FROM Unknown x",
            "db_id": "synthetic_db",
            "question": "synthetic",
        }
        with self.assertRaises(RowContractError) as raised:
            bird._classify_row(row, self.schemas)
        self.assertEqual(
            raised.exception.code, "SQL_table_unknown_or_ambiguous"
        )

    def test_bound_member_uses_two_exact_ranges_and_verifies_payload(
        self,
    ) -> None:
        member = "train/train.json"
        payload = json.dumps(
            [{"SQL": "SELECT 1", "db_id": "x", "question": "q"}]
        ).encode("utf-8")
        package = io.BytesIO()
        with zipfile.ZipFile(
            package, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            archive.writestr(member, payload)
        raw_archive = package.getvalue()
        with zipfile.ZipFile(io.BytesIO(raw_archive)) as archive:
            info = archive.getinfo(member)
        local_header = raw_archive[
            info.header_offset : info.header_offset + 30
        ]
        fields = struct.unpack("<4s5H3L2H", local_header)
        name_length, extra_length = fields[-2:]
        binding = {
            "archive": "train",
            "compressed_bytes": info.compress_size,
            "compression_method": info.compress_type,
            "crc32": f"{info.CRC:08x}",
            "flags": info.flag_bits,
            "local_header_offset": info.header_offset,
            "uncompressed_bytes": info.file_size,
        }
        archive_binding = {
            "archive_byte_count": len(raw_archive),
            "etag": "synthetic",
            "last_modified": "synthetic",
        }
        calls: list[tuple[int, int]] = []

        def fake_range_get(**kwargs: object) -> bytes:
            start = int(kwargs["start"])
            end = int(kwargs["end"])
            calls.append((start, end))
            return raw_archive[start : end + 1]

        with patch.object(bird, "_range_get", side_effect=fake_range_get):
            parsed, receipt = bird._open_bound_member(
                member, binding, archive_binding
            )
        self.assertEqual(parsed[0]["db_id"], "x")
        self.assertEqual(receipt["range_GET_attempt_count"], 2)
        self.assertEqual(
            calls,
            [
                (info.header_offset, info.header_offset + 29),
                (
                    info.header_offset + 30,
                    info.header_offset
                    + 30
                    + name_length
                    + extra_length
                    + info.compress_size
                    - 1,
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
