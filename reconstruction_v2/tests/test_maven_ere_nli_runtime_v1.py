from __future__ import annotations

from pathlib import Path

from replication_runtime.qasc_nli_v1.contract import decode_request

from assumption_agent.benchmarks import maven_ere_nli_runtime_v1 as runtime


def test_design_binds_exact_two_local_workers() -> None:
    project = Path(__file__).parents[1]
    receipt = runtime.verify_maven_design(project)
    assert receipt["worker_count"] == 2
    assert receipt["torch_threads_per_worker"] == 4
    assert receipt["design_sha256"] == runtime.DESIGN_SELF_SHA256


def test_pool_addresses_both_workers_twice_and_keeps_item_keys_local(
    monkeypatch, tmp_path: Path
) -> None:
    project = Path(__file__).parents[1]
    model = tmp_path / "model"
    model.mkdir()
    canary_count = len(runtime.canonical_canary_pairs())
    canary_vector = tuple(range(canary_count))
    monkeypatch.setattr(
        runtime,
        "CANARY_SCORE_VECTOR_SHA256",
        runtime.stable_hash(list(canary_vector)),
    )
    monkeypatch.setattr(
        runtime.nli_binding,
        "verify_runtime_asset",
        lambda _project, _model: {"status": "synthetic_verified"},
    )

    class FakeWorker:
        instances: list["FakeWorker"] = []

        def __init__(self, **_kwargs: object) -> None:
            self.canary_calls = 0
            self.closed = False
            self.instances.append(self)

        def score(self, request: bytes, *, expected_count: int) -> tuple[int, ...]:
            pairs = decode_request(request)
            assert len(pairs) == expected_count
            if expected_count == canary_count:
                self.canary_calls += 1
                return canary_vector
            return tuple(len(pair.premise) - len(pair.hypothesis) for pair in pairs)

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(runtime, "_PersistentWorker", FakeWorker)
    with runtime.MavenEreNLIWorkerPool(
        model,
        project_root=project,
    ) as pool:
        result = pool.score_items(
            (
                ("private-key-1", ({"premise": "abc", "hypothesis": "z"},)),
                ("private-key-2", ({"premise": "abcdef", "hypothesis": "xy"},)),
            )
        )
        assert result == {"private-key-1": (2,), "private-key-2": (4,)}
        assert pool.receipt.canary["repeat_count_per_worker"] == 2
    assert len(FakeWorker.instances) == 2
    assert all(worker.canary_calls == 2 for worker in FakeWorker.instances)
    assert all(worker.closed for worker in FakeWorker.instances)
