from gscl_closure_fixture_v1 import stable_fixture_hash


def test_fixture_hash_is_stable() -> None:
    assert stable_fixture_hash({"b": 2, "a": 1}) == stable_fixture_hash(
        {"a": 1, "b": 2}
    )
