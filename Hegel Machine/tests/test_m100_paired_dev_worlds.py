"""Exercise the crossed M10 development conditions using public data only."""
from dataclasses import replace

import pytest

from hegel_machine.cognition import m10_0_reliable_modular_amortization_and_bidirectional_withdrawal_world as world


@pytest.mark.parametrize("source", world.DEV_M100_WORLD_SPECS, ids=lambda s: s.world_id)
def test_source_cluster_is_paired_across_order_and_grid(source):
    default = world.build_m100_world_tasks(source)
    seen_inputs = {}
    seen_event_evidence = {}
    cells_by_condition = {}
    cluster_ids = set()
    condition_ids = set()

    for grid in ("2x3", "3x3"):
        for order in ("sep_first", "coupled_first"):
            stage, tasks, manifest, logs = world.build_m100_world_tasks(source, grid_size=grid, order_mode=order)
            assert stage == default[0]
            assert len(manifest) == len(tasks) == (7 if grid == "2x3" else 10)
            if grid == "3x3" and order == source.order_mode:
                assert tasks == default[1] and logs == default[3]
                for row, initial in zip(manifest, default[2], strict=True):
                    assert all(row[key] == value for key, value in initial.items())

            cluster_ids.update(row["source_cluster_id"] for row in manifest)
            condition_ids.update(row["condition_id"] for row in manifest)
            ids = [row["paired_event_id"] for row in manifest]
            first, second = ("separable_d2_discovery", "coupled_d2_discovery")
            if order == "coupled_first":
                first, second = second, first
            assert ids.index(first) < ids.index(second)
            assert ids.index("u1_discovery") < ids.index("coupled_d2_discovery")
            assert ids.index("d1_discovery") < ids.index("u1_d1_combination")
            assert max(ids.index(first), ids.index(second)) < min(ids.index("separable_d2_revisit"), ids.index("coupled_d2_revisit"))
            if grid == "3x3":
                assert ids.index("u2_discovery") < min(ids.index("u2_d1_combination"), ids.index("u2_d2_combination"))
            assert (logs[ids.index(first)+1]["facility_telemetry_trace"]
                    == logs[ids.index(second)+1]["facility_telemetry_trace"])
            assert (logs[ids.index("separable_d2_revisit")+1]["facility_telemetry_trace"]
                    == logs[ids.index("coupled_d2_revisit")+1]["facility_telemetry_trace"])
            cells = {("U0", "D0")}
            for i, (task, row) in enumerate(zip(tasks, manifest, strict=True)):
                key = row["paired_event_id"]
                if task.public_setup_py not in seen_inputs:
                    seen_inputs[task.public_setup_py] = world.materialize_m98_public_item_and_sha(task)
                data, public_sha = seen_inputs[task.public_setup_py]
                signature = (public_sha, logs[i+1]["facility_telemetry_trace"])
                assert signature == seen_event_evidence.setdefault(key, signature)
                if key in (first, second):
                    assert data["campaign_diagnostics"]
                if key.endswith("_revisit"):
                    assert not data["campaign_diagnostics"]
                cells.add((row["true_upstream_mode"], row["true_downstream_precursor"]))
            assert len(cells) == (6 if grid == "2x3" else 9)
            cells_by_condition[grid, order] = cells
    assert len(seen_inputs) == len(seen_event_evidence) == 10
    assert len(condition_ids) == 4 and cluster_ids == {source.cluster_id}
    assert cells_by_condition["2x3", "coupled_first"] == cells_by_condition["2x3", "sep_first"]
    assert cells_by_condition["3x3", "coupled_first"] == cells_by_condition["3x3", "sep_first"]
    assert cells_by_condition["2x3", "sep_first"] < cells_by_condition["3x3", "sep_first"]


def test_paired_controls_reject_unknown_conditions_and_other_splits_before_construction():
    assert len({spec.cluster_id for spec in world.DEV_M100_WORLD_SPECS}) == 4
    source = world.DEV_M100_WORLD_SPECS[0]
    with pytest.raises(ValueError, match="Unsupported"):
        world.build_m100_world_tasks(source, grid_size="2x2")
    with pytest.raises(ValueError, match="Unsupported"):
        world.build_m100_world_tasks(source, order_mode="unregistered")
    with pytest.raises(ValueError, match="development"):
        world.build_m100_world_tasks(replace(source, split="reserved"), order_mode="sep_first")
