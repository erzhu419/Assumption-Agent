"""Anonymous dependencies are distinguished through purchased interventions."""

import pytest

from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.models import Probe, SearchBudget
from hegel_machine.cognition.object_induction import infer_object_model
from hegel_machine.cognition.object_interventions import propose_object_probe


def acquire_records(lab, probe, records):
    batch = lab.acquire(probe, ("fine",))
    history = []
    for position, (controls, row, transition) in enumerate(zip(
            probe.sequence, batch.observations, batch.transitions)):
        records.append({"controls": dict(controls), "before": transition["before"],
            "after": transition["after"], "target": row.target, "probe_id": probe.probe_id,
            "position": position, "history": list(history)})
        history.append(dict(controls))
    return batch


def discover(lab):
    model, records, actions = None, [], []
    budget = SearchBudget(300000)
    assert lab.spec.context == {}
    for serial in range(160):
        cost = dict(lab.costs)
        probe, meta = propose_object_probe(lab.spec, model, records, serial)
        assert lab.costs == cost
        if probe is None:
            return model, records, actions
        assert len(records) + probe.cost <= 160
        assert meta["observation_rows"] == probe.cost
        assert all(set(point) == set(lab.spec.feature_bounds) and all(
            low <= point[name] <= high for name, (low, high) in lab.spec.feature_bounds.items())
                   for point in probe.sequence)
        acquire_records(lab, probe, records)
        actions.append(meta)
        model = infer_object_model(lab.spec, records, budget, prior=model)
    pytest.fail("unresolved dependencies kept acquiring the same questions")


@pytest.mark.parametrize("objects,wiring", [(3, 313), (4, 313), (4, 991)])
def test_active_anonymous_objects_are_identified_and_predict_new_impulse_effects(objects, wiring):
    with AnonymousObjectLab(objects=objects, readout="focus", apparatus_seed=9013,
                            name_seed=5177, wiring_seed=wiring) as lab:
        model, records, actions = discover(lab)
        assert model["complete"]
        assert len(records) < 90
        assert len(model["groups"]) == objects
        assert any(action["purpose"] == "distinguish_role_definition" for action in actions)
        assert model["dependencies"]  # The role experiments also identify position ownership.
        assert all(action["observation_rows"] == 1 for action in actions
                   if action["purpose"] == "distinguish_position_dependency")
        assert lab.observation_cost == len(records)
        assert lab.measurement_cost == 3 * objects * len(records)

        # Test the inferred ownership on fresh mixed configurations. No private
        # object schema participates in either discovery or this comparison.
        fresh = lab.random_probe(8191, length=objects).sequence
        for index, focus_group in enumerate(model["groups"]):
            active = next(row for row in records if row["before"][focus_group["f"]] == 1.)
            base = {**fresh[index], model["focus_control"]: active["controls"][model["focus_control"]]}
            for drive_group in model["groups"]:
                control = drive_group["controls"]["u"]
                low, high = lab.spec.feature_bounds[control]
                a = lab.acquire(Probe("new_low", ({**base, control: low},)), ("fine",))
                b = lab.acquire(Probe("new_high", ({**base, control: high},)), ("fine",))
                delta = b.observations[0].features[drive_group["u"]] - a.observations[0].features[drive_group["u"]]
                assert b.transitions[0]["after"][drive_group["q"]] - a.transitions[0]["after"][drive_group["q"]] == pytest.approx(delta)
                expected = delta * a.observations[0].features[drive_group["f"]]
                assert b.observations[0].target - a.observations[0].target == pytest.approx(expected)


def test_total_response_keeps_unidentifiable_memberships_open():
    with AnonymousObjectLab(objects=4, readout="total", apparatus_seed=9013,
                            name_seed=5177, wiring_seed=313) as lab:
        model, records, actions = discover(lab)
        assert not model["complete"] and model["schema"] is None
        assert all(len(candidate["quantities"]) == 1 for candidate in model["position_candidates"].values())
        assert all(set(candidate["quantities"]) == set(model["dynamic_fields"])
                   for candidate in model["focus_candidates"].values())
        assert any(action.get("matched_prefix_rows") == 1 for action in actions)
        assert len(records) < 140


def test_two_object_exchange_keeps_the_symmetric_position_assignment_open():
    with AnonymousObjectLab(objects=2, readout="focus", apparatus_seed=9013,
                            name_seed=5177, wiring_seed=313) as lab:
        model, _, actions = discover(lab)
        assert not model["complete"]
        assert all(len(candidate["quantities"]) == 2 for candidate in model["position_candidates"].values())
        assert not any(action["purpose"] == "distinguish_position_dependency" for action in actions)


def test_response_probe_repeats_a_known_setup_at_a_matching_reset():
    with AnonymousObjectLab(objects=3, readout="event", apparatus_seed=9013,
                            name_seed=5177, wiring_seed=313) as lab:
        records, model, budget = [], None, SearchBudget(100000)
        for serial in range(20):
            probe, meta = propose_object_probe(lab.spec, model, records, serial)
            assert probe is not None
            if meta["purpose"] == "distinguish_membership_dependency":
                assert probe.cost == 1
                assert any(row["controls"] == probe.sequence[0] and row["position"] > 0 for row in records)
                assert not any(row["controls"] == probe.sequence[0] and row["position"] == 0 for row in records)
                before = len(records)
                acquire_records(lab, probe, records)
                assert records[before]["position"] == 0 and records[before]["history"] == []
                break
            acquire_records(lab, probe, records)
            model = infer_object_model(lab.spec, records, budget, prior=model)
        else:
            pytest.fail("membership research never reached a response comparison")


def test_paid_control_scan_progresses_with_twenty_rows_and_resumes_missing_contrasts():
    settings = dict(objects=4, readout="window_total_event", apparatus_seed=96012004,
        name_seed=96089862, wiring_seed=96030714, variant="exchange",
        profile="confirmation", actuator_curvature=.31, relation_radius=.9687842617019095)
    with AnonymousObjectLab(**settings) as lab:
        records, budget = [], SearchBudget(30000)
        acquire_records(lab, lab.random_probe(96023011, length=12), records)
        model = infer_object_model(lab.spec, records, budget)
        full, _ = propose_object_probe(lab.spec, model, records, 0)
        assert full.cost == 27
        first, meta = propose_object_probe(lab.spec, model, records, 0, remaining_rows=20)
        assert first.cost == meta["observation_rows"] == 20
        acquire_records(lab, first, records)
        partial = infer_object_model(lab.spec, records, budget)
        assert partial["quantity_controls"]
        assert any(group.get("controls") for group in partial["groups"])
        # Old code declined the entire 27-row scan, leaving only four anonymous
        # quantities. The remaining seven setups now keep their measured value.
        following, _ = propose_object_probe(lab.spec, partial, records, 1, remaining_rows=20)
        assert following.cost == 7
        assert list(first.sequence) + list(following.sequence) == list(full.sequence)
        acquire_records(lab, following, records)
        split = infer_object_model(lab.spec, records, budget)
        with AnonymousObjectLab(**settings) as unsplit_lab:
            unsplit_records = []
            acquire_records(unsplit_lab, unsplit_lab.random_probe(96023011, length=12), unsplit_records)
            acquire_records(unsplit_lab, full, unsplit_records)
            unsplit = infer_object_model(unsplit_lab.spec, unsplit_records, SearchBudget(30000))
        assert split["quantity_controls"] == unsplit["quantity_controls"]
        assert {key: value["before_fields"] for key, value in split["control_effects"].items()} == {
            key: value["before_fields"] for key, value in unsplit["control_effects"].items()}


def test_membership_history_is_deferred_when_only_its_prefix_is_affordable():
    with AnonymousObjectLab(objects=3, readout="event", apparatus_seed=9013,
                            name_seed=5177, wiring_seed=313) as lab:
        records, model, budget = [], None, SearchBudget(100000)
        for serial in range(100):
            probe, meta = propose_object_probe(lab.spec, model, records, serial)
            assert probe is not None
            if meta.get("matched_prefix_rows") == 1:
                assert probe.cost == 2
                deferred, decision = propose_object_probe(lab.spec, model, records, serial,
                    remaining_rows=1)
                assert deferred is None and decision["required_rows"] == 2
                assert decision["purpose"] == "distinguish_membership_dependency"
                funded, _ = propose_object_probe(lab.spec, model, records, serial, remaining_rows=2)
                assert funded == probe
                break
            acquire_records(lab, probe, records)
            model = infer_object_model(lab.spec, records, budget, prior=model)
        else:
            pytest.fail("event research never reached a matched-history comparison")
