"""Shared transfer fitting must keep distinct laws executable after new data."""

from dataclasses import replace
import json

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.measurement_lab import MeasurementBatch
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, Probe, SearchBudget
from hegel_machine.cognition.shared_transfer_fitting import SharedTransferFitter

from test_cognition_compound_unified_research import (
    CONFIG, FinitePriorResearcher, WindowLaboratory, hypotheses,
)
from test_cognition_transfer_condition_runtime import (
    EventFinitePriorResearcher, EventWindowLaboratory, _independent_check,
)


ACTIVE = (.271234, .683214)


def restored(library):
    return KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))


def seed_law(library, laboratory, program, label):
    probe = Probe(label, tuple(dict(zip(laboratory.names, point)) for point in
        ((0., 0.), ACTIVE, ACTIVE, (.9, .9), (.9, .9))))
    rows = laboratory.acquire(probe).observations
    assert all(program.predict(row) == row.target for row in rows)
    return library.learn(Candidate(label, program, "observed_source"),
                         replace(laboratory.spec, task_id=label), rows)[0]


def shared_fits(report):
    events = [item for item in report["actions"] if item["action"] == "shared_transfer_fit"]
    assert events and all(item["candidate_count"] >= 2 for item in events)
    assert all(item["sharing"] for item in events)
    return events


def test_two_saved_gain_laws_survive_shared_fits_rename_new_gain_and_return():
    library = KnowledgeLibrary()
    for gain, program in zip((1., 2.), hypotheses(("x", "y"))):
        seed_law(library, WindowLaboratory(gain=gain), program, f"source_gain_{gain}")
    sources = {item.concept_id: item.program.to_dict() for item in library.concepts}
    library = restored(library)
    reports = []
    # Gain3 is outside the supplied ordinary finite prior. The old calls must
    # survive their all-zero pilot and support a real coefficient revision.
    for seed, lab in ((173, WindowLaboratory(("renamed_a", "renamed_b"), gain=3.)),
                      (179, WindowLaboratory(gain=2.))):
        report = FinitePriorResearcher(replace(CONFIG, seed=seed), library=library).run(lab)
        assert report["status"] == "completed"
        _independent_check(lab, report)
        events = shared_fits(report)
        assert events[0]["candidate_count"] >= 4
        assert sum(len(rows) for _, rows in lab.acquisitions) == report["costs"]["observation_cost"]
        reports.append(report)
        library = restored(library)
    kept = {item.concept_id: item for item in library.concepts}
    assert all(kept[name].program.to_dict() == payload and kept[name].status != "revoked"
               for name, payload in sources.items())
    assert any(item["action"] == "refit_transfer_after_experiment" for item in reports[0]["actions"])


class ShiftedWindowLaboratory(WindowLaboratory):
    def __init__(self, center, **kwargs):
        super().__init__(**kwargs)
        self.center = center

    def acquire(self, probe, channels=()):
        rows, history = [], []
        for point in probe.sequence:
            active = all(abs(point[name] - center) < 1e-5 for name, center in zip(self.names, self.center))
            target = self.gain if active else 0.
            rows.append(Observation(dict(point), target, tuple(history)))
            history.append({**point, "__target__": target})
        self.acquisitions.append((probe, tuple(rows)))
        return MeasurementBatch(rows, len(rows), 0, ())


def test_equal_inactive_pilot_columns_do_not_merge_different_activation_laws():
    alternate = (.371234, .583214)

    def shifted(node):
        if node.op == "const" and node.value is not None:
            for old, new in zip(ACTIVE, alternate):
                if abs(node.value - old) < 2e-5:
                    return replace(node, value=node.value + new - old)
        return replace(node, children=tuple(shifted(child) for child in node.children))

    first = hypotheses(("x", "y"))[1]
    second = shifted(first)
    library = KnowledgeLibrary()
    seed_law(library, WindowLaboratory(gain=2.), first, "first_region")
    lab = ShiftedWindowLaboratory(alternate, gain=2.)
    source_rows = lab.acquire(Probe("alternate_source", tuple(dict(zip(lab.names, point))
        for point in ((0., 0.), alternate, (.9, .9))))).observations
    alternate_id, _ = library.learn(Candidate("second_region", second, "observed_source"),
                                    replace(lab.spec, task_id="second_region"), source_rows)
    lab = ShiftedWindowLaboratory(alternate, gain=2.)
    report = FinitePriorResearcher(replace(CONFIG, seed=181), library=restored(library)).run(lab)
    assert report["status"] == "completed"
    assert all(row.target == 0. for row in lab.acquisitions[0][1])
    shared_fits(report)
    experiments = [item for item in report["actions"]
                   if item["action"] == "test_transfer_condition_predictions"]
    assert experiments and len(experiments[0]["plan"]["hypotheses"]) >= 3
    frozen = [Program.from_dict(item["program"]) for item in experiments[0]["plan"]["hypotheses"]]
    probe_rows = [Observation(dict(zip(lab.names, point)), 0.) for point in (ACTIVE, alternate)]
    signatures = {tuple(program.predict(row) for row in probe_rows) for program in frozen}
    assert (2., 0.) in signatures and (0., 2.) in signatures
    terminal = ShiftedWindowLaboratory(alternate, gain=2.).acquire(Probe("terminal", tuple(
        dict(zip(lab.names, point)) for point in ((0., 0.), ACTIVE, alternate, (.9, .9))))).observations
    predictor = Program.from_dict(report["program"])
    assert max((predictor.predict(row) - row.target) ** 2 for row in terminal) < 1e-7
    assert alternate_id in report["used_sources"]


class EntryPilotResearcher(EventFinitePriorResearcher):
    def _probe(self, spec, serial, length, candidates, budget, actions):
        if serial != 0:
            return super()._probe(spec, serial, length, candidates, budget, actions)
        # Reproduce an entry-only paid pilot: it distinguishes neither a
        # scaled event nor its square. Exit remains a later real experiment.
        points = [(0., 0.), *([ACTIVE] * (length - 1))]
        return Probe("entry_only_pilot", tuple(dict(zip(spec.feature_names, point)) for point in points))


def test_shared_entry_fit_is_revised_when_later_event_exit_changes_column_relations():
    library = KnowledgeLibrary()
    base = Program("event_delta", lag=1, children=(hypotheses(("x", "y"))[1],))
    seed_law(library, EventWindowLaboratory(gain=2.), base, "event_source")
    target = EventWindowLaboratory(("later_a", "later_b"), gain=1.)
    report = EntryPilotResearcher(replace(CONFIG, seed=191), library=restored(library)).run(target)
    assert report["status"] == "completed"
    assert all(row.target >= 0. for row in target.acquisitions[0][1])
    assert any(row.target < 0. for _, rows in target.acquisitions[1:] for row in rows)
    assert len(shared_fits(report)) >= 2
    _independent_check(target, report, temporal=True)


def test_separate_instrument_views_and_new_targets_get_their_own_fit_and_validation():
    fitter = SharedTransferFitter()
    reading = Program("feature", name="reading")
    controls = (-1., -.4, .2, .7, 1.)
    view_a = [Observation({"reading": value}, 2. * value + 1.) for value in controls]
    view_b = [Observation({"reading": -value}, 2. * value + 1.) for value in controls]
    budget = SearchBudget(10000)
    first = fitter.fit([reading], view_a, budget, "shared_view", view=("a", "fit"))
    second = fitter.fit([reading], view_b, budget, "shared_view", view=("b", "fit"))
    assert first is not None and second is not None
    programs = [item["candidate"].program for item in (first, second)]
    assert abs(programs[0].predict(Observation({"reading": .3}, 0.)) - 1.6) < 1e-7
    assert abs(programs[1].predict(Observation({"reading": .3}, 0.)) - .4) < 1e-7
    original = fitter.loss(programs[0], view_a, budget, 0., .5, view=("a", "validation"))
    changed = [Observation(row.features, row.target + 3., row.history) for row in view_a]
    later = fitter.loss(programs[0], changed, budget, 0., .5, view=("a", "validation"))
    assert original < 1e-7 and later > 8.9
    revised = fitter.fit([reading], changed, budget, "shared_view", view=("a", "fit"))
    assert abs(revised["candidate"].program.predict(Observation({"reading": .3}, 0.)) - 4.6) < 1e-7
