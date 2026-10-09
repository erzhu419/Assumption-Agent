"""Program revision retains its cost owner through real relation-cell refinement."""

from random import Random

from hegel_machine.cognition.measurement_lab import MeasurementBatch
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, Probe, ResearchConfig, TaskSpec
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from test_cognition_nested_relation_revisions import composition


class NestedResponseLab:
    channels = ()

    def __init__(self, radius):
        self.radius = radius
        bounds = {'f0': (1., 1.), 'f1': (0., 0.), 'q0': (-1., 1.),
                  'q1': (-1., 1.), 'x0': (-1., 1.), 'x1': (-1., 1.)}
        self.spec = TaskSpec('controlled_nested_response', 'Predict the controlled response',
            tuple(bounds), (), (), feature_bounds=bounds, max_probe_length=64)
        self.acquisitions = []

    def spec_for(self, channels=()):
        assert not channels
        return self.spec

    def sample(self, seed, length):
        rng = Random(seed)
        return self.acquire(Probe('source_observations', tuple(
            {name: rng.uniform(*limits) for name, limits in self.spec.feature_bounds.items()}
            for _ in range(length))))

    def acquire(self, probe, channels=()):
        assert not channels
        rows, history, flows, combined = [], [], [], []
        for point in probe.sequence:
            flow = ((point['f0'] - point['f1']) * (point['q1'] - point['q0'])
                    if abs(point['x0'] - point['x1']) < self.radius else 0.)
            event = flow - flows[-1] if flows else 0.
            flows.append(flow)
            combined.append(1.3 * (point['q0'] + point['q1']) - .7 * event)
            target = sum(combined[-3:])
            rows.append(Observation(dict(point), target, tuple(history)))
            history.append({**point, '__target__': target})
        self.acquisitions.append({'probe': probe, 'observations': rows})
        return MeasurementBatch(rows, len(rows), 0, ())


def test_program_refinement_pays_real_followup_without_losing_its_original_trial(record_property):
    source = NestedResponseLab(.6)
    measured = source.sample(31, 64).observations
    learned = composition(.6).expanded()
    assert max(abs(learned.predict(row) - row.target) for row in measured) < 1e-12
    library = KnowledgeLibrary()
    library.learn(Candidate('paid_source_response', learned, 'observed_response'), source.spec, measured)
    config = ResearchConfig(observation_budget=240, compute_budget=240000,
        minimum_improvement=1e-7, max_rounds=4, active_probes=True, max_candidates=8)
    laboratory = NestedResponseLab(1.2)
    report = UnifiedResearchAgent(config, library=library).run(laboratory)
    refinements = [event for event in report['actions'] if event['action'] == 'refine_relation_parameter_space']
    acquisitions = [event for event in report['actions'] if event['action'] == 'acquire'
                    and event['purpose'] == 'distinguish_program_conditions']
    followups = [event for event in report['actions'] if event['action'] == 'research_action_followup'
                 and event['stage'] == 'program_condition_research']
    assert refinements and acquisitions and followups, [(event['action'], event.get('purpose'))
                                                       for event in report['actions']]
    assert len(refinements) == len(acquisitions) + 1
    charged = sum(event['observation_cost'] for event in followups)
    assert charged == sum(event['observation_cost'] for event in acquisitions)
    assert all(event['measurement_cost'] == 0 for event in followups)
    assert sum(event['compute_cost'] for event in followups) >= sum(event['compute_cost'] for event in refinements)
    initial = [event for event in report['actions'] if event['action'] == 'research_action_feedback']
    for event in followups:
        origin = next(item for item in initial if item['research_action'] == event['research_action']
                      and item['context'] == event['context'] and item['feedback']['trials'] == event['trial'])
        assert report['actions'].index(origin) < report['actions'].index(event)
        previous = [item for item in report['actions'][:report['actions'].index(event)]
            if item['action'] in {'research_action_feedback', 'research_action_confirmation'}
            and item['research_action'] == event['research_action'] and item['context'] == event['context']]
        latest = previous[-1]['feedback']
        assert event['feedback']['trials'] == latest['trials']
        assert event['feedback']['expected_gain'] == latest['expected_gain']
    assert report['costs']['observation_cost'] == sum(len(batch['observations']) for batch in laboratory.acquisitions)
    assert report['costs']['observation_cost'] <= config.observation_budget
    assert report['costs']['compute_cost'] <= config.compute_budget
    record_property('research_status', report['status'])
    record_property('program_probes', len(acquisitions))
    record_property('program_followup_observations', charged)
    record_property('program_followup_compute', sum(event['compute_cost'] for event in followups))
