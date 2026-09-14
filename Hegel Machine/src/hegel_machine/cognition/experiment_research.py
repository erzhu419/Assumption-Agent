"""Resolve competing learned theories by executing and transferring experiments."""

from dataclasses import asdict, replace
from random import Random

from .controller import CognitiveAgent
from .evaluation import initial_prediction, loss, point_loss
from .memory import KnowledgeLibrary
from .models import Candidate, Observation, Probe, ResearchConfig, SearchBudget, TaskSpec
from .probing import choose_probe, construct_probes, observed_classes
from .workspace import ResearchWorkspace


class ExperimentalResearch:
    """An executable question about alternatives that current data cannot separate.

    Candidates may come from an external research graph or earlier ordinary
    learning episodes. The supplied environment implements spec and observe.
    Existing observations are already paid; newly requested rows are charged
    here, and the ordinary controller receives the resulting paid workspace.
    """

    def __init__(self, library=None, config=None):
        self.library = library if library is not None else KnowledgeLibrary()
        self.config = config or ResearchConfig(observation_budget=32, compute_budget=10000,
                                               max_rounds=2, seed=557)

    def run(self, environment, candidates, *, feature_mapping=None, observations=None, queried=()):
        config, spec = self.config, environment.spec
        candidates = [replace(candidate, program=candidate.program.remap_features(feature_mapping))
                      if feature_mapping else candidate for candidate in candidates]
        if not candidates:
            raise ValueError("experimental research needs executable competing explanations")
        rows, probes = list(observations or ()), list(queried)
        observation_cost = 0
        if not rows:
            for probe in spec.initial_probes:
                if observation_cost + probe.cost > config.observation_budget:
                    raise ValueError("initial experiments exceed the observation budget")
                acquired = environment.observe(probe)
                rows.extend(acquired)
                probes.append(probe)
                observation_cost += len(acquired)
        if len(rows) < 3:
            raise ValueError("supply at least three observed rows to continue a research question")
        initial_rows = list(rows)
        fallback = initial_prediction(rows, spec.risk_quantile)
        research_work = SearchBudget(config.compute_budget // 3)
        before_classes = observed_classes(candidates, rows, fallback, research_work)
        initial_scores = []
        for candidate in candidates:
            if research_work.spend(len(rows)):
                initial_scores.append((loss(candidate.program, rows, fallback, spec.risk_quantile), candidate))
        if len(initial_scores) != len(candidates):
            raise ValueError("computation budget cannot compare the supplied explanations")
        best_initial = min(value for value, _ in initial_scores)
        plausible_ids = {candidate.candidate_id for value, candidate in initial_scores
                         if value <= best_initial + config.minimum_improvement}
        ambiguous = {identifier for group in before_classes if len(group) > 1
                     for identifier in group if identifier in plausible_ids}
        alternatives = [candidate for candidate in candidates if candidate.candidate_id in ambiguous]
        experiment, recalled, learned = None, [], None
        events = []
        remaining = config.observation_budget - observation_cost
        if len(alternatives) > 1 and remaining >= 4:
            offers = self.library.experiments.propose(
                spec, alternatives, rows, fallback=fallback, budget=research_work, remaining=remaining - 3,
                used=probes)
            recalled = list(self.library.experiments.last_recall)
            choices = offers or [probe for probe in construct_probes(
                spec, alternatives, round_number=0, seed=config.seed, max_history=config.max_history)
                if probe.cost <= remaining - 3 and all(probe.sequence != old.sequence for old in probes)]
            experiment, predicted_separation = choose_probe(choices, alternatives, fallback, research_work,
                                                            observations=rows)
            if experiment is not None and predicted_separation > 1e-8:
                acquired = environment.observe(experiment)
                observation_cost += len(acquired)
                learned = self.library.experiments.record(spec, alternatives, rows, experiment, acquired,
                                                          fallback=fallback, budget=research_work)
                self.library.experiments.feedback(experiment, supported=learned is not None)
                events.append({"action": "execute_recalled_experiment" if offers else "execute_new_experiment",
                    "probe": asdict(experiment), "observations": [asdict(row) for row in acquired],
                    "observation_cost": len(acquired), "predicted_separation": predicted_separation,
                    "learned_experiment": learned})
                rows.extend(acquired)
                probes.append(experiment)
        scores = []
        for candidate in candidates:
            if not research_work.spend(len(rows)):
                raise ValueError("computation budget exhausted before choosing an explanation")
            scores.append((loss(candidate.program, rows, fallback, spec.risk_quantile),
                           candidate.program.size, candidate))
        _, _, winner = min(scores, key=lambda row: row[:2])
        after_classes = observed_classes(candidates, rows, fallback, research_work)
        workspace = ResearchWorkspace(spec.task_id, observations=rows, reference=list(rows), queried=probes,
            incumbent=winner, evidence_champion=winner, bank=candidates, fallback=fallback,
            initial_loss=sum(point_loss(fallback, row.target, spec.risk_quantile) for row in initial_rows) / len(initial_rows))
        episode = CognitiveAgent(replace(config,
            observation_budget=config.observation_budget - observation_cost,
            compute_budget=config.compute_budget - research_work.spent), self.library).run(
                environment, research_state=workspace.to_dict())
        return {"task_id": spec.task_id, "status": "completed", "initial_classes": before_classes,
                "initial_candidate_losses": {candidate.candidate_id: value for value, candidate in initial_scores},
                "after_classes": after_classes, "selected_explanation": winner.candidate_id,
                "outcome_losses": {candidate.candidate_id: value for value, _, candidate in scores},
                "recall": recalled, "events": events, "episode": episode.to_dict(),
                "costs": {"observation_cost": observation_cost + episode.observation_cost,
                          "compute_cost": research_work.spent + episode.compute_cost,
                          "experimental_observation_cost": observation_cost,
                          "experimental_compute_cost": research_work.spent}}


class _ResponseLab:
    """Independent scalar measurements used by the executable transfer example."""

    def __init__(self, name, response, *, feature="x", upper=2., initial=None):
        self._response = response
        self.feature = feature
        controls = initial or ((0., .3, .7, 1.1), (1.4, 1.7, 1.9, 2.))
        self.spec = TaskSpec(name, "Predict the numerical response to controlled inputs.", (feature,),
            tuple(Probe(f"initial_{index}", tuple({feature: value} for value in values))
                  for index, values in enumerate(controls)), (), feature_bounds={feature: (0., upper)},
            max_probe_length=4)
        self.observation_cost = 0
        self.queried = []

    def observe(self, probe):
        history, rows = [], []
        for point in probe.sequence:
            value = point[self.feature]
            if not self.spec.feature_bounds[self.feature][0] <= value <= self.spec.feature_bounds[self.feature][1]:
                raise ValueError("probe is outside the published control range")
            target = self._response(value)
            rows.append(Observation(dict(point), target, tuple(history)))
            history.append({**point, "__target__": target})
        self.queried.append(probe)
        self.observation_cost += len(rows)
        return rows

    def evaluation_rows(self, seed=607, count=64):
        rng = Random(seed)
        return [Observation({self.feature: value}, self._response(value))
                for value in (rng.uniform(*self.spec.feature_bounds[self.feature]) for _ in range(count))]


def run_experiment_transfer(library=None):
    """Learn two explanations, learn their separating experiment, then replay it."""
    library = library if library is not None else KnowledgeLibrary()
    config = ResearchConfig(observation_budget=32, compute_budget=10000, max_rounds=2, seed=557)
    sources = [_ResponseLab("experimental_source_a", lambda value: value),
               _ResponseLab("experimental_source_b", lambda value: value * value)]
    source_episodes = [CognitiveAgent(config, library).run(environment) for environment in sources]
    candidates = [Candidate(f"learned_explanation_{index}", episode.program, "source_learning",
                            "Explanation learned by ordinary research on paid source observations",
                            memory_id=episode.concepts_added[-1] if episode.concepts_added else None)
                  for index, episode in enumerate(source_episodes)]
    initial = ((0., 1., 0., 1.),)
    learning_lab = _ResponseLab("experimental_question", lambda value: value * value, initial=initial)
    learned = ExperimentalResearch(library, config).run(learning_lab, candidates)
    target_lab = _ResponseLab("experimental_rebound", lambda value: value, feature="u", upper=4., initial=initial)
    transferred = ExperimentalResearch(library, config).run(target_lab, candidates, feature_mapping={"x": "u"})
    for report, environment in ((learned, learning_lab), (transferred, target_lab)):
        from .language import Program
        rows = environment.evaluation_rows()
        report["evaluation"] = {"mse": loss(Program.from_dict(report["episode"]["program"]), rows,
                                             report["episode"]["fallback"], .5),
                                "rows": len(rows), "compute_cost": len(rows)}
    summary = {
        "source_expressions": [episode.program.describe() for episode in source_episodes],
        "source_selected_explanation": learned["selected_explanation"],
        "target_selected_explanation": transferred["selected_explanation"],
        "actual_recalled_queries": sum(event["action"] == "execute_recalled_experiment"
                                       for event in transferred["events"]),
        "target_evaluation_mse": transferred["evaluation"]["mse"],
        "observation_cost": sum(episode.observation_cost for episode in source_episodes)
                            + learned["costs"]["observation_cost"] + transferred["costs"]["observation_cost"],
        "compute_cost": sum(episode.compute_cost for episode in source_episodes)
                        + learned["costs"]["compute_cost"] + transferred["costs"]["compute_cost"],
    }
    return {"status": "completed", "sources": [episode.to_dict() for episode in source_episodes],
            "learning": learned, "transfer": transferred, "summary": summary}
