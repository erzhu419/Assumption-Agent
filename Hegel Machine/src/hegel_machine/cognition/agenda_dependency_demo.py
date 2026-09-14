"""A shared research prerequisite consumed by two concrete successor tasks."""

import json
from pathlib import Path

from .agenda import RESOURCES, ResearchAgenda
from .environments import make_suite
from .evaluation import loss
from .language import Program
from .models import ResearchConfig


def dependency_agenda(directory, *, seed=71, successors=("parameter_transfer", "renamed_transfer")):
    config = ResearchConfig(seed=seed, observation_budget=48, compute_budget=9000, max_rounds=2)
    agenda = ResearchAgenda(directory=directory, observation_budget=240, compute_budget=60000)
    # These are declared research requirements. The runtime must actually
    # discover their answers and later engines must actually use the library.
    for identifier, index, offset in (("parameter_transfer", 5, 1), ("renamed_transfer", 14, 2)):
        if identifier in successors:
            agenda.add_task(identifier, source={"kind": "suite", "seed": seed + offset,
                "task_index": index}, config=config, depends_on=("shared_mechanism",))
    agenda.add_task("shared_mechanism", source={"kind": "suite", "seed": seed, "task_index": 0},
                    config=config)
    return agenda


def _summary(agenda):
    evaluation = {}
    for identifier, item in agenda.state["questions"].items():
        if item["result"] is None:
            continue
        source, episode = item["source"], item["result"]
        environment = make_suite(source["seed"])[source["task_index"]]
        rows = environment.evaluation_rows()
        evaluation[identifier] = {"loss": loss(Program.from_dict(episode["program"]), rows,
            episode["fallback"], environment.spec.risk_quantile), "rows": len(rows),
            "selected_operation": episode["selected_operation"],
            "selected_memory_id": episode["selected_memory_id"]}
    return {"status": agenda.state["status"], "spent": agenda.state["spent"],
            "visits": agenda.state["visits"], "evaluation": evaluation,
            "answers": {key: item.get("answer_concepts", []) for key, item in agenda.state["questions"].items()}}


def run_dependency_demo(directory, *, seed=71):
    """Compare one shared source with each successor funding its own source.

    Both arms execute the same declared prerequisites and successor apparatuses
    with the same per-visit resources. The comparator repeats required source
    research in separate libraries; it is not the ordinary-code Agent control.
    Final evaluation is read only after all learning and selection has ended.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    shared = dependency_agenda(directory / "shared", seed=seed)
    before = [{key: offer[key] for key in ("question_id", "demand_count", "expected_gain")}
              for offer in shared.offers()]
    blocked = {key: list(item["blocked_on"]) for key, item in shared.state["questions"].items()}
    checkpoint = shared.run(max_steps=1)
    after_source = {"visits": len(checkpoint["visits"]),
                    "answer_concepts": checkpoint["questions"]["shared_mechanism"]["answer_concepts"]}
    shared = ResearchAgenda(directory=directory / "shared", state=json.loads(json.dumps(checkpoint)))
    completed = shared.run()
    (directory / "shared_checkpoint.json").write_text(json.dumps(completed, indent=2) + "\n", encoding="utf-8")
    separate_agendas = []
    for identifier in ("parameter_transfer", "renamed_transfer"):
        individual = dependency_agenda(directory / identifier, seed=seed, successors=(identifier,))
        individual.run()
        separate_agendas.append(individual)
    separate = [_summary(individual) for individual in separate_agendas]
    report = {"kind": "shared_research_prerequisite", "seed": seed,
        "selection_basis": "declared successor demand; expected gains remain unknown until measured",
        "initial_offers": before, "initial_blocked": blocked, "after_source_before_resume": after_source,
        "shared": _summary(shared), "separate": separate,
        "separate_total_costs": {name: sum(item["spent"][name] for item in separate) for name in RESOURCES},
        "shared_minus_separate_costs": {name: shared.state["spent"][name]
            - sum(item["spent"][name] for item in separate) for name in RESOURCES}}
    (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
