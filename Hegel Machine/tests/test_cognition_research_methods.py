from hegel_machine.cognition.action_policy import ActionOption
from hegel_machine.cognition.research_methods import ResearchMethods


def step(action, gain, method_id=None):
    return {"action": action, "context": "numeric:residual", "gain": gain,
            "compute_cost": 100, "observation_cost": 8, "method_id": method_id}


def test_failed_steps_are_removed_then_the_method_is_tested_and_restricted_by_new_failure():
    book = ResearchMethods()
    condition = {"response": "numeric", "controls": 2}
    trace = [step("refit", 0.), step("representation", .7), step("task_value", .3)]
    identifier = book.observe("numeric:residual", trace, condition=condition, source_task="a")
    method = book.methods[0]
    assert method["original_steps"] == ["refit", "representation", "task_value"]
    assert method["steps"] == ["representation", "task_value"]
    restored = ResearchMethods(book.to_dict())
    options = [ActionOption("refit", 0, 1), ActionOption("representation", 0, 100), ActionOption("task_value", 8, 50)]
    chosen = restored.choose("numeric:residual", condition, options, [], task_id="b",
                             observation_remaining=30, compute_remaining=1000)
    assert chosen.name == "representation"
    next_trace = [step("representation", .4, identifier)]
    restored.observe("numeric:residual", next_trace, condition=condition, source_task="b")
    assert restored.methods[0]["status"] == "supported"
    assert restored.choose("numeric:residual", condition, options, next_trace, task_id="b",
                            observation_remaining=30, compute_remaining=1000).name == "task_value"
    failed = step("task_value", -.2, identifier)
    restored.observe("numeric:residual", [*next_trace, failed], condition=condition, source_task="b")
    assert condition in restored.methods[0]["excluded_conditions"]
    assert any(change["change"] == "restrict_failed_condition" for change in restored.revisions)
    assert restored.opportunity("numeric:residual", condition, task_id="a",
                                 observation_remaining=30, compute_remaining=1000) is not None


def test_neutral_preparation_preserves_the_unfinished_sequence_and_its_next_actual_step():
    book = ResearchMethods()
    condition = {"response": "numeric", "controls": 2}
    identifier = book.observe("numeric:residual", [step("refit", .2), step("representation", .8)],
                              condition=condition, source_task="source")
    book.observe("numeric:residual", [step("refit", .3)], condition=condition, source_task="other")
    original_gain = book.methods[0]["observed_gain"]
    trace = [step("refit", 0., identifier)]
    book.observe("numeric:residual", trace, condition=condition, source_task="target")
    assert book.methods[0]["observed_gain"] == original_gain
    restored = ResearchMethods(book.to_dict())
    options = [ActionOption("refit", 0, 10), ActionOption("representation", 0, 100)]
    assert restored.choose("numeric:residual", condition, options, trace, task_id="target",
                           observation_remaining=30, compute_remaining=1000).name == "representation"
    assert restored.last_choice["method_id"] == identifier
    trace.append(step("representation", .7, identifier))
    restored.observe("numeric:residual", trace, condition=condition, source_task="target")
    assert restored.methods[0]["observed_gain"] == .7
    assert restored.choose("numeric:residual", condition, options, trace, task_id="target",
                           observation_remaining=30, compute_remaining=1000).name == "refit"
