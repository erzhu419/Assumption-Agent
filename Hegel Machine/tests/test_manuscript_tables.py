from scripts.regenerate_manuscript_tables import recurrent_execution_row


def trial(*, fresh, terminal, calls, rows, seconds):
    return {"fresh_confirmation_passed": fresh, "terminal": terminal,
            "costs": {"model_calls": calls, "observation_cost": rows,
                      "python_execution_seconds": seconds}}


def test_recurrent_pass_requires_fresh_confirmation_and_terminal_success():
    trials = [trial(fresh=True, terminal={"passed": True}, calls=2, rows=264, seconds=10),
              trial(fresh=False, terminal={"passed": True}, calls=3, rows=264, seconds=20),
              trial(fresh=True, terminal={"passed": False}, calls=4, rows=360, seconds=30)]
    assert recurrent_execution_row("Original live", "Hegel", trials) == (
        "Original live", "Hegel", "1/3", 9, "888", "60.0")


def test_recurrent_execution_budget_failure_remains_in_denominator_and_costs():
    trials = [trial(fresh=True, terminal={"passed": True}, calls=4, rows=264, seconds=266.3095),
              trial(fresh=False, terminal=None, calls=9, rows=360, seconds=600.1372)]
    assert recurrent_execution_row("Development rerun", "Code", trials) == (
        "Development rerun", "Code", "1/2", 13, "624", "866.4")
