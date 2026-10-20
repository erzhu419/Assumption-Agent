from scripts.regenerate_manuscript_tables import (
    boundary_transfer_rows,
    complete_parameter_groups,
    generate_differences,
    generate_tables,
    recurrent_execution_row,
)


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


def test_boundary_table_keeps_false_confirmations_and_unconfirmed_control_distinct():
    passed = {"case": "a", "fresh_confirmation_passed": True, "terminal": {"passed": True}}
    false_confirmation = {"case": "b", "fresh_confirmation_passed": True,
                          "terminal": {"passed": False}}
    result = {"retention": [passed], "new_parameters": [passed, false_confirmation],
              "frozen_source_answer": [{"case": "a", "terminal": {"passed": False}}]}
    protocol = {"transfer_cases": [
        {"id": "a", "parameter_pair": 0, "apparatus_seed": 17},
        {"id": "b", "parameter_pair": 0, "apparatus_seed": 17}]}
    assert boundary_transfer_rows(result, protocol) == [
        ("Earlier cohort / accepted 11/12 method / old 10", "1/1", "1/1", "—"),
        ("Earlier cohort / accepted 11/12 method / new 4", "2/2", "1/2", "0/1"),
        ("Earlier cohort / frozen source predictor / new 4", "Not run", "0/1", "0/1")]


def test_parameter_group_requires_all_registered_stages_and_confirmation():
    protocol = {"transfer_cases": [
        {"id": "a", "parameter_pair": 0, "apparatus_seed": 17},
        {"id": "b", "parameter_pair": 0, "apparatus_seed": 17},
        {"id": "c", "parameter_pair": 1, "apparatus_seed": 18},
        {"id": "d", "parameter_pair": 1, "apparatus_seed": 18}]}
    cases = [{"case": name, "fresh_confirmation_passed": name != "b",
              "terminal": {"passed": True}} for name in ("a", "b", "c")]
    assert complete_parameter_groups(cases, protocol) == "0/2"
    assert complete_parameter_groups(cases, protocol, require_confirmation=False) == "1/2"


def test_saved_adaptive_transfer_table_keeps_cohorts_and_paired_costs_separate():
    table = generate_tables()["9"]
    assert "Earlier cohort / accepted 11/12 method / new 4 | 4/4 | 2/4 | 1/2" in table
    assert "Fresh cohort / published 12/12 method / old 10 | 10/10 | 10/10 | —" in table
    assert "Fresh cohort / published 12/12 method / new 8 | 8/8 | 8/8 | 4/4" in table
    assert "Fresh cohort / accepted 11/12 method / same 8 | 4/8 | 0/8 | 0/4" in table
    assert "Fresh cohort / frozen source predictor / same 8 | Not run | 0/8 | 0/4" in table
    differences = generate_differences()
    assert differences["adaptive_boundary_new_parameter_costs"] == {
        "observation_cost": 1862, "compute_cost": 433837, "model_calls": 0}
    assert differences["accepted_boundary_same_parameter_costs"] == {
        "observation_cost": 1736, "compute_cost": 171847, "model_calls": 0}


def test_saved_window_table_preserves_no_model_failures_and_old_task_costs():
    table = generate_tables()["10"]
    assert "14/14 method / original ten | 10/10 | 10/10 | — | 2,160" in table
    assert "14/14 method / earlier boundary eight | 8/8 | 8/8 | — | 1,862" in table
    assert "14/14 method / earlier composition eight | 8/8 | 8/8 | — | 6,464" in table
    assert "14/14 method / new window eight | 8/8 | 8/8 | 4/4 | 6,464" in table
    assert "13/13 method / same window eight | 0/8 | 0/8 | 0/4 | 16,960" in table
    assert "Fixed source predictor / same window eight | Not run | 0/8 | 0/4 | —" in table
