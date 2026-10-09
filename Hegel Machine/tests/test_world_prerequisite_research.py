from hegel_machine.cognition.world_prerequisite_research import next_prerequisite, operation_failures


def failed(index, action, target, message):
    return {"index": index, "action": {"action": action, "arg1": target},
            "feedback": {"success": False, "errors": []},
            "observation": {"lastActionMessage": message}}


def reactor_failures():
    return [failed(101, "READ", 10581, "That object (crystal reactor (uncalibrated)) is not readable."),
            failed(102, "OPEN", 10581, "That object (crystal reactor (uncalibrated)) is not openable/closeable."),
            failed(103, "ACTIVATE", 10581, "That object (crystal reactor (uncalibrated)) is not activatable/deactivatable.")]


def test_real_unsupported_messages_remain_distinct_from_access_failure():
    records = reactor_failures() + [failed(43, "TELEPORT_TO_OBJECT", 33979,
        "I couldn't find a valid location beside the object (generator core (50% activated), uuid: 33979) to teleport to.")]
    grouped = operation_failures(records)
    assert {item["kind"] for item in grouped[10581]} == {"unsupported_action"}
    assert grouped[33979][0]["kind"] == "inaccessible"
    assert grouped[10581][0]["reason"] == records[0]["observation"]["lastActionMessage"]


def test_missing_argument_does_not_claim_object_is_inaccessible():
    records = [failed(19, "ACTIVATE", 10581, "Action 'activate' requires argument 1.")]
    assert operation_failures(records)[10581][0]["kind"] == "operation_failed"
    assert next_prerequisite(records * 2, [], {}) is None


def test_explicit_api_access_error_is_preserved_and_classified_for_second_argument():
    record = {"index": 153, "action": {"action": "USE", "arg1": 16419, "arg2": 17557},
        "feedback": {"errors": ["arg2: Could not find object with UUID '17557' Are you sure it's accessible (i.e in your inventory, or directly in front of you?). "],
                     "success": False},
        "observation": {"lastActionMessage": "Action 'use object' requires two argument objects. Missing argument 2."}}
    failure = operation_failures([record])[17557][0]
    assert failure["kind"] == "inaccessible"
    assert "Could not find object with UUID '17557'" in failure["reason"]
    assert "Missing argument 2" in failure["reason"]


def test_repeated_real_failure_opens_prerequisite_without_supplying_solution():
    visible = {"accessibleEnvironmentObjects": [{"uuid": 10581, "name": "crystal reactor (uncalibrated)",
                                                  "description": "public description"}]}
    focus = next_prerequisite(reactor_failures(), [], visible)
    assert focus["target"] == 10581
    assert focus["question"]["id"] == "operation_10581"
    assert focus["question"]["origin"] == "execution_failure"
    assert focus["question"]["parent_id"] == "task"
    assert focus["public_object"] == visible["accessibleEnvironmentObjects"][0]
    assert focus["evidence"] == [101, 102, 103]
    assert "next_action" not in focus and "answer" not in focus


def test_supported_child_suppresses_repeat_but_open_child_retains_identity():
    question = next_prerequisite(reactor_failures(), [], {})["question"]
    assert next_prerequisite(reactor_failures(), [{**question, "status": "supported"}], {}) is None
    repeated = next_prerequisite(reactor_failures(), [{**question, "status": "open"}], {})
    assert repeated["question"]["id"] == question["id"]


def test_failures_on_different_objects_do_not_manufacture_repeated_failure():
    records = [failed(0, "READ", 1, "That object (meter) is not readable."),
               failed(1, "READ", 2, "That object (switch) is not readable.")]
    assert next_prerequisite(records, [], {}) is None


def test_resolved_object_does_not_hide_another_unresolved_prerequisite():
    records = reactor_failures() + [failed(104, "READ", 34810, "That object (reactor) is not readable."),
                                    failed(105, "ACTIVATE", 34810, "That object (reactor) is not activatable/deactivatable.")]
    assert next_prerequisite(records, [], {})["target"] == 34810
    assert next_prerequisite(records, [{"id": "known_interface", "target": 34810, "status": "supported"}], {})["target"] == 10581


def test_successful_action_and_navigation_failure_do_not_open_object_question():
    records = reactor_failures()[:1]
    records.append({**records[0], "index": 104, "feedback": {"success": True}})
    records.append(failed(105, "MOVE_DIRECTION", "north", "I can't move there. There is something in the way (table)."))
    assert next_prerequisite(records, [], {}) is None


def test_existing_successful_dialog_is_exposed_as_paid_evidence_without_auto_answer():
    dialog = {"is_in_dialog": True, "dialogText": "Public controls", "dialogOptions": {"1": "Inspect"}}
    records = reactor_failures() + [{"index": 104, "action": {"action": "TALK", "arg1": 10581},
        "feedback": {"success": True, "errors": []},
        "observation": {"lastActionMessage": "A dialog opens.", "dialog_box": dialog}}]
    focus = next_prerequisite(records, [], {})
    assert focus["successful_interactions"] == [{"index": 104,
        "action": {"action": "TALK", "arg1": 10581}, "message": "A dialog opens.", "dialog_box": dialog}]
    assert "answer" not in focus["question"] and focus["question"].get("status") != "supported"
    records[-1]["observation"]["dialog_box"] = {"is_in_dialog": False}
    assert next_prerequisite(records, [], {})["successful_interactions"] == []


def test_successful_use_with_object_as_second_argument_preserves_real_dialog_evidence():
    dialog = {"is_in_dialog": True, "dialogIn": "Hello, I am Crystal Reactor #3.\nThe current resonance frequence is: 5000 Hertz.",
              "dialogOptions": {"1": "Increase frequency by 1000 Hz", "9": "Exit"}}
    record = {"index": 139, "action": {"action": "USE", "arg1": 17557, "arg2": 10581},
              "feedback": {"errors": [], "success": True}, "observation": {"dialog_box": dialog}}
    focus = next_prerequisite([*reactor_failures(), record], [], {})
    assert focus["successful_interactions"] == [{"index": 139, "action": record["action"],
                                                  "message": "", "dialog_box": dialog}]
