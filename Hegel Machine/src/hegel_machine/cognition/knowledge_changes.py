"""Notice learned changes that can make an unfinished experiment useful."""


_WITHDRAWN = {"revoked", "refuted", "counterexample"}
_ACCOUNTING = {"uses", "method_uses", "compute_cost", "measurement_cost", "observation_cost",
               "fit_loss", "training_loss", "validation_loss", "confirmation_loss",
               "support_rows", "validation_rows", "confirmation_rows", "max_squared_error",
               "fit_rows", "evidence_count", "search_compute", "loss"}
_METHOD_FIELDS = ("status", "solver_strategy", "sequence_mode", "coordinate_role",
                  "context_fractions", "predicate_pattern", "target_strategy",
                  "condition_patterns", "condition_intervals", "response_context",
                  "instrument_recipe", "temporal_recipe", "steps", "failure_scopes")


def _copy(value, *, method=False):
    """Keep executable content and evidence, excluding research accounting."""
    if isinstance(value, dict):
        ignored = _ACCOUNTING | ({"task_id", "source_task_id", "probe_id", "source_probe"} if method else set())
        return {key: _copy(item, method=method) for key, item in value.items() if key not in ignored}
    if isinstance(value, (list, tuple)):
        return [_copy(item, method=method) for item in value]
    return value


def _fields(record, names, *, method=False):
    return {name: _copy(record[name], method=method) for name in names if name in record}


def _instrument(model):
    result = _fields(model, ("status", "source_schema", "source_object_model", "command_features",
                             "command_bounds", "observed_history_depth", "refuted_projections"))
    result["fields"] = {}
    for name, field in model.get("fields", {}).items():
        item = _fields(field, ("program", "role", "members", "status", "kind", "observed_history_depth"))
        if "hypotheses" in field:
            item["hypotheses"] = [_fields(hypothesis, ("hypothesis_id", "program", "status", "kind"))
                                  for hypothesis in field["hypotheses"]]
        result["fields"][name] = item
    result["available"] = model.get("status") not in _WITHDRAWN and any(
        hypothesis.get("status") not in _WITHDRAWN
        for field in result["fields"].values() for hypothesis in field.get("hypotheses", [field]))
    return result


def _method(record):
    result = _fields(record, _METHOD_FIELDS, method=True)
    # Repeating a failed condition under another task label adds no new scope.
    scopes = []
    for scope in result.get("failure_scopes", []):
        if scope not in scopes:
            scopes.append(scope)
    if "failure_scopes" in result:
        result["failure_scopes"] = scopes
    result["available"] = record.get("status") == "supported"
    if "parent_method_id" in record:
        result["parent_method_id"] = record["parent_method_id"]
    return result


def _relation(prior, concept_status):
    result = _fields(prior, ("status", "source_features", "control_features", "object_schema",
                             "structure", "research_program", "input_program", "input_research_program",
                             "counterexamples"))
    result["concept_status"] = concept_status
    result["available"] = prior.get("status") not in _WITHDRAWN and concept_status not in _WITHDRAWN
    return result


def _draft(record):
    """Track usable unfinished computations, rather than their replay counters."""
    if record.get("risk_method") is not None:
        result = _fields(record, ("risk_quantile", "source_features", "control_features",
            "observed_features", "risk_method"))
        result["available"] = record["status"] in {"unresolved", "supported"}
        return result
    procedure = record["research_program"]
    result = _fields(record, ("risk_quantile", "source_schema", "source_features",
        "control_features", "observed_features", "input_template"))
    result["program"] = _copy(procedure["graph"]["nodes"][procedure["output_node"]]["answer"])
    result["steps"] = [_fields(step, ("operation", "width", "placement", "gate_names",
        "gate_coordinate", "threshold", "threshold_interval", "threshold_intervals",
        "width_candidates", "placement_candidates", "branch_basis", "evidence_contexts"))
        for step in procedure.get("selection_steps", ()) if not step.get("historical")]
    condition = record["condition_uncertainty"]
    result["conditions"] = [_fields(item, ("condition_field", "threshold_intervals",
        "threshold_interval", "width", "gate_width", "placement", "width_candidates",
        "placement_candidates", "status")) for item in condition.get("conditions", ())]
    result["available"] = not record.get("resolved_concept_id") and record["status"] == "unresolved"
    return result


def knowledge_snapshot(library):
    """Copy instrument laws, method applicability and unfinished computations.

    Session boundaries compare these transient snapshots. Support counters and
    repeated method successes do not change what experiment can be performed.
    Relation trajectories remain included because they constrain later ranges.
    """
    statuses = {concept.concept_id: concept.status for concept in library.concepts}
    return {
        "instrument": {model["model_id"]: _instrument(model) for model in library.instrument_models},
        "method": {method["method_id"]: _method(method) for method in library.program_experiment_methods.records},
        "relation": {prior["source_concept_id"]: _relation(prior, statuses.get(prior["source_concept_id"]))
                     for prior in library.meta_priors},
        "draft": {record["draft_id"]: _draft(record) for record in library.research_drafts
                  if record.get("condition_uncertainty") is not None or record.get("risk_method") is not None},
    }


def knowledge_changes(before, after):
    """Return changed knowledge IDs, without claiming that a change is useful.

    ``available`` distinguishes retained knowledge from withdrawals. Both can
    warrant reconsideration; only a priced current experiment warrants a visit.
    """
    changes = []
    for kind in ("instrument", "method", "relation", "draft"):
        old_records, new_records = before.get(kind, {}), after.get(kind, {})
        for identifier in dict.fromkeys([*old_records, *new_records]):
            old, new = old_records.get(identifier), new_records.get(identifier)
            if old == new:
                continue
            # A new ID or parent attribution alone does not make the same
            # executable method and the same applicability conditions novel.
            if kind in {"method", "draft"} and new is not None:
                content = {key: value for key, value in new.items() if key != "parent_method_id"}
                peers = [old] if old is not None else old_records.values()
                if any(content == {key: value for key, value in peer.items() if key != "parent_method_id"}
                       for peer in peers):
                    continue
            available = bool(new and new["available"])
            change = ("withdrawn" if new is None or not available else "added" if old is None else "revised")
            changed = [key for key in dict.fromkeys([*(old or {}), *(new or {})])
                       if (old or {}).get(key) != (new or {}).get(key)]
            record = {"kind": kind, "knowledge_id": identifier, "change": change,
                      "available": available, "changed_fields": changed}
            if kind == "method" and new and "parent_method_id" in new:
                record["parent_method_id"] = new["parent_method_id"]
            changes.append(record)
    return changes
