"""Which historical coordinates have actually been distinguished by observations."""

from .language import Program


def history_branch_support(program: Program, observations):
    """Describe unobserved or current-input-aliased historical computations.

    An ``available`` guard alone does not imply a historical mechanism. The
    reads inside its branches must actually use a lag, window, event or state.
    This describes the acquired controls; it never predicts an unknown target.
    """
    observations = tuple(observations)
    observed_depth = max((len(row.history) for row in observations), default=0)
    historical = {}

    def visit(node, startup_depth=0):
        if node.op in {"lag", "window_sum", "event_delta", "accumulate"}:
            key = (node.op, max(startup_depth, node.history_depth), tuple(sorted(node.observed_variables())))
            historical.setdefault(key, node)
        if node.op == "if" and node.children[0].op == "available":
            guard, mature, startup = node.children
            visit(mature, max(startup_depth, guard.lag))
            visit(startup, startup_depth)
            return
        for child in node.children:
            visit(child, startup_depth)

    visit(program.expanded().simplified())
    unresolved = []
    for (operation, depth, variables), node in historical.items():
        if depth > observed_depth:
            unresolved.append({"operation": operation, "depth": depth,
                               "variables": [list(item) for item in variables],
                               "reason": "history_not_observed"})
        elif operation == "lag":
            mature = [row for row in observations if len(row.history) >= depth]
            current_names = set(mature[0].features) if mature else set()
            aliases = sorted(name for name in current_names if all(
                row.history[-node.lag][node.name] == row.features[name] for row in mature))
            if aliases:
                unresolved.append({"operation": operation, "depth": depth,
                                   "variables": [list(item) for item in variables],
                                   "reason": "history_equals_current_input", "current_aliases": aliases})
    return {"required_depth": max((key[1] for key in historical), default=0),
            "observed_depth": observed_depth, "unresolved": unresolved}


def history_frontier_key(support):
    """One comparison slot per unresolved temporal coordinate structure."""
    return tuple(sorted((item["operation"], item["depth"],
                         tuple(tuple(variable) for variable in item["variables"]))
                        for item in support["unresolved"]))
