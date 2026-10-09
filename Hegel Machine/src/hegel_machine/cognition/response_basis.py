"""Recover response coordinates without carrying a contradicted relation."""

from dataclasses import replace

from .relation_revisions import _expose


def relation_free_response_terms(program, budget):
    """Propose the linear response background of an executable explanation.

    Calls are opened with their current argument bindings. Linear temporal
    consumers distribute over additive terms, while a nonlinear consumer of a
    relation stays out of this basis. Its full expression remains available to
    parameter research. Coefficients of every returned term must be refitted.
    """
    opened = _expose(program, budget)
    if opened is None:
        return []
    relations = {}

    def contains_relation(node):
        if node not in relations:
            relations[node] = node.op in {"neighbor_sum", "relation_sum"} or any(
                contains_relation(child) for child in node.children)
        return relations[node]

    def terms(node):
        if node.op in {"add", "sub"} or (node.op == "object_sum" and contains_relation(node)):
            return [term for child in node.children for term in terms(child)]
        if node.op == "mul":
            for index, child in enumerate(node.children):
                if child.op == "const":
                    return [] if child.value == 0 else terms(node.children[1 - index])
        if node.op in {"window_sum", "event_delta"} and contains_relation(node):
            return [replace(node, children=(term,)) for term in terms(node.children[0])]
        return [] if contains_relation(node) else [node]

    def canonical_sums(node):
        children = tuple(canonical_sums(child) for child in node.children)
        if node.op == "object_sum":
            # Rebinding the same object collection can permute its summands.
            # They are one proposed reduction, not four copied old responses.
            children = tuple(sorted(children, key=lambda child: child.describe()))
        return replace(node, children=children)

    return list(dict.fromkeys(canonical_sums(term).simplified()
        for term in terms(opened) if term.op != "const"))
