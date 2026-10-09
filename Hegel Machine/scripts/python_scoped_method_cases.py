"""Separate obsolete response evidence from an instrument-family change."""

from scripts.python_selective_method_cases import (
    KINDS as PREVIOUS_KINDS, case_spec as previous_case_spec, protocol as previous_protocol,
)

NEW_KIND = "post_validation_instrument_change"
KINDS = (*PREVIOUS_KINDS, NEW_KIND)


def case_spec(source, kind, *, seed_offset, response_offset):
    if kind != NEW_KIND:
        return previous_case_spec(source, kind,
            seed_offset=seed_offset, response_offset=response_offset)
    spec = previous_case_spec(source, "response_parameter_change",
        seed_offset=seed_offset, response_offset=response_offset)
    spec["kind"] = kind
    spec["task"]["calibration_change"] = {"after_rows": 96, "instrument_mode": "recurrent"}
    return spec


def protocol(sources):
    """Keep every previous case intact; append one structural-change case per source."""
    plan = previous_protocol(sources)
    admitted = [(index, source) for index, source in enumerate(sources) if source["candidate"] is not None]
    plan["practice_cases"]["case_4"] = case_spec(sources[plan["author_source_index"]], NEW_KIND,
        seed_offset=900011, response_offset=1000003)
    plan["transfer_cases"].extend({"id": f"target_{position}_4", "source_index": source_index,
        **case_spec(source, NEW_KIND, seed_offset=1100009, response_offset=1200007)}
        for position, (source_index, source) in enumerate(admitted[1:]))
    plan.update(study="python_scoped_method_development", target_kinds=list(KINDS),
        calibration_change="The fourth and fifth cases share the same initial finite-history "
            "instrument, changed response, paid prefix and public 96-row checkpoint. After the "
            "checkpoint the fourth changes response parameters; the fifth changes the instrument "
            "from finite history to recurrent state, retaining its numerical parameters and "
            "response. All subsequent complete trajectories use the settled apparatus.",
        intervention="Revise the previous method on five anonymous practice cases and freeze it "
            "for ten cross-source targets. Preserve the four previous cases and their exact "
            "settings. Reuse the same refit, record selection, measurement and response operators.",
        failure_policy="Retain every author and practice attempt. Unpublished revisions leave "
            "their ten target slots unexecuted; reference outcomes remain separate.",
        scope="Two existing archives under five conditions. The new condition distinguishes "
            "obsolete response evidence from inadequate finite-history prediction after a "
            "recurrent instrument change. This is development within existing apparatus families.")
    return plan
