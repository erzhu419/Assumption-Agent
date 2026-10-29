"""Retain a structural fit as the existing executable coordinate-refit plan.

The selected spatial and temporal computations remain fixed during this replay.
Their structure and measured boundary remain available for ordinary structural
research when later observations require a different computation.
"""

from copy import deepcopy

from .continuous_state import _sum_terms
from .language import Program
from .mechanism_revision import _signed_terms
from .nonlinear_parameters import _coefficient
from .research_graph import ResearchGraph
from .research_programs import _plan


def structural_fit_contract(entry):
    """Read actual fitted columns, or the surviving terms of an older program.

    Old snapshots did not save zero-weight columns. Recover only their actual
    executable coordinates; response_terms is not the entire fitted basis.
    No observations are evaluated and no parameters are fitted here.
    """
    return structural_program_fit_contract(entry['candidate'].program,
        entry['structure'], loss=entry['training_loss'])


def structural_program_fit_contract(program, structure, loss=0.0):
    """Read the saved structural solve without manufacturing a candidate."""
    record = structure['fits'][0]
    if 'fit_terms' in record and 'coefficients' in record:
        terms = tuple(Program.from_dict(term) for term in record['fit_terms'])
        coefficients = list(record['coefficients'])
        basis = 'complete_fitted_columns'
    else:
        parsed = [_coefficient(term) for term in _signed_terms(program)]
        unit = Program('const', value=1.)
        intercept = sum(weight for weight, term in parsed if term == unit)
        active = [(weight, term) for weight, term in parsed if term != unit]
        terms = tuple(term for _, term in active)
        coefficients = [intercept, *(weight for weight, _ in active)]
        basis = 'saved_executable_terms'
    if len(coefficients) != len(terms) + 1 or _sum_terms(
            terms, coefficients[1:], coefficients[0]) != program:
        raise ValueError('Structural fit columns do not reconstruct the saved executable program')
    return {'fit_terms': terms, 'coefficients': coefficients, 'program': program,
            'loss': loss, 'coordinate_basis': basis}


def research_program_fit_contract(payload):
    """Recover the saved linear readout, including its zero-weight columns."""
    graph = ResearchGraph.from_dict(payload['graph'])
    keys = payload['term_nodes']
    terms = tuple(graph.nodes[key].answer for key in keys)
    output = graph.nodes[payload['output_node']]
    if output.expression is None or output.answer is None or any(term is None for term in terms):
        raise ValueError('Saved research program has an unfinished fit contract')
    weights, intercept = dict.fromkeys(keys, 0.0), 0.0
    unit = Program('const', value=1.)
    for term in _signed_terms(output.expression):
        weight, coordinate = _coefficient(term)
        if coordinate == unit:
            intercept += weight
        elif coordinate.op == 'arg' and coordinate.name in weights:
            weights[coordinate.name] += weight
        else:
            raise ValueError('Saved research output is not a readout of its declared columns')
    coefficients = [intercept, *(weights[key] for key in keys)]
    rebuilt = _sum_terms(terms, coefficients[1:], coefficients[0])
    if rebuilt != output.answer and rebuilt.expanded().simplified() != output.answer.expanded().simplified():
        raise ValueError('Research fit columns do not reconstruct the saved executable program')
    return {'fit_terms': terms, 'coefficients': coefficients, 'program': output.answer,
            'loss': output.loss, 'coordinate_basis': 'saved_research_graph_columns'}


def attach_structural_research_program(entry, *, fit_rows, source_ref=None):
    """Keep the selected answer and attach a replayable, still unconfirmed plan."""
    retained = dict(entry)
    if retained.get('research_program'):
        return retained
    fit = structural_fit_contract(entry)
    if not fit['fit_terms']:
        return retained
    structure = entry['structure']
    plan = _plan(fit, [], fit_rows, structure['compute_cost'])
    record = structure['fits'][0]
    plan['structural_replay'] = {
        'coordinate_basis': fit['coordinate_basis'],
        'parameter_scope': 'linear_readout_weights_on_saved_structural_coordinates',
        **{key: deepcopy(record[key]) for key in ('family', 'radius', 'window', 'radius_interval')
           if key in record},
        'structure_revision_entry': 'propose_structures'}
    if source_ref is not None:
        plan['source_fit_ref'] = deepcopy(source_ref)
    retained['research_program'] = plan
    return retained
