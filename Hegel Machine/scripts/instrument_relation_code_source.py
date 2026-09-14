"""Export this study's supplied source knowledge as editable ordinary Python."""

from textwrap import dedent


def export_source(measured_program, instrument_binding, schema):
    """Translate the recorded source program; no current-task law is consulted."""
    relations = []

    def visit(node):
        if node['op'] == 'relation_sum':
            relations.append(node)
        for child in node.get('children', []):
            visit(child)

    visit(measured_program)
    if len(relations) != 1:
        raise ValueError('The supplied study source must contain one relation')
    relation = relations[0]
    radius = relation['children'][0]['value']

    def expression(node, *, edges=False):
        op = node['op']
        children = node.get('children', [])
        if op == 'const':
            return repr(float(node['value']))
        if op == 'feature':
            return f"float(row['features'][{node['name']!r}])"
        if op == 'lag':
            return f"float(row['history'][-{node['lag']}][{node['name']!r}])"
        if op == 'available':
            return f"float(len(row.get('history', ())) >= {node['lag']})"
        if op == 'arg' and edges:
            names = {'edge_source_position': 'position', 'edge_source_quantity': 'quantity',
                     'edge_target_position': 'other_position', 'edge_target_quantity': 'other_quantity'}
            return names[node['name']]
        if op == 'relation_sum':
            return '_relation_flow(row, radius)'
        if op in ('window_sum', 'event_delta'):
            return f"_{op}(lambda row: {expression(children[0])}, row, {node['lag']})"
        values = [expression(child, edges=edges) for child in children]
        if op in ('add', 'sub', 'mul'):
            operator = {'add': '+', 'sub': '-', 'mul': '*'}[op]
            return f'({values[0]} {operator} {values[1]})'
        if op in ('lt', 'ge'):
            return f"float({values[0]} {'<' if op == 'lt' else '>='} {values[1]})"
        if op in ('and', 'or'):
            return f'float(bool({values[0]}) {op} bool({values[1]}))'
        if op == 'if':
            return f'({values[1]} if {values[0]} else {values[2]})'
        if op == 'not':
            return f'float(not ({values[0]}))'
        if op == 'abs':
            return f'abs({values[0]})'
        if op in ('min', 'max'):
            return f"{op}({', '.join(values)})"
        if op == 'object_sum':
            return f"sum([{', '.join(values)}])"
        raise ValueError(f'Unsupported operation in the supplied source: {op}')

    predicate = expression(relation['children'][1], edges=True)
    parts = relation['children'][2:]
    objects = ',\n        '.join('(' + ', '.join(expression(part) for part in parts[i:i + 3]) + ')'
                                  for i in range(0, len(parts), 3))
    direction = relation.get('name')
    if direction in ('left', 'right'):
        distance = 'other_position - position' if direction == 'right' else 'position - other_position'
        active = f'0.0 < ({distance}) < radius'
        contribution = '(other_selector - selector) * quantity'
    else:
        active = 'abs(position - other_position) < radius'
        contribution = 'selector * (other_quantity - quantity)'
    calibration = ',\n        '.join(f'{name!r}: {expression(program)}'
                                     for name, program in instrument_binding.items())
    return dedent('''\
        """Editable source model and fitted input readings from the preceding task."""

        SOURCE_RADIUS = RADIUS
        OBJECT_SCHEMA = SCHEMA


        def calibrate(row):
            return {
                CALIBRATION
            }


        def _past(row, index):
            history = row.get('history', ())
            return {'features': history[index], 'history': history[:index]}


        def _window_sum(function, row, width):
            history = row.get('history', ())
            return function(row) + sum(function(_past(row, index))
                for index in range(max(0, len(history) - width + 1), len(history)))


        def _event_delta(function, row, lag):
            history = row.get('history', ())
            return 0.0 if len(history) < lag else function(row) - function(_past(row, len(history) - lag))


        def _relation_flow(row, radius):
            objects = [
                OBJECTS
            ]
            result = 0.0
            for index, (selector, position, quantity) in enumerate(objects):
                for other, (other_selector, other_position, other_quantity) in enumerate(objects):
                    if index != other and ACTIVE and PREDICATE:
                        result += CONTRIBUTION
            return result


        def predict_measured(row, radius=SOURCE_RADIUS):
            return RESPONSE


        def predict(row, radius=SOURCE_RADIUS):
            history = row.get('history', ())
            measured_history = []
            for index, previous in enumerate(history):
                fields = calibrate(_past(row, index))
                if '__target__' in previous:
                    fields['__target__'] = previous['__target__']
                measured_history.append(fields)
            return predict_measured({'features': calibrate(row), 'history': measured_history}, radius)
        ''').replace('RADIUS\n', repr(float(radius)) + '\n', 1).replace('= SCHEMA', '= ' + repr(schema), 1
        ).replace('CALIBRATION', calibration).replace('OBJECTS', objects).replace('ACTIVE', active
        ).replace('PREDICATE', predicate).replace('CONTRIBUTION', contribution
        ).replace('RESPONSE', expression(measured_program))
