# Frozen paid-data predictor; controls and recorded history only.
_PROGRAM = ('add', None, None, 0, (('add', None, None, 0, (('add', None, None, 0, (('mul', None, None, 0, (('const', -0.4612373668633239, None, 0, ()), ('if', None, None, 0, (('ge', None, None, 0, (('feature', None, 'c', 0, ()), ('const', -0.17615689, None, 0, ()))), ('window_sum', None, None, 3, (('mul', None, None, 0, (('feature', None, 'z', 0, ()), ('feature', None, 'z', 0, ()))),)), ('const', 0.0, None, 0, ()))))), ('mul', None, None, 0, (('const', 0.7065292517659245, None, 0, ()), ('if', None, None, 0, (('ge', None, None, 0, (('feature', None, 'c', 0, ()), ('const', -0.17615689, None, 0, ()))), ('const', 0.0, None, 0, ()), ('window_sum', None, None, 3, (('mul', None, None, 0, (('feature', None, 'z', 0, ()), ('feature', None, 'z', 0, ()))),)))))))), ('add', None, None, 0, (('mul', None, None, 0, (('const', -0.9391812239147984, None, 0, ()), ('if', None, None, 0, (('ge', None, None, 0, (('feature', None, 'c', 0, ()), ('const', -0.17615689, None, 0, ()))), ('window_sum', None, None, 3, (('feature', None, 'x', 0, ()),)), ('const', 0.0, None, 0, ()))))), ('mul', None, None, 0, (('const', 0.8153019571292132, None, 0, ()), ('if', None, None, 0, (('ge', None, None, 0, (('feature', None, 'c', 0, ()), ('const', -0.17615689, None, 0, ()))), ('const', 0.0, None, 0, ()), ('window_sum', None, None, 3, (('feature', None, 'x', 0, ()),)))))))))), ('add', None, None, 0, (('mul', None, None, 0, (('const', -0.12600507951005507, None, 0, ()), ('if', None, None, 0, (('ge', None, None, 0, (('feature', None, 'c', 0, ()), ('const', -0.17615689, None, 0, ()))), ('const', 0.0, None, 0, ()), ('window_sum', None, None, 3, (('const', 1.0, None, 0, ()),)))))), ('mul', None, None, 0, (('const', -0.27472345792022435, None, 0, ()), ('if', None, None, 0, (('ge', None, None, 0, (('feature', None, 'c', 0, ()), ('const', -0.17615689, None, 0, ()))), ('window_sum', None, None, 3, (('const', 1.0, None, 0, ()),)), ('const', 0.0, None, 0, ())))))))))

def _eval(node, features, history):
    op, value, name, lag, children = node
    if op == "const":
        return float(value)
    if op == "feature":
        return float(features[name])
    if op == "available":
        return float(len(history) >= lag)
    if op == "lag":
        if len(history) < lag:
            raise ValueError("insufficient control history")
        return float(history[-lag][name])
    if op == "if":
        return _eval(children[1 if _eval(children[0], features, history) else 2], features, history)
    if op == "object_sum":
        return sum(_eval(child, features, history) for child in children)
    if op == "accumulate":
        state = _eval(children[1], features, history)
        for index, point in enumerate(history):
            state = value * state + _eval(children[0], point, history[:index])
        return value * state + _eval(children[0], features, history)
    if op == "event_delta":
        if len(history) < lag:
            return 0.0
        index = len(history) - lag
        return (_eval(children[0], features, history)
                - _eval(children[0], history[index], history[:index]))
    if op == "window_sum":
        start = max(0, len(history) - lag + 1)
        return _eval(children[0], features, history) + sum(
            _eval(children[0], history[index], history[:index])
            for index in range(start, len(history)))
    left = _eval(children[0], features, history)
    if op == "abs":
        return abs(left)
    if op == "not":
        return float(not left)
    right = _eval(children[1], features, history)
    if op == "add":
        return left + right
    if op == "sub":
        return left - right
    if op == "mul":
        return left * right
    if op == "min":
        return min(left, right)
    if op == "max":
        return max(left, right)
    if op == "ge":
        return float(left >= right)
    if op == "lt":
        return float(left < right)
    if op == "and":
        return float(bool(left) and bool(right))
    if op == "or":
        return float(bool(left) or bool(right))
    raise ValueError("unsupported operator: " + op)

def predict(row):
    features = row["features"] if isinstance(row, dict) else row.features
    history = row["history"] if isinstance(row, dict) else row.history
    return float(_eval(_PROGRAM, features, history))
