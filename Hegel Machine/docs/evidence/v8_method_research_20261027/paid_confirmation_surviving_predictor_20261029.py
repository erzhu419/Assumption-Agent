# Frozen paid-data predictor; controls and recorded history only.
_PROGRAM = ('state_replay', None, None, 0, (('if', None, None, 0, (('available', None, None, 1, ()), ('if', None, None, 0, (('ge', None, None, 0, (('lag', None, '__target__', 1, ()), ('const', 0.5, None, 0, ()))), ('ge', None, None, 0, (('feature', None, 'u', 0, ()), ('const', -0.25, None, 0, ()))), ('ge', None, None, 0, (('feature', None, 'u', 0, ()), ('const', 0.55, None, 0, ()))))), ('ge', None, None, 0, (('feature', None, 'u', 0, ()), ('const', 0.475, None, 0, ()))))),))

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

# Replay a learned transition from reset using only its own output history.
_scalar_eval = _eval

def _eval(node, features, history):
    if node[0] != "state_replay":
        return _scalar_eval(node, features, history)
    predicted_history = []
    for point in (*history, features):
        controls = {name: value for name, value in point.items() if name != "__target__"}
        value = _eval(node[4][0], controls, predicted_history)
        predicted_history.append({**controls, "__target__": value})
    return value
