"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_09", "depth": 3, "fanout_by_node": {"node_00": 1, "node_01": 1, "node_02": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__", "node_00"], "node_02": ["node_01"]}, "max_fanout": 1, "node_code_shas": {"node_00": "79ce137f6695cf934dbedd83284029b3772f9ffd425fa6459c408910d08857fa", "node_01": "6ba9b4796b2c4c34208b3dd255c1a02e2c663bc6c35325baf458de5ba27424e3", "node_02": "a1f21591641389e1d2b4545bedbc10892d3880e5005c5e836e1b465118487e37"}, "shared_fanout_nodes": [], "sink_node_id": "node_02", "topo_order": ["node_00", "node_01", "node_02"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\n\\n\\ndef _features(u):\\n    n, d = u.shape\\n    channel = np.broadcast_to(np.eye(d), (n, d, d))\\n    neighbor = np.roll(u, -1, axis=1)\\n    return np.concatenate((channel, channel * u[..., None],\\n                           neighbor[..., None], (u * neighbor)[..., None],\\n                           (u ** 3)[..., None]), axis=2).reshape(n * d, 2 * d + 3)\\n\\n\\ndef _fit(u, v):\\n    design = _features(u)\\n    y = v.ravel()\\n    d = u.shape[1]\\n    weights = np.ones(len(u))\\n    for scale in (0.15, 0.05, 0.015, 0.004, 0.0015, 0.0015):\\n        w = np.repeat(weights, d)\\n        p = np.linalg.lstsq(design * w[:, None], y * w, rcond=None)[0]\\n        residual = np.linalg.norm((design @ p).reshape(v.shape) - v, axis=1)\\n        weights = 1 / np.hypot(1, residual / scale)\\n    keep = np.repeat(np.argsort(np.argsort(residual)) < len(u) - 3, d)\\n    return np.linalg.lstsq(design[keep], y[keep], rcond=None)[0]\\n\\n\\ndef _invert(v, p):\\n    d = v.shape[1]\\n    offset, gain = p[:d], p[d:2*d]\\n    coupling, interaction, cubic = p[-3:]\\n    j = np.arange(d)\\n    nxt = (j + 1) % d\\n    linear = np.diag(gain)\\n    linear[j, nxt] = coupling\\n    x = np.linalg.solve(linear, (v - offset).T).T\\n\\n    def error(x):\\n        neighbor = x[:, nxt]\\n        return offset + gain * x + (coupling + interaction * x) * neighbor + cubic * x ** 3 - v\\n\\n    for _ in range(25):\\n        r = error(x)\\n        if np.max(np.abs(r), initial=0.0) < 1e-11:\\n            break\\n        jac = np.zeros((len(x), d, d))\\n        jac[:, j, j] = gain + interaction * x[:, nxt] + 3 * cubic * x ** 2\\n        jac[:, j, nxt] = coupling + interaction * x\\n        step = np.linalg.solve(jac, r[..., None])[..., 0]\\n        trial = x - step\\n        baseline = np.linalg.norm(r, axis=1)\\n        for _ in range(12):\\n            worse = np.linalg.norm(error(trial), axis=1) > baseline\\n            if not np.any(worse):\\n                break\\n            step[worse] *= 0.5\\n            trial[worse] = x[worse] - step[worse]\\n        x = trial\\n    return x\\n\\n\\nfrom itertools import combinations_with_replacement\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _polynomial(x, degree):\\n    return np.column_stack([np.prod(x[:, q], axis=1)\\n                            for k in range(degree + 1)\\n                            for q in combinations_with_replacement(range(x.shape[1]), k)])\\n\\n\\ndef _index(d, y, n, mode):\\n    a = d[:n]\\n    if mode == 0:\\n        # A cubic index with a bounded nonlinear correction of that same index.\\n        initial = np.linalg.lstsq(a, y, rcond=None)[0]\\n        penalty = np.r_[np.zeros(15), np.full(20, np.sqrt(0.002))]\\n        def predict(p, x):\\n            z = x @ p[:-3]\\n            return z + p[-2] * np.tanh(p[-1] * z + p[-3])\\n        fits = [least_squares(lambda p: np.r_[predict(p, a)-y, penalty*p[:-3]],\\n                             np.r_[initial, 0., .2, f], max_nfev=100,\\n                             bounds=(np.r_[np.full(37, -np.inf), .2], np.r_[np.full(37, np.inf), 4.]))\\n                for f in (1., 2.5)]\\n        return predict(min(fits, key=lambda p: p.cost).x, d)\\n\\n    # Profile the linear index out of the bounded additive correction.\\n    # Penalize cubic terms more than quadratic terms to stabilize extrapolation.\\n    degree = 2 if mode in (1, 2) else 3\\n    m = 15 if degree == 2 else 35\\n    a = a[:, :m]\\n    interaction = d[:, :15 if mode in (2, 4) else 5]\\n    b = interaction[:n]\\n    reg = np.r_[np.zeros(5), np.full(10, .003), np.full(m-15, .035)]\\n    ext = np.vstack((a, np.diag(np.sqrt(reg))))\\n    pinv = np.linalg.pinv(ext)\\n    target = np.r_[y, np.zeros(m)]\\n    py = target - ext @ (pinv @ target)\\n    project = np.eye(n+m) - ext @ pinv\\n    proj = project[:, :n]\\n    shrink = np.r_[.002, np.full(4, .012), np.full(b.shape[1]-5, .040)]\\n    def fun(p):\\n        h = np.tanh(b @ p[:-1])\\n        return np.r_[py - p[-1]*(proj@h), shrink*p[:-1]]\\n    def jac(p):\\n        h = np.tanh(b @ p[:-1]); g = np.empty((n+m+len(shrink), len(p)))\\n        g[:n+m, :-1] = -p[-1]*(proj @ ((1-h*h)[:,None]*b))\\n        g[:n+m, -1] = -(proj@h)\\n        g[n+m:] = 0.; g[n+m:, :-1] = np.diag(shrink)\\n        return g\\n    base = np.linalg.lstsq(b, y, rcond=None)[0]\\n    rng = np.random.default_rng(103)\\n    starts = [base, -base, *rng.normal(0, .65, size=(5,len(base)))]\\n    fits = [least_squares(fun, np.r_[q, .22], jac=jac, max_nfev=75,\\n                          bounds=(np.r_[np.full(len(q), -np.inf), -.65], np.r_[np.full(len(q), np.inf), .65]))\\n            for q in starts]\\n    p = min(fits, key=lambda w:w.cost).x\\n    correction = p[-1]*np.tanh(interaction@p[:-1])\\n    coef = pinv @ (target - np.r_[correction[:n], np.zeros(m)])\\n    return d[:, :m] @ coef + correction\\n\\n\\ndef _coordinates(inputs, mode, chart):\\n    study = inputs[\'__input__\']; panel = study[\'panel_a\']\\n    train = study[\'train\']; n = len(train[\'x\'])\\n    x = _invert(np.vstack((train[\'x\'], study[\'test\'][\'x\'])), _fit(panel[\'u\'], panel[\'v\']))\\n    b = study[\'panel_b\']; operator = np.linalg.lstsq(_affine(b[\'u\']), b[\'v\'], rcond=None)[0]\\n    z = _affine(x) @ operator\\n    d = _polynomial(x, 3)\\n    trajectory = train[\'traj\']\\n    centered = trajectory - trajectory.mean(axis=0)\\n    modes = np.linalg.svd(centered, full_matrices=False)[2][:2]\\n    scores = centered @ modes.T\\n    # Both time modes share the same quadratic coordinate. Estimate that\\n    # coordinate jointly rather than fitting independent interactions.\\n    q = d[:, 5:15]; a = d[:, :5]\\n    independent = q[:n] - a[:n] @ np.linalg.lstsq(a[:n], q[:n], rcond=None)[0]\\n    weights = np.linalg.lstsq(independent, scores, rcond=None)[0]\\n    left, _, _ = np.linalg.svd(weights, full_matrices=False)\\n    f = np.column_stack((a, q @ left[:, 0]))\\n    scores = f @ np.linalg.lstsq(f[:n], scores, rcond=None)[0]\\n    signal = _index(d, train[\'y\'], n, mode)\\n    extra = z[:, 0] if chart == 0 else z[:, 1] if chart == 1 else z[:, 3]\\n    target = np.column_stack((signal, scores, extra))\\n    target = (target - target[:n].mean(axis=0)) / np.maximum(target[:n].std(axis=0), 1e-8)\\n    calibrated = np.linalg.solve(operator[1:].T, (target - operator[0]).T).T\\n    return {\'x_cal_tr\': calibrated[:n], \'x_cal_te\': calibrated[n:]}\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    return _coordinates(inputs, 2, 0)\\n", "node_01": "import numpy as np\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    calibrated = inputs[\'node_00\']\\n    panel = study[\'panel_b\']\\n    operator = np.linalg.lstsq(_affine(panel[\'u\']), panel[\'v\'], rcond=None)[0]\\n    train = study[\'train\']\\n    return {\'z_tr\': _affine(calibrated[\'x_cal_tr\']) @ operator,\\n            \'z_te\': _affine(calibrated[\'x_cal_te\']) @ operator,\\n            \'y\': train[\'y\'], \'traj\': train[\'traj\']}\\n", "node_02": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _fit(x, y):\\n    return np.linalg.lstsq(x, y, rcond=None)[0]\\n\\n\\ndef _readout_design(z):\\n    interaction = z[:, 0] * z[:, 2] + z[:, 1] * z[:, 3]\\n    return np.column_stack((_affine(z), interaction))\\n\\n\\ndef _readout(p, design):\\n    q = design @ p[:-3]\\n    offset, amplitude, frequency = p[-3:]\\n    return q + amplitude * np.tanh(frequency * q + offset)\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    features = inputs[\'node_01\']\\n    z, z_test = features[\'z_tr\'], features[\'z_te\']\\n    design = _readout_design(z)\\n    y = features[\'y\']\\n    linear_start = _fit(design, y)\\n    lower = np.r_[np.full(len(linear_start) + 2, -np.inf), 0.12]\\n    upper = np.r_[np.full(len(linear_start) + 2, np.inf), 4.0]\\n    candidates = [least_squares(lambda p: _readout(p, design) - y,\\n                                np.r_[linear_start, 0., amplitude, frequency],\\n                                bounds=(lower, upper), max_nfev=180)\\n                  for frequency in (0.65, 1.3, 1.7, 2.5)\\n                  for amplitude in (0.2, 0.45)]\\n    p = min(candidates, key=lambda fit: fit.cost).x\\n    y_hat = _readout(p, _readout_design(z_test))\\n\\n    traj = features[\'traj\']\\n    modes = np.linalg.svd(traj, full_matrices=False)[2][:2]\\n    state_pred = (_affine(z_test) @ _fit(_affine(z), traj @ modes.T)) @ modes\\n    last = state_pred[:, -1]\\n    diagnostics = np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()])\\n    return {\'y_hat\': y_hat, \'state_pred\': state_pred, \'diagnostics_vec\': diagnostics}\\n"}'

def _canon_digest(obj) -> str:
    h = hashlib.sha256()
    def _feed(v):
        if v is None: h.update(b'N;')
        elif isinstance(v, bool): h.update(b'B1;' if v else b'B0;')
        elif isinstance(v, (int, np.integer)): h.update(f'I:{int(v)};'.encode())
        elif isinstance(v, (float, np.floating)):
            fv = float(v)
            if math.isnan(fv): h.update(b'F:NaN;')
            elif math.isinf(fv): h.update(b'F:Inf;')
            else: h.update(b'F:' + np.asarray([0.0 if fv == 0.0 else fv], dtype='<f8').tobytes() + b';')
        elif isinstance(v, str): h.update(f'S:{len(v)}:{v};'.encode())
        elif isinstance(v, np.ndarray):
            h.update(f'A:{v.dtype.kind}:{v.shape}:'.encode())
            if v.dtype.kind == 'f':
                a = np.asarray(v, dtype=np.float64)
                nm = np.isnan(a)
                cl = np.where(nm, 0.0, a)
                cl = np.where(cl == 0.0, 0.0, cl).astype('<f8')
                h.update(cl.tobytes('C'))
            else:
                h.update(np.asarray(v, dtype='<i8').tobytes('C'))
            h.update(b';')
        elif isinstance(v, dict):
            h.update(f'D:{len(v)}:'.encode())
            for k in sorted(v.keys(), key=str): _feed(str(k)); _feed(v[k])
            h.update(b';')
        elif isinstance(v, (list, tuple)):
            h.update(f'L:{len(v)}:'.encode())
            for it in v: _feed(it)
            h.update(b';')
        else:
            h.update(repr(v).encode())
    _feed(obj)
    return h.hexdigest()

def _load_node_fn(nid: str, src: str):
    ns = {'__name__': f'_hegel_node_{nid}'}
    exec(compile(src, f'<dag_node_{nid}>', 'exec'), ns)
    fn = ns.get('run_node') or ns.get('solve_batch')
    if not callable(fn):
        raise RuntimeError(f'DAG node {nid} missing callable run_node/solve_batch')
    return fn

_COMPILED_FNS = {nid: _load_node_fn(nid, src) for nid, src in json.loads(_NODE_SOURCES).items()}
_SPEC_OBJ = json.loads(_DAG_SPEC)

def solve_batch(study: dict) -> dict:
    outputs = {'__input__': deepcopy(study)}
    exec_counts = {nid: 0 for nid in _SPEC_OBJ['topo_order']}
    non_identity_checks = {}
    for nid in _SPEC_OBJ['topo_order']:
        srcs = _SPEC_OBJ['incoming'][nid]
        if len(_SPEC_OBJ['topo_order']) == 1 and srcs == ['__input__']:
            node_in = deepcopy(outputs['__input__'])
        else:
            node_in = {s: deepcopy(outputs[s]) for s in srcs}
        in_dig = _canon_digest(node_in)
        out_v = _COMPILED_FNS[nid](node_in)
        exec_counts[nid] += 1
        if exec_counts[nid] != 1:
            raise RuntimeError(f'DAG node {nid} executed {exec_counts[nid]} times != 1')
        if not isinstance(out_v, dict) or not out_v:
            raise RuntimeError(f'DAG node {nid} must return a non-empty dict')
        out_dig = _canon_digest(out_v)
        if nid != _SPEC_OBJ['sink_node_id']:
            # Anti-identity-shell check: intermediate node cannot merely echo its raw predecessor dict
            echoed = (out_dig == in_dig) or any(out_dig == _canon_digest(outputs[s]) for s in srcs)
            if echoed:
                raise RuntimeError(f'DAG node {nid} failed anti-identity-shell check')
            non_identity_checks[nid] = True
        outputs[nid] = out_v
    trace_fp = os.environ.get('HEGEL_M101_DAG_TRACE_PATH')
    if trace_fp:
        trace_obj = {
            'dag_id': _SPEC_OBJ['dag_id'],
            'active_node_count': len(_SPEC_OBJ['topo_order']),
            'depth': _SPEC_OBJ['depth'],
            'max_fanout': _SPEC_OBJ['max_fanout'],
            'shared_fanout_nodes': _SPEC_OBJ['shared_fanout_nodes'],
            'topo_order': _SPEC_OBJ['topo_order'],
            'exec_counts': exec_counts,
            'all_exec_count_eq_1': all(v == 1 for v in exec_counts.values()),
            'non_identity_checks': non_identity_checks,
            'node_code_shas': _SPEC_OBJ['node_code_shas'],
        }
        open(trace_fp, 'w', encoding='utf-8').write(json.dumps(trace_obj, sort_keys=True))
    return outputs[_SPEC_OBJ['sink_node_id']]
