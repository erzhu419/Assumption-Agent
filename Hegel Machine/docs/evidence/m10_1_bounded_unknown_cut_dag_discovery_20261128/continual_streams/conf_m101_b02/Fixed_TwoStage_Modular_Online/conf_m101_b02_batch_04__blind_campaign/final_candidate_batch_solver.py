"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_06", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "79addd46e9610f41756b8dc5b4c9309c6ce8333f9c2bff273ceb9bf187aba99c", "node_01": "1ab0e98631990d7b38e6759d6dfa5f256b227ad1d5262b63fa625ace5de4742e"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _fit(design, target):\\n    return np.linalg.lstsq(design, target, rcond=None)[0]\\n\\n\\ndef _align(panel, x):\\n    design, v = _affine(panel[\'u\']), panel[\'v\']\\n    coef = _fit(design, v)\\n    for _ in range(12):\\n        residual = np.linalg.norm(v - design @ coef, axis=1)\\n        scale = max(3 * np.median(residual), np.finfo(float).eps)\\n        weight = 1 / np.hypot(1, residual / scale)\\n        coef = _fit(design * weight[:, None], v * weight[:, None])\\n    return np.linalg.solve(coef[1:].T, (x - coef[0]).T).T\\n\\n\\ndef _cycle(x):\\n    return x * np.roll(x, -1, axis=1)\\n\\n\\ndef _b_basis(x, frequency):\\n    return np.column_stack((_affine(x), np.sin(frequency * x)))\\n\\n\\ndef _c2_basis(x):\\n    return np.column_stack((_affine(x), x*x, np.cos(2.6*x) - 1))\\n\\n\\ndef _panel_b(panel, x, unmix=False):\\n    u, v = panel[\'u\'], panel[\'v\']\\n    def loss(f):\\n        d = _b_basis(u, f)\\n        return np.linalg.norm(v - d @ _fit(d, v))\\n    grid = np.linspace(0.5, 7.0, 66)\\n    best = grid[np.argmin([loss(f) for f in grid])]\\n    left, right = best - .1, best + .1\\n    for _ in range(20):\\n        a, b = (2*left + right)/3, (left + 2*right)/3\\n        if loss(a) < loss(b):\\n            right = b\\n        else:\\n            left = a\\n    f = (left + right)/2\\n    coef = _fit(_b_basis(u, f), v)\\n    b = _b_basis(x, f) @ coef\\n    if unmix:\\n        h = np.linalg.solve(coef[1:5].T, (b - coef[0]).T).T\\n        return b, h\\n    return b\\n\\n\\ndef _features(s, x, unmix=False):\\n    z = _align(s[\'panel_a\'], x)\\n    b = _panel_b(s[\'panel_b\'], z, unmix)\\n    c = s[\'panel_c\']\\n    c1 = _affine(_cycle(z)) @ _fit(_affine(_cycle(c[\'u\'])), c[\'v1\'])\\n    c2 = _c2_basis(z) @ _fit(_c2_basis(c[\'u\']), c[\'v2\'])\\n    return z, b, c1, c2\\n\\n\\ndef _polynomial_features(blocks, n, degree=4, rate=.15, scale=3.):\\n    # Factor the centered polynomial kernel in the training-row span.\\n    design = []\\n    for x, weight in zip(blocks, (.25, .50, .25)):\\n        x = x - x[:n].mean(axis=0)\\n        energy = np.sum(x[:n] ** 2) / n\\n        design.append(x * np.sqrt(weight / max(energy, np.finfo(float).eps)))\\n    x = np.column_stack(design)\\n    k = (1 + rate * (x @ x[:n].T)) ** degree\\n    row_mean = k.mean(axis=1)\\n    train_mean = row_mean[:n]\\n    k = k - row_mean[:, None] - train_mean[None, :] + train_mean.mean()\\n    value, vector = np.linalg.eigh(k[:n])\\n    keep = value > value[-1] * 1e-12\\n    mapped = k @ (vector[:, keep] * np.sqrt(scale / value[keep]))\\n    return np.column_stack((np.ones(len(x)), mapped))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    s = inputs[\'__input__\']\\n    train = s[\'train\']\\n    x = np.vstack((train[\'x\'], s[\'test\'][\'x\']))\\n    z, (b, h), c1, c2 = _features(s, x, unmix=True)\\n    return {\'g\': _polynomial_features((z, b, h), len(train[\'x\'])),\\n            \'fd\': b, \'cd\': np.column_stack((c1, c2)),\\n            \'y\': train[\'y\'], \'traj\': train[\'traj\']}\\n", "node_01": "import numpy as np\\n\\n\\ndef _ridge(g, target, penalty):\\n    n = len(target)\\n    training = g[:n]\\n    normal = training.T @ training + penalty * np.eye(g.shape[1])\\n    normal[0, 0] -= penalty\\n    return g[n:] @ np.linalg.solve(normal, training.T @ target)\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    features = inputs[\'node_00\']\\n    g, fd, cd = features[\'g\'], features[\'fd\'], features[\'cd\']\\n    traj = features[\'traj\']\\n    basis = np.linalg.svd(traj, full_matrices=False)[2][:2]\\n    target = np.column_stack((features[\'y\'], traj @ basis.T))\\n    base = _ridge(g, target, .0003)\\n    prediction = .5 * base + (\\n        _ridge(np.column_stack((g, fd)), target, .003)\\n        + _ridge(np.column_stack((g, cd)), target, .0003)\\n        + _ridge(np.column_stack((g, fd, cd)), target, .001)\\n    ) / 6\\n    y_hat = prediction[:, 0]\\n    state_pred = prediction[:, 1:] @ basis\\n    terminal = state_pred[:, -1]\\n    return {\'y_hat\': y_hat, \'state_pred\': state_pred,\\n            \'diagnostics_vec\': np.array([y_hat.mean(), y_hat.std(),\\n                                         terminal.mean(), terminal.std()])}\\n"}'

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
