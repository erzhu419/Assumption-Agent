"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_02", "depth": 1, "fanout_by_node": {"node_00": 0}, "incoming": {"node_00": ["__input__"]}, "max_fanout": 0, "node_code_shas": {"node_00": "793e3f4380664516555084b9bfd907b3def1d7fb5dda42f4e3aafbc29bcddd9c"}, "shared_fanout_nodes": [], "sink_node_id": "node_00", "topo_order": ["node_00"]}'
_NODE_SOURCES = '{"node_00": "from itertools import combinations_with_replacement\\n\\nimport numpy as np\\nfrom scipy.linalg import solve\\nfrom scipy.optimize import least_squares\\nfrom scipy.spatial.distance import cdist\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _least_squares(x, y):\\n    return np.linalg.lstsq(x, y, rcond=None)[0]\\n\\n\\ndef _calibrate(panel, x):\\n    u, v = panel[\'u\'], panel[\'v\']\\n    keep = np.ones(len(u), dtype=bool)\\n    for _ in range(5):\\n        w = np.array([_least_squares(_affine(u[keep, j:j+1]), v[keep, j])\\n                      for j in range(u.shape[1])]).T\\n        residual = np.linalg.norm(w[0] + u * w[1] - v, axis=1)\\n        keep = residual < max(0.001, 4 * np.median(residual))\\n    return w[0] + x * w[1]\\n\\n\\ndef _even_operator(panel, x):\\n    u, v = panel[\'u\'], panel[\'v2\']\\n\\n    def design(q, k):\\n        return np.column_stack((_affine(q), np.sqrt(1 + k * q**2) - 1))\\n\\n    def residual(k):\\n        h = design(u, k)\\n        return (h @ _least_squares(h, v) - v).ravel()\\n\\n    k = least_squares(residual, [0.2], bounds=(0.001, 8)).x\\n    return design(x, k) @ _least_squares(design(u, k), v)\\n\\n\\ndef _kernel_inputs(study, x):\\n    panel = study[\'panel_b\']\\n    b = _least_squares(_affine(panel[\'u\']), panel[\'v\'])\\n    u = _affine(_calibrate(study[\'panel_a\'], x)) @ b\\n    d = _even_operator(study[\'panel_c\'], u)\\n    quadratic = np.column_stack([u[:, i] * u[:, j]\\n                                 for i, j in combinations_with_replacement(range(u.shape[1]), 2)])\\n    interactions = (u[:, :, None] * d[:, None, :]).reshape(len(u), -1)\\n    f = np.column_stack((_affine(u), quadratic, d, interactions))\\n    return f @ f.T, cdist(u, u, metric=\'sqeuclidean\')\\n\\n\\ndef _predict(k, train, test, observed, ridge=0.03):\\n    gram = k[np.ix_(train, train)] + ridge**2 * np.eye(len(train))\\n    return k[np.ix_(test, train)] @ solve(gram, observed, assume_a=\'pos\')\\n\\n\\ndef solve_batch(study: dict) -> dict:\\n    train = study[\'train\']\\n    x, q = train[\'x\'], study[\'test\'][\'x\']\\n    i, j = np.arange(len(x)), np.arange(len(x), len(x) + len(q))\\n    base, distance = _kernel_inputs(study, np.vstack((x, q)))\\n    short = base + 0.3 * np.exp(-0.5 * distance)\\n    long = base + np.exp(-0.125 * distance)\\n    y_hat = _predict(short, i, j, train[\'y\'])\\n    # Fit a shared temporal shape and regress its amplitude.\\n    trajectory = train[\'traj\']\\n    shape = np.linalg.svd(trajectory, full_matrices=False)[2][0]\\n    amplitude = trajectory @ shape\\n    predicted_amplitude = (_predict(short, i, j, amplitude) + _predict(long, i, j, amplitude)) / 2\\n    state_pred = predicted_amplitude[:, None] * shape\\n    final = state_pred[:, -1]\\n    diagnostics = np.array([y_hat.mean(), y_hat.std(), final.mean(), final.std()])\\n    return {\'y_hat\': y_hat, \'state_pred\': state_pred, \'diagnostics_vec\': diagnostics}\\n"}'

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
