"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_09", "depth": 1, "fanout_by_node": {"node_00": 0}, "incoming": {"node_00": ["__input__"]}, "max_fanout": 0, "node_code_shas": {"node_00": "0cbe08369afd27eb43c3366ab60f22a4c56a6a4f8e96d6ccb9199996fc6fcc42"}, "shared_fanout_nodes": [], "sink_node_id": "node_00", "topo_order": ["node_00"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Recover latent coordinates and fit scalar and trajectory readouts.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.optimize import least_squares, root\\nfrom scipy.special import erf\\n\\n\\ndef _ls(x, y):\\n    return np.linalg.lstsq(x, y, rcond=None)[0]\\n\\n\\ndef _design(*blocks):\\n    return np.column_stack((np.ones(len(blocks[0])), *blocks))\\n\\n\\ndef _calibration(x):\\n    i, j = np.triu_indices(4)\\n    return _design(x, x[:, i] * x[:, j], x**3)\\n\\n\\ndef _latent(panel, x):\\n    z, v = _calibration(panel[\'u\']), panel[\'v\']\\n    keep = np.arange(len(z))\\n    for _ in range(8):\\n        w = _ls(z[keep], v[keep])\\n        keep = np.argsort(np.linalg.norm(z @ w - v, axis=1))[:45]\\n    w = _ls(z[keep], v[keep])\\n    return np.array([root(lambda p: _calibration(p[None])[0] @ w - row, row).x\\n                     for row in x])\\n\\n\\ndef _bridge(panel, x, kind):\\n    u, v = panel[\'u\'], panel[\'v\']\\n    def design(z, p):\\n        if kind == \'gaussian\':\\n            zz = z + p[2:]\\n            return _design(zz * (1 - p[1] * np.exp(-(p[0] * zz)**2)))\\n        return _design(z - p[1] * erf(p[0] * z), z*z)\\n\\n    def residual(p):\\n        z = design(u, p)\\n        return (z @ _ls(z, v) - v).ravel()\\n\\n    init = [1., 1.08, 0., 0., 0., 0.] if kind == \'gaussian\' else [1.45, .65]\\n    p = least_squares(residual, init).x\\n    return design(x, p) @ _ls(design(u, p), v)\\n\\n\\ndef _pairs(x):\\n    return _design(x[:, 0]*x[:, 1], x[:, 0]*x[:, 3],\\n                   x[:, 1]*x[:, 2], x[:, 2]*x[:, 3])\\n\\n\\ndef _features(study, x):\\n    a = _latent(study[\'panel_a\'], x)\\n    pc = study[\'panel_c\']\\n    w = _ls(_pairs(pc[\'u\']), pc[\'v1\'])\\n    for kind in (\'gaussian\', \'erf\'):\\n        c = _pairs(_bridge(study[\'panel_b\'], a, kind)) @ w\\n        yield np.column_stack((a, c, (a[:, :, None] * c[:, None, :]).reshape(len(a), 16)))\\n\\n\\ndef _readout(x, y, n, ridge):\\n    mean = x[:n].mean(axis=0)\\n    scale = np.maximum(x[:n].std(axis=0), 1e-5)\\n    z = (x - mean) / scale\\n    u, s, vt = np.linalg.svd(z[:n], full_matrices=False)\\n    center = y.mean(axis=0)\\n    return center + ((z[n:] @ vt.T) * (s / (s*s + ridge))) @ (u.T @ (y - center))\\n\\n\\ndef solve_batch(study: dict) -> dict:\\n    train = study[\'train\']\\n    n = len(train[\'x\'])\\n    targets = np.column_stack((train[\'y\'], train[\'traj\']))\\n    x = np.vstack((train[\'x\'], study[\'test\'][\'x\']))\\n    # The scalar readout is more sensitive to high-leverage interactions.\\n    features = list(_features(study, x))\\n    scalar = np.mean([_readout(f, targets[:, :1], n, .03) for f in features], axis=0)\\n    trajectory = np.mean([_readout(f, targets[:, 1:], n, .003) for f in features], axis=0)\\n    pred = np.column_stack((scalar, trajectory))\\n    y_hat, state_pred = pred[:, 0], pred[:, 1:]\\n    last = state_pred[:, -1]\\n    return {\'y_hat\': y_hat, \'state_pred\': state_pred,\\n            \'diagnostics_vec\': np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()])}\\n"}'

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
