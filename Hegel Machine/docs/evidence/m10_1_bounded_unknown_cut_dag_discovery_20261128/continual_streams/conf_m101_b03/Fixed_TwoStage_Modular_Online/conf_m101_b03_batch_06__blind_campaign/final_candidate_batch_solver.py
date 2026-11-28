"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_08", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__", "node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "721a49fd55da52d7eb722ffd542cc4d5e26c9bebb05612b461f903a8fd226f46", "node_01": "9cc0c51cada7f39dda50d6fb23252ac8ab2ed673e2937623ff7dc4cd3fb2bafe"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\n\\n\\ndef _fit(design, values):\\n    return np.linalg.lstsq(design, values, rcond=None)[0]\\n\\n\\ndef _affine_calibration(panel):\\n    u, v = panel[\'u\'], panel[\'v\']\\n    design = np.column_stack((np.ones(len(u)), u))\\n    coef = _fit(design, v)\\n    # Fit in the forward direction so corrupted measurements are residuals,\\n    # not high-leverage predictors. Trim the three identified outlier rows.\\n    for _ in range(10):\\n        residual = np.linalg.norm(v - design @ coef, axis=1)\\n        weight = np.sqrt(np.minimum(1.0, 1.5 * max(np.median(residual), 1e-15)\\n                                    / np.maximum(residual, 1e-15)))\\n        coef = _fit(design * weight[:, None], v * weight[:, None])\\n    residual = np.linalg.norm(v - design @ coef, axis=1)\\n    keep = np.argsort(residual)[:-3]\\n    return _fit(design[keep], v[keep])\\n\\n\\ndef _pair_basis(u):\\n    a, b, c, d = u.T\\n    return np.column_stack((np.ones(len(u)), a*b, a*d, b*c, c*d))\\n\\n\\ndef _cosine_basis(u):\\n    return np.column_stack((np.ones(len(u)), np.cos(2.4 * u)))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    train, test = study[\'train\'][\'x\'], study[\'test\'][\'x\']\\n    x = np.concatenate((train, test), axis=0)\\n    affine = _affine_calibration(study[\'panel_a\'])\\n    u = np.linalg.solve(affine[1:].T, (x - affine[0]).T).T\\n\\n    panel = study[\'panel_c\']\\n    v1 = _pair_basis(u) @ _fit(_pair_basis(panel[\'u\']), panel[\'v1\'])\\n    v2 = _cosine_basis(u) @ _fit(_cosine_basis(panel[\'u\']), panel[\'v2\'])\\n    interaction = (v1[:, :, None] * v2[:, None, :]).reshape(len(u), -1)\\n    design = np.column_stack((v1, v2, interaction))\\n    return {\'design\': design, \'n_train\': np.asarray(len(train))}\\n", "node_01": "import numpy as np\\n\\n\\ndef _predict(design, values, train, test, penalty):\\n    a = design[train]\\n    center = a.mean(axis=0)\\n    scale = a.std(axis=0)\\n    f = (a-center)/scale\\n    offset = values[train].mean(axis=0)\\n    gram = f.T @ f + penalty*np.eye(f.shape[1])\\n    w = np.linalg.solve(gram, f.T @ (values[train]-offset))\\n    return (design[test]-center)/scale @ w + offset\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    data = inputs[\'__input__\'][\'train\']\\n    features = inputs[\'node_00\']\\n    design, n = features[\'design\'], features[\'n_train\']\\n    y = _predict(design, data[\'y\'], slice(n), slice(n, None), .003)\\n    state = _predict(design, data[\'traj\'], slice(n), slice(n, None), .0001)\\n    terminal = state[:, -1]\\n    return {\'y_hat\': y, \'state_pred\': state,\\n            \'diagnostics_vec\': np.array([y.mean(), y.std(), terminal.mean(), terminal.std()])}\\n"}'

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
