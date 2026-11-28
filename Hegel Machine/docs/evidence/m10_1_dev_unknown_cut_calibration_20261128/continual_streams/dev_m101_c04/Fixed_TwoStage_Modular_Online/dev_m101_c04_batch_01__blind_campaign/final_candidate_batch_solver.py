"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_02", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "2bea63b73cb37a2d54317d05f94728c9bfaa50d8b018425d13893fe8c7333af0", "node_01": "979cac7692cf6835f38b09ec689183029d8692b07212498893fc1e5d9352d6c5"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\n\\n\\ndef _ls(a, b):\\n    return np.linalg.lstsq(a, b, rcond=None)[0]\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _calibration(study):\\n    a = study[\'panel_a\']\\n    u, v = a[\'u\'], a[\'v\']\\n    i, j = np.triu_indices(len(u), 1)\\n    coeff = []\\n    for x, y in zip(u.T, v.T):\\n        dx = x[i] - x[j]\\n        good = dx != 0\\n        gain = np.median((y[i] - y[j])[good] / dx[good])\\n        offset = np.median(y - gain * x)\\n        err = np.abs(y - offset - gain * x)\\n        keep = err <= max(0.001, 5 * np.median(err))\\n        coeff.append(_ls(_affine(x[keep]), y[keep]))\\n    offset, gain = np.asarray(coeff).T\\n    b = study[\'panel_b\']\\n    mixing = _ls(_affine(b[\'u\']), b[\'v\'])\\n    return offset, gain, mixing\\n\\n\\ndef _coordinates(x, calibration):\\n    offset, gain, mixing = calibration\\n    return _affine((x - offset) / gain) @ mixing\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    calibration = _calibration(study)\\n    train = study[\'train\']\\n    z_tr = _coordinates(train[\'x\'], calibration)\\n    z_te = _coordinates(study[\'test\'][\'x\'], calibration)\\n    return {\'z_tr\': z_tr, \'z_te\': z_te, \'y\': train[\'y\'],\\n            \'traj\': train[\'traj\'], \'eval_times\': study[\'eval_times\']}\\n", "node_01": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _affine(z):\\n    return np.column_stack((np.ones(len(z)), z))\\n\\n\\ndef _scalar_basis(z, affine):\\n    return np.column_stack((affine, z[:, 0] * z[:, 2], z[:, 1] * z[:, 3]))\\n\\n\\ndef _scalar_predict(basis, params):\\n    score = basis @ params[:-2]\\n    return score + params[-2] * np.tanh(params[-1] * score)\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    data = inputs[\'node_00\']\\n    z_tr, z_te = data[\'z_tr\'], data[\'z_te\']\\n    tr, te = _affine(z_tr), _affine(z_te)\\n    state_pred = te @ np.linalg.lstsq(tr, data[\'traj\'], rcond=None)[0]\\n\\n    basis = _scalar_basis(z_tr, tr)\\n    y = data[\'y\']\\n    linear = np.linalg.lstsq(basis, y, rcond=None)[0]\\n    fit = least_squares(\\n        lambda p: _scalar_predict(basis, p) - y,\\n        np.r_[0.75 * linear, 0.45, 1.5],\\n        bounds=(np.r_[np.full(len(linear), -np.inf), 0., 0.1],\\n                np.r_[np.full(len(linear), np.inf), 2., 8.]),\\n    )\\n    y_hat = _scalar_predict(_scalar_basis(z_te, te), fit.x)\\n    endpoint = state_pred[:, -1]\\n    return {\\n        \'y_hat\': y_hat,\\n        \'state_pred\': state_pred,\\n        \'diagnostics_vec\': np.array([y_hat.mean(), y_hat.std(), endpoint.mean(), endpoint.std()]),\\n    }\\n"}'

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
