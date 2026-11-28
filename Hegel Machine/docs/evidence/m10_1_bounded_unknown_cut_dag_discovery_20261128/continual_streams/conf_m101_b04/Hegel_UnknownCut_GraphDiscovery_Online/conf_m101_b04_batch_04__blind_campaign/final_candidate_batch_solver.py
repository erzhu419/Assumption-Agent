"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_03", "depth": 1, "fanout_by_node": {"node_00": 0}, "incoming": {"node_00": ["__input__"]}, "max_fanout": 0, "node_code_shas": {"node_00": "8cd6fb02c92beb8d76a369863f79bd2acbbe261e125b908e6ee7b7ff283e8f18"}, "shared_fanout_nodes": [], "sink_node_id": "node_00", "topo_order": ["node_00"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Invert the cyclic calibration and fit scalar and relaxation readouts.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _fit(design, target):\\n    return np.linalg.lstsq(design, target, rcond=None)[0]\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _calibration(panel):\\n    u, v = panel[\'u\'], panel[\'v\']\\n    rng = np.random.default_rng(123)\\n    rows = []\\n    for a, b, y in zip(u.T, np.roll(u, -1, axis=1).T, v.T):\\n        design = np.column_stack((np.ones(len(a)), a, b, a*b, a**3))\\n        best_loss = np.inf\\n        for start in range(48):\\n            keep = rng.choice(len(a), 6, replace=False) if start else np.arange(len(a))\\n            for _ in range(7):\\n                resid = np.abs(y - design @ _fit(design[keep], y[keep]))\\n                keep = np.argsort(resid)[:44]\\n            coef = _fit(design[keep], y[keep])\\n            resid = np.abs(y - design @ coef)\\n            loss = np.linalg.norm(resid[keep])\\n            if loss < best_loss:\\n                best_loss, best_resid = loss, resid\\n        keep = best_resid < 0.001\\n        rows.append(_fit(design[keep], y[keep]))\\n    return np.array(rows)\\n\\n\\ndef _decode(x, calibration):\\n    offset, gain, coupling, product, cubic = calibration.T\\n    z = (x - offset) / gain\\n    ix = np.arange(z.shape[1])\\n    jac = np.zeros((len(z), len(ix), len(ix)))\\n    for _ in range(20):\\n        next_z = np.roll(z, -1, axis=1)\\n        resid = offset + gain*z + coupling*next_z + product*z*next_z + cubic*z**3 - x\\n        if np.max(np.abs(resid)) < 1e-12:\\n            break\\n        jac[:, ix, ix] = gain + product*next_z + 3*cubic*z*z\\n        jac[:, ix, (ix + 1) % len(ix)] = coupling + product*z\\n        z -= np.linalg.solve(jac, resid[..., None])[..., 0]\\n    return z\\n\\n\\ndef _scalar_design(z):\\n    return np.column_stack((_affine(z), z[:, 0]*z[:, 2], z[:, 1]*z[:, 3]))\\n\\n\\ndef _trajectory(times, observed, z, zt):\\n    def basis(rates):\\n        return -np.expm1(-times[:, None]*rates)\\n\\n    def residual(rates):\\n        temporal = basis(rates)\\n        return (temporal @ _fit(temporal, observed.T) - observed.T).ravel()\\n\\n    rates = least_squares(residual, [1.8, 0.4], bounds=([0.7, 0.03], [5.0, 0.68])).x\\n    temporal = basis(rates)\\n    amplitude = _fit(temporal, observed.T).T\\n    return (_affine(zt) @ _fit(_affine(z), amplitude)) @ temporal.T\\n\\n\\ndef solve_batch(study: dict) -> dict:\\n    train, xt = study[\'train\'], study[\'test\'][\'x\']\\n    calibration = _calibration(study[\'panel_a\'])\\n    panel = study[\'panel_b\']\\n    operator = _fit(_affine(panel[\'u\']), panel[\'v\'])\\n    z, zt = [_affine(_decode(x, calibration)) @ operator for x in (train[\'x\'], xt)]\\n    y_hat = _scalar_design(zt) @ _fit(_scalar_design(z), train[\'y\'])\\n    state_pred = _trajectory(study[\'eval_times\'], train[\'traj\'], z, zt)\\n    terminal = state_pred[:, -1]\\n    return {\'y_hat\': y_hat, \'state_pred\': state_pred,\\n            \'diagnostics_vec\': np.array([y_hat.mean(), y_hat.std(), terminal.mean(), terminal.std()])}\\n"}'

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
