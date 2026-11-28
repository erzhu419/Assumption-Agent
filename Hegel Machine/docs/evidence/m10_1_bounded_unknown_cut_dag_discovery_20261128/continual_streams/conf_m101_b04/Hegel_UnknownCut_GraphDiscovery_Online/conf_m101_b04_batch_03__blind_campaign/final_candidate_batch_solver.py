"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_02", "depth": 1, "fanout_by_node": {"node_00": 0}, "incoming": {"node_00": ["__input__"]}, "max_fanout": 0, "node_code_shas": {"node_00": "6dd711d676ef703874a4e67ecbde2bf818296b7025f1736c72896c841472027e"}, "shared_fanout_nodes": [], "sink_node_id": "node_00", "topo_order": ["node_00"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Predict scalar and temporal readouts from the calibrated panel-B operator.\\"\\"\\"\\nfrom itertools import combinations\\n\\nimport numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _calibration(panel):\\n    # Each channel is an independent gain and offset; panel outliers are sparse.\\n    coefs = []\\n    for u, v in zip(panel[\'u\'].T, panel[\'v\'].T):\\n        a = _affine(u[:, None])\\n        keep = np.arange(len(u))\\n        for _ in range(6):\\n            c = np.linalg.lstsq(a[keep], v[keep], rcond=None)[0]\\n            keep = np.argsort(np.abs(v - a @ c))[:len(u) * 5 // 6]\\n        coefs.append(np.linalg.lstsq(a[keep], v[keep], rcond=None)[0])\\n    return np.asarray(coefs).T\\n\\n\\ndef _operator(panel):\\n    u, v = panel[\'u\'], panel[\'v\']\\n    candidates = []\\n    for fun, start, limits in ((np.sin, 1.6, (.25, 3.5)),\\n                               (np.tanh, 2.5, (.25, 6.0))):\\n        def design(x, rate):\\n            return np.column_stack((_affine(x), fun(x * rate)))\\n\\n        def residual(rate):\\n            a = design(u, rate)\\n            return (a @ np.linalg.lstsq(a, v, rcond=None)[0] - v).ravel()\\n\\n        rate = least_squares(residual, [start], bounds=limits).x[0]\\n        a = design(u, rate)\\n        coef = np.linalg.lstsq(a, v, rcond=None)[0]\\n        q = np.linalg.qr(a)[0]\\n        loo = (v - a @ coef) / (1.0 - np.sum(q*q, axis=1))[:, None]\\n        candidates.append((np.mean(loo**2), fun, rate, coef))\\n    _, fun, rate, coef = min(candidates, key=lambda item: item[0])\\n    return lambda x: np.column_stack((_affine(x), fun(x * rate))) @ coef\\n\\n\\ndef _scalar_readout(z, zt, y):\\n    # Select two cross-channel products using only the supplied fit rows.\\n    pairs = list(combinations(range(z.shape[1]), 2))\\n    products = np.column_stack([z[:, i] * z[:, j] for i, j in pairs])\\n    linear = _affine(z)\\n\\n    def loss(cols):\\n        design = np.column_stack((linear, products[:, cols]))\\n        residual = design @ _ridge(design, y, .001) - y\\n        return np.mean(residual**2)\\n\\n    cols = min(combinations(range(len(pairs)), 2), key=loss)\\n    pt = np.column_stack([zt[:, pairs[k][0]] * zt[:, pairs[k][1]] for k in cols])\\n    p = products[:, cols]\\n    design = np.column_stack((linear, p, p**2))\\n    test_design = np.column_stack((_affine(zt), pt, pt**2))\\n    return test_design @ _ridge(design, y, .015)\\n\\n\\ndef _ridge(a, y, penalty):\\n    extra = np.diag(np.r_[0., np.full(a.shape[1] - 1, np.sqrt(penalty))])\\n    return np.linalg.lstsq(np.vstack((a, extra)),\\n                           np.r_[y, np.zeros((len(extra),) + y.shape[1:])],\\n                           rcond=None)[0]\\n\\n\\ndef solve_batch(study: dict) -> dict:\\n    train, xt = study[\'train\'], study[\'test\'][\'x\']\\n    offset, gain = _calibration(study[\'panel_a\'])\\n    panel_b = _operator(study[\'panel_b\'])\\n    z, zt = [panel_b((x - offset) / gain) for x in (train[\'x\'], xt)]\\n    y_hat = _scalar_readout(z, zt, train[\'y\'])\\n\\n    times, observed = study[\'eval_times\'], train[\'traj\'].T\\n\\n    def basis(rates):\\n        return -np.expm1(-times[:, None] * rates)\\n\\n    def residual(rates):\\n        b = basis(rates)\\n        return (b @ np.linalg.lstsq(b, observed, rcond=None)[0] - observed).ravel()\\n\\n    rates = least_squares(residual, [1.8, .4], bounds=([.7, .03], [5., .68])).x\\n    modes = np.linalg.qr(basis(rates))[0]\\n    state_pred = _affine(zt) @ _ridge(_affine(z), observed.T @ modes, .001) @ modes.T\\n    terminal = state_pred[:, -1]\\n    return {\'y_hat\': y_hat, \'state_pred\': state_pred,\\n            \'diagnostics_vec\': np.array([y_hat.mean(), y_hat.std(), terminal.mean(), terminal.std()])}\\n"}'

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
