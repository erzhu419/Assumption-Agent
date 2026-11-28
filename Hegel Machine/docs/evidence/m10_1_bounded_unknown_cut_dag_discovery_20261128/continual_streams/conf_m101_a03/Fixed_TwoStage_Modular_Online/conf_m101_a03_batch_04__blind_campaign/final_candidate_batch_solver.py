"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_05", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "e9f0e60a8b46e1d14d8e882734aaccb15759e31d21e65958365c266733ffa4cc", "node_01": "55286fc833ab6194ff38d42c4cc2d16ae155e7acee7f1277af1a448cc8613427"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Recover latent coordinates from Panel A, then apply the B and C v2 calibrations.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.optimize import least_squares, minimize_scalar\\n\\n\\ndef _panel_a_basis(u):\\n    neighbor = np.roll(u, -1, axis=1)\\n    return np.stack((np.ones_like(u), u, neighbor, u * neighbor, u**3), axis=-1)\\n\\n\\ndef _panel_a_fit(panel):\\n    design = _panel_a_basis(panel[\'u\'])\\n    v = panel[\'v\']\\n    w = np.ones(len(v))\\n    for _ in range(35):\\n        coef = np.stack([np.linalg.lstsq(design[:, j] * w[:, None], v[:, j] * w,\\n                                         rcond=None)[0] for j in range(v.shape[1])])\\n        residual = np.linalg.norm(np.einsum(\'njk,jk->nj\', design, coef) - v, axis=1)\\n        w = 1 / np.hypot(1, residual / max(np.median(residual), 1e-8))\\n    return coef\\n\\n\\ndef _invert(x, coef):\\n    diag = np.arange(coef.shape[0])\\n    linear = np.diag(coef[:, 1])\\n    linear[diag, (diag + 1) % len(diag)] = coef[:, 2]\\n    init = np.linalg.solve(linear, (x - coef[:, 0]).T).T\\n\\n    def observation(u):\\n        return np.einsum(\'jk,jk->j\', _panel_a_basis(u[None])[0], coef)\\n\\n    return np.stack([least_squares(lambda u: observation(u) - row, start).x\\n                     for row, start in zip(x, init)])\\n\\n\\ndef _transfer(panel, key, z, basis, bounds):\\n    u, v = panel[\'u\'], panel[key]\\n\\n    def design(x, rate):\\n        return np.column_stack((np.ones(len(x)), basis(x, rate)))\\n\\n    def fit(rate):\\n        a = design(u, rate)\\n        q, r = np.linalg.qr(a, mode=\'reduced\')\\n        proj = q.T @ v\\n        residual = (q @ proj - v) / (1 - np.sum(q*q, axis=1))[:, None]\\n        return np.mean(residual**2), np.linalg.solve(r, proj)\\n\\n    optimum = minimize_scalar(lambda rate: fit(rate)[0], bounds=bounds, method=\'bounded\')\\n    return design(z, optimum.x) @ fit(optimum.x)[1]\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    train, test = study[\'train\'], study[\'test\']\\n    n = len(train[\'x\'])\\n    z = _invert(np.concatenate((train[\'x\'], test[\'x\'])), _panel_a_fit(study[\'panel_a\']))\\n    b = _transfer(study[\'panel_b\'], \'v\', z,\\n                  lambda u, rate: np.column_stack((u, np.tanh(rate*u))), (0.4, 3.0))\\n    c = _transfer(study[\'panel_c\'], \'v2\', z,\\n                  lambda u, rate: np.column_stack((u, u*u, np.tanh(rate*u*u))), (0.1, 4.0))\\n    return {\'b_train\': b[:n], \'b_test\': b[n:], \'c_train\': c[:n], \'c_test\': c[n:],\\n            \'y_train\': np.asarray(train[\'y\']), \'traj_train\': np.asarray(train[\'traj\'])}\\n", "node_01": "\\"\\"\\"Fit a quadratic scalar readout and a joint linear trajectory readout.\\"\\"\\"\\nimport numpy as np\\n\\n\\ndef _predict(train, test, target, degree, ridge):\\n    scale = np.maximum(train.std(axis=0), 0.1) * np.sqrt(train.shape[1])\\n    train, test = train / scale, test / scale\\n    gram = (1 + train @ train.T)**degree\\n    gram.flat[::len(train)+1] += ridge\\n    center = target.mean(axis=0)\\n    return center + (1 + test @ train.T)**degree @ np.linalg.solve(gram, target - center)\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    upstream = inputs[\'node_00\']\\n    b_train, b_test = upstream[\'b_train\'], upstream[\'b_test\']\\n    y = _predict(b_train, b_test, upstream[\'y_train\'], 2, 0.03)\\n    train = np.column_stack((b_train, upstream[\'c_train\']))\\n    test = np.column_stack((b_test, upstream[\'c_test\']))\\n    traj = _predict(train, test, upstream[\'traj_train\'], 1, 0.003)\\n    terminal = traj[:, -1]\\n    diagnostics = np.array([y.mean(), y.std(), terminal.mean(), terminal.std()])\\n    return {\'y_hat\': y, \'state_pred\': traj, \'diagnostics_vec\': diagnostics}\\n"}'

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
