"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_05", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "aaf75fc06dc271b90af448d3ac4023d892ad428f7f8d658060f7283f3ff1a604", "node_01": "fc36b2716bdc666ef23126280a50c381f060842aeac195f197ff64cbed9c2d7f"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\n\\n\\ndef _calibration(panel):\\n    u, v = panel[\'u\'], panel[\'v\']\\n    rng = np.random.default_rng(452)\\n    best = (np.inf, None, None)\\n    for _ in range(100):\\n        i, j = rng.choice(len(u), 2, replace=False)\\n        d = u[j] - u[i]\\n        if np.min(np.abs(d)) < 0.06:\\n            continue\\n        gain = (v[j] - v[i]) / d\\n        offset = v[i] - gain * u[i]\\n        err = np.linalg.norm(u * gain + offset - v, axis=1)\\n        score = np.median(err)\\n        if score < best[0]:\\n            best = (score, gain, offset)\\n    _, gain, offset = best\\n    err = np.linalg.norm(u * gain + offset - v, axis=1)\\n    keep = err <= max(0.001, 5 * np.median(err))\\n    uc, vc = u[keep], v[keep]\\n    a = uc - uc.mean(axis=0)\\n    gain = np.sum(a * (vc - vc.mean(axis=0)), axis=0) / np.sum(a * a, axis=0)\\n    return gain, vc.mean(axis=0) - uc.mean(axis=0) * gain\\n\\n\\ndef _affine_features(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _coordinates(study, x):\\n    gain, offset = _calibration(study[\'panel_a\'])\\n    panel = study[\'panel_b\']\\n    operator = np.linalg.lstsq(_affine_features(panel[\'u\']), panel[\'v\'], rcond=None)[0]\\n    return _affine_features((x - offset) / gain) @ operator\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    train = study[\'train\']\\n    z = _coordinates(study, np.concatenate((train[\'x\'], study[\'test\'][\'x\'])))\\n    n = len(train[\'x\'])\\n    return {\'z_tr\': z[:n], \'z_te\': z[n:], \'train_y\': train[\'y\'],\\n            \'train_traj\': train[\'traj\'], \'eval_times\': study[\'eval_times\']}\\n", "node_01": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _saturate(x):\\n    s = x * x\\n    return (2.5 * x) * (1.0 + 3.66 * s) / (1.0 + 6.23 * s + 2.01 * s * s)\\n\\n\\ndef _scalar_design(x):\\n    q = _saturate(x)\\n    s = x * x\\n    cross = [q[:, i] * q[:, j] for i in range(4) for j in range(i, 4)]\\n    return q, np.column_stack((np.ones(len(x)), q, *cross, x + 0.542 * s / (1.0 + 0.164 * s)))\\n\\n\\ndef _scalar_predict(x, y, xt):\\n    q, base = _scalar_design(x)\\n    qt, baset = _scalar_design(xt)\\n    orth = np.linalg.qr(base, mode=\'reduced\')[0]\\n    target = y - orth @ (orth.T @ y)\\n\\n    def extra(z, p):\\n        return np.tanh(z @ p[:4] + p[4])\\n\\n    def residual(p):\\n        h = extra(q, p)\\n        h -= orth @ (orth.T @ h)\\n        amplitude = (h @ target) / (h @ h + 1e-9)\\n        return np.r_[target - amplitude * h, 3e-5 * p, 3e-5 * amplitude]\\n\\n    rng = np.random.default_rng(45)\\n    starts = [np.full(5, 0.1)] + [rng.normal(0, s, 5) for s in (0.4, 0.8, 1.5) for _ in range(3)]\\n    best = min((least_squares(residual, p, bounds=(-6, 6), max_nfev=140) for p in starts), key=lambda fit: fit.cost)\\n    a = np.column_stack((base, extra(q, best.x)))\\n    at = np.column_stack((baset, extra(qt, best.x)))\\n    return at @ np.linalg.lstsq(a, y, rcond=None)[0]\\n\\n\\ndef _trajectory_design(x):\\n    return np.column_stack((np.ones(len(x)), x, np.arctan(2.4 * x)))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    data = inputs[\'node_00\']\\n    x, xt = data[\'z_tr\'], data[\'z_te\']\\n    y_hat = _scalar_predict(x, data[\'train_y\'], xt)\\n    coef = np.linalg.lstsq(_trajectory_design(x), data[\'train_traj\'], rcond=None)[0]\\n    state_pred = _trajectory_design(xt) @ coef\\n    return {\'y_hat\': y_hat, \'state_pred\': state_pred,\\n            \'diagnostics_vec\': np.array([y_hat.mean(), y_hat.std(), state_pred[:, -1].mean(), state_pred[:, -1].std()])}\\n"}'

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
