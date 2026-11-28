"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_03", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "d0bc95b693b15d0f27e34f2802c8282b2a8dc754ce5b3b2aed20c257b6e4a8b7", "node_01": "0a96dc4ec6cb3cbd44cdd079b0ae1fdfb5837a2bcf3d1318dd74aae3aa8937c5"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _parameters(panel):\\n    u, v = panel[\'u\'], panel[\'v\']\\n    successor = np.roll(u, -1, axis=1)\\n    design = np.stack((np.ones_like(u), u, successor, u * successor, u**3), axis=2).transpose(1, 0, 2)\\n    weights = np.ones(len(u))\\n    for _ in range(30):\\n        coef = np.array([np.linalg.lstsq(d * weights[:, None], col * weights, rcond=None)[0] for d, col in zip(design, v.T)])\\n        residual = np.linalg.norm(v.T - np.einsum(\'cnp,cp->cn\', design, coef), axis=0)\\n        scale = max(1e-12, 2 * np.median(residual))\\n        weights = np.sqrt(np.minimum(1, scale / np.maximum(residual, 1e-12)))\\n    good = np.argsort(residual)[:len(u)-3]\\n    return np.array([np.linalg.lstsq(d[good], col[good], rcond=None)[0] for d, col in zip(design, v.T)]).T\\n\\n\\ndef _forward(u, coef):\\n    offset, linear, cross, pair, cubic = coef\\n    successor = np.roll(u, -1, axis=1)\\n    return offset + linear*u + cross*successor + pair*u*successor + cubic*u**3\\n\\n\\ndef _invert(v, coef):\\n    offset, linear, cross, pair, cubic = coef\\n    i = np.arange(v.shape[1]); j = (i + 1) % len(i)\\n    affine = np.diag(linear); affine[i, j] = cross\\n    u = np.linalg.solve(affine, (v-offset).T).T\\n    jac = np.zeros((len(v), len(i), len(i)))\\n    for _ in range(30):\\n        r = _forward(u, coef) - v\\n        if np.max(np.abs(r)) < 1e-11:\\n            break\\n        jac[:, i, i] = linear + pair*u[:, j] + 3*cubic*u*u\\n        jac[:, i, j] = cross + pair*u\\n        step = np.linalg.solve(jac, r[..., None])[..., 0]\\n        alpha = np.ones((len(u), 1))\\n        trial = u - step\\n        for _ in range(16):\\n            worse = np.linalg.norm(_forward(trial, coef) - v, axis=1) > np.linalg.norm(r, axis=1)\\n            if not worse.any():\\n                break\\n            alpha[worse] *= .5\\n            trial = u - alpha*step\\n        u = trial\\n    return u\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    train, test = study[\'train\'], study[\'test\']\\n    panel_a, panel_b = study[\'panel_a\'], study[\'panel_b\']\\n    a_coef = _parameters(panel_a)\\n    b_coef = np.linalg.lstsq(_affine(panel_b[\'u\']), panel_b[\'v\'], rcond=None)[0]\\n    # Undo the Panel A measurement map, then express the recovered points in Panel B coordinates.\\n    z = _affine(_invert(train[\'x\'], a_coef)) @ b_coef\\n    z_test = _affine(_invert(test[\'x\'], a_coef)) @ b_coef\\n    return {\'z\': z, \'z_test\': z_test, \'y\': train[\'y\'], \'traj\': train[\'traj\']}\\n", "node_01": "from itertools import combinations_with_replacement\\n\\nimport numpy as np\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _fit(x, y):\\n    return np.linalg.lstsq(x, y, rcond=None)[0]\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    data = inputs[\'node_00\']\\n    z, z_test = data[\'z\'], data[\'z_test\']\\n    a, a_test = _affine(z), _affine(z_test)\\n\\n    i, j = np.array(list(combinations_with_replacement(range(z.shape[1]), 2))).T\\n    products = z[:, i] * z[:, j]\\n    quadratic = _fit(np.column_stack((a, products)), data[\'y\'])[a.shape[1]:]\\n    keep = np.argsort(np.abs(quadratic))[-2:]\\n    pooled = products[:, keep].sum(axis=1)\\n    pooled_test = (z_test[:, i[keep]] * z_test[:, j[keep]]).sum(axis=1)\\n    y_hat = np.column_stack((a_test, pooled_test)) @ _fit(np.column_stack((a, pooled)), data[\'y\'])\\n\\n    traj = data[\'traj\']\\n    modes = np.linalg.svd(traj, full_matrices=False)[2][:2]\\n    state_pred = (a_test @ _fit(a, traj @ modes.T)) @ modes\\n    last = state_pred[:, -1]\\n    diagnostics = np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()])\\n    return {\'y_hat\': y_hat, \'state_pred\': state_pred, \'diagnostics_vec\': diagnostics}\\n"}'

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
