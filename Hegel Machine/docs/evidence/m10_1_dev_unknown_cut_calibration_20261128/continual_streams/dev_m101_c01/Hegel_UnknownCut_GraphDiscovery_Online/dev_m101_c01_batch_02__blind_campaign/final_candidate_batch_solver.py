"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_01", "depth": 1, "fanout_by_node": {"node_00": 0}, "incoming": {"node_00": ["__input__"]}, "max_fanout": 0, "node_code_shas": {"node_00": "d1f498af429b49373133f5a0285edac37c0ca58afe4b451bd261bd50a47ba0f5"}, "shared_fanout_nodes": [], "sink_node_id": "node_00", "topo_order": ["node_00"]}'
_NODE_SOURCES = '{"node_00": "from itertools import combinations_with_replacement\\n\\nimport numpy as np\\nfrom scipy.linalg import cho_factor, cho_solve\\nfrom scipy.optimize import least_squares, minimize\\nfrom scipy.spatial.distance import cdist\\n\\n\\ndef _calibrate(panel, x):\\n    u, v = panel[\'u\'], panel[\'v\']\\n    fit = np.column_stack([np.ones(len(u)), u])\\n    keep = np.ones(len(u), dtype=bool)\\n    for _ in range(6):\\n        coef = np.linalg.lstsq(fit[keep], v[keep], rcond=None)[0]\\n        r = np.linalg.norm(fit @ coef - v, axis=1)\\n        keep = r < max(0.001, 4 * np.median(r))\\n    return x @ coef[1:] + coef[0]\\n\\n\\ndef _poly(x, d):\\n    return np.column_stack([np.prod(x[:, p], axis=1)\\n                            for p in combinations_with_replacement(range(x.shape[1]), d)]) if d else np.ones((len(x), 1))\\n\\n\\ndef _temporal(t, z):\\n    def basis(p):\\n        k, damp, omega = p\\n        return np.column_stack([-np.expm1(-k * t), np.exp(-damp*t)*np.sin(omega*t)])\\n\\n    def residual(p):\\n        f = basis(p)\\n        return (f @ np.linalg.lstsq(f, z.T, rcond=None)[0] - z.T).ravel()\\n\\n    p = least_squares(residual, [.5, .2, 1.1], bounds=([.01, 0, .5], [4, 2, 2])).x\\n    f = basis(p)\\n    q, _ = np.linalg.qr(f)\\n    return q\\n\\n\\ndef _quadratic_gram(x):\\n    f = np.column_stack([_poly(x, j) for j in range(3)])\\n    return f @ f.T\\n\\n\\ndef _blocks(x):\\n    d = cdist(x, x, \'sqeuclidean\')\\n    r = np.sqrt(5*d)/2\\n    base = np.stack([_quadratic_gram(x), np.exp(-d/8), (1+r+r*r/3)*np.exp(-r)])\\n    harmonic = np.stack([_quadratic_gram(fn(c*x))\\n                         for fn in (np.tanh, np.sin) for c in (.6, 1.3, 2.)])\\n    wide = np.stack([np.exp(-d/24), np.exp(-d/3.5)])\\n    return base, np.concatenate([base, harmonic]), np.concatenate([base, wide])\\n\\n\\ndef _regress(blocks, y):\\n    n, m = y.shape\\n    identity = np.eye(n)\\n    k = blocks[:, :n, :n].copy()\\n    l = blocks[:, n:, :n].copy()\\n    norm = np.diagonal(k, axis1=1, axis2=2).mean(axis=1)\\n    k = np.concatenate([k / norm[:, None, None], identity[None]])\\n    l = np.concatenate([l / norm[:, None, None], np.zeros_like(l[:1])])\\n    std = np.maximum(y.std(axis=0), 1e-12)\\n    y = y / std\\n\\n    def likelihood(p):\\n        w = np.exp(p)\\n        factor = cho_factor(np.einsum(\'k,kij->ij\', w, k), check_finite=False)\\n        alpha = cho_solve(factor, y, check_finite=False)\\n        inv = cho_solve(factor, identity, check_finite=False)\\n        value = .5*np.sum(alpha*y) + m*np.log(np.diag(factor[0])).sum()\\n        grad = .5*w*np.einsum(\'kij,ij->k\', k, m*inv-alpha@alpha.T)\\n        return value, grad\\n\\n    p = minimize(likelihood, np.r_[np.zeros(len(k)-1), -8.], jac=True,\\n                 method=\'L-BFGS-B\', bounds=[(-19, 8)]*(len(k)-1)+[(-18, -1)]).x\\n    w = np.exp(p)\\n    factor = cho_factor(np.einsum(\'k,kij->ij\', w, k), check_finite=False)\\n    return (np.einsum(\'k,kij->ij\', w, l) @ cho_solve(factor, y, check_finite=False))*std\\n\\n\\ndef solve_batch(study: dict) -> dict:\\n    train = study[\'train\']\\n    raw = np.vstack([train[\'x\'], study[\'test\'][\'x\']])\\n    x = _calibrate(study[\'panel_a\'], raw)\\n    base, harmonic, wide = _blocks(x)\\n    target = train[\'y\'][:, None]\\n    y = 0.5 * (_regress(base, target) + _regress(harmonic, target))[:, 0]\\n\\n    z = train[\'traj\']\\n    q = _temporal(study[\'eval_times\'], z)\\n    amplitude = z @ q\\n    state = (0.5 * (_regress(base, amplitude) + _regress(wide, amplitude))) @ q.T\\n    end = state[:, -1]\\n    return {\'y_hat\': y, \'state_pred\': state,\\n            \'diagnostics_vec\': np.array([y.mean(), y.std(), end.mean(), end.std()])}\\n"}'

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
