"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_06", "depth": 3, "fanout_by_node": {"node_00": 2, "node_01": 1, "node_02": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["node_00"], "node_02": ["__input__", "node_00", "node_01"]}, "max_fanout": 2, "node_code_shas": {"node_00": "7c5d63e0ce11e9736d541a4a4e3b4b4460d39e846a8818f6ab55e2b77f2c7218", "node_01": "027b3ecfea2a98eb3240a00512d49b6275d89ce552abc97ffdff8b7e6f53311a", "node_02": "bcb9903e49d793d3f6db5193e5882cae3c2c6ca50180ed82e17389dec370ef6b"}, "shared_fanout_nodes": ["node_00"], "sink_node_id": "node_02", "topo_order": ["node_00", "node_01", "node_02"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    raw = np.vstack([study[\'train\'][\'x\'], study[\'test\'][\'x\']])\\n    panel = study[\'panel_a\']\\n    u, v = panel[\'u\'], panel[\'v\']\\n    fit = np.column_stack([np.ones(len(u)), u])\\n    keep = np.ones(len(u), dtype=bool)\\n    for _ in range(6):\\n        coef = np.linalg.lstsq(fit[keep], v[keep], rcond=None)[0]\\n        r = np.linalg.norm(fit @ coef - v, axis=1)\\n        keep = r < max(0.001, 4 * np.median(r))\\n    return {\'x\': raw @ coef[1:] + coef[0]}\\n", "node_01": "from itertools import combinations_with_replacement\\nimport numpy as np\\nfrom scipy.spatial.distance import cdist\\n\\n\\ndef _poly(x, d):\\n    return np.column_stack([np.prod(x[:, p], axis=1)\\n                            for p in combinations_with_replacement(range(x.shape[1]), d)]) if d else np.ones((len(x), 1))\\n\\n\\ndef _quadratic_gram(x):\\n    f = np.column_stack([_poly(x, j) for j in range(3)])\\n    return f @ f.T\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    x = inputs[\'node_00\'][\'x\']\\n    d = cdist(x, x, \'sqeuclidean\')\\n    r = np.sqrt(5*d)/2\\n    base = np.stack([_quadratic_gram(x), np.exp(-d/8), (1+r+r*r/3)*np.exp(-r)])\\n    harmonic = np.stack([_quadratic_gram(fn(c*x))\\n                         for fn in (np.tanh, np.sin) for c in (.6, 1.3, 2.)])\\n    wide = np.stack([np.exp(-d/24), np.exp(-d/3.5)])\\n    return {\'base\': base, \'harmonic\': np.concatenate([base, harmonic]),\\n            \'wide\': np.concatenate([base, wide])}\\n", "node_02": "from itertools import combinations_with_replacement\\n\\nimport numpy as np\\nfrom scipy.linalg import cho_factor, cho_solve\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    train = inputs[\'__input__\'][\'train\']\\n    x = inputs[\'node_00\'][\'x\']\\n    kernel = inputs[\'node_01\'][\'base\'][2]\\n    targets = np.column_stack((train[\'y\'], train[\'traj\']))\\n    n = len(targets)\\n    trend = np.column_stack((np.ones(len(x)), x, *(\\n        x[:, i] * x[:, j]\\n        for i, j in combinations_with_replacement(range(x.shape[1]), 2)\\n    )))\\n    factor = cho_factor(kernel[:n, :n] + 0.001 * np.eye(n))\\n    weighted_trend = cho_solve(factor, trend[:n])\\n    weighted_targets = cho_solve(factor, targets)\\n    coef = np.linalg.lstsq(trend[:n].T @ weighted_trend,\\n                           trend[:n].T @ weighted_targets, rcond=1e-9)[0]\\n    pred = trend[n:] @ coef + kernel[n:, :n] @ (weighted_targets - weighted_trend @ coef)\\n    y, traj = pred[:, 0], pred[:, 1:]\\n    return {\'y_hat\': y, \'state_pred\': traj,\\n            \'diagnostics_vec\': np.array([y.mean(), y.std(), traj[:, -1].mean(), traj[:, -1].std()])}\\n"}'

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
