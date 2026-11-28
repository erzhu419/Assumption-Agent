"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_06", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "04680b8bf6fcb1e4701e8f8f69cb894cd5cfae35f821fea92fe509bcce9d4ca7", "node_01": "2de3e9e721e84af9274913413a321be71673216419072945dc03591ca709556c"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\n\\n\\ndef _calibrate(panel):\\n    u, v = panel[\'u\'], panel[\'v\']\\n    nxt = np.roll(u, -1, axis=1)\\n    basis = [np.column_stack((np.ones(len(u)), x, z, x*z, x**3))\\n             for x, z in zip(u.T, nxt.T)]\\n    ids = np.arange(len(u))\\n    for _ in range(8):\\n        coef = np.column_stack([np.linalg.lstsq(f[ids], v[ids, j], rcond=None)[0]\\n                                for j, f in enumerate(basis)])\\n        residual = np.column_stack([f @ coef[:, j] for j, f in enumerate(basis)]) - v\\n        ids = np.argsort(np.linalg.norm(residual, axis=1))[:len(u)-3]\\n    return coef\\n\\n\\ndef _inverse(x, c):\\n    from scipy.optimize import least_squares\\n\\n    def residual(z, row):\\n        w = np.roll(z, -1)\\n        return c[0] + c[1]*z + c[2]*w + c[3]*z*w + c[4]*z**3 - row\\n\\n    return np.array([least_squares(residual, (row-c[0])/c[1], args=(row,),\\n                                   gtol=1e-11).x for row in x])\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    train, test = study[\'train\'], study[\'test\']\\n    z = _inverse(np.vstack((train[\'x\'], test[\'x\'])), _calibrate(study[\'panel_a\']))\\n    return {\'z\': z, \'n_train\': len(train[\'x\']), \'y\': train[\'y\'],\\n            \'traj\': train[\'traj\'], \'eval_times\': study[\'eval_times\']}\\n", "node_01": "import numpy as np\\nfrom scipy.linalg import solve\\n\\n\\n_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0))\\n_CUBIC = sorted({tuple(sorted((i, j, k)))\\n                 for i, j in _EDGES for k in range(4)})\\n\\n\\ndef _features(z):\\n    pairs = np.column_stack([z[:, i] * z[:, j] for i, j in _EDGES])\\n    cubic = np.column_stack([z[:, i] * z[:, j] * z[:, k]\\n                             for i, j, k in _CUBIC])\\n    squares = z**2\\n    modulated_pairs = (pairs[:, :, None] * squares[:, None, :]).reshape(len(z), 16)\\n    return np.column_stack((np.ones(len(z)), z, squares, pairs, cubic,\\n                            modulated_pairs, squares**2))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    data = inputs[\'node_00\']\\n    n = data[\'n_train\']\\n    features = _features(data[\'z\'])\\n    scale = np.sqrt(np.mean(features[:n]**2, axis=0))\\n    train = features[:n] / scale\\n    targets = np.column_stack((data[\'y\'], data[\'traj\']))\\n    # Fit all horizons on the same normalized basis with mild shrinkage.\\n    gram = train.T @ train + (1e-4 * n) * np.eye(train.shape[1])\\n    coefficients = solve(gram, train.T @ targets, assume_a=\'pos\')\\n    prediction = (features[n:] / scale) @ coefficients\\n    response, trajectory = prediction[:, 0], prediction[:, 1:]\\n    final_state = trajectory[:, -1]\\n    return {\'y_hat\': response, \'state_pred\': trajectory,\\n            \'diagnostics_vec\': np.array([response.mean(), response.std(),\\n                                         final_state.mean(), final_state.std()])}\\n"}'

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
