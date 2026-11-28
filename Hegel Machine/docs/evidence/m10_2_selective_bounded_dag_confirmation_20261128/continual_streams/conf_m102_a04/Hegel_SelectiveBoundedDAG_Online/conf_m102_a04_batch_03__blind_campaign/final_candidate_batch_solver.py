"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_03", "depth": 3, "fanout_by_node": {"node_00": 2, "node_01": 2, "node_02": 1, "node_03": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__"], "node_02": ["node_00", "node_01"], "node_03": ["node_00", "node_01", "node_02"]}, "max_fanout": 2, "node_code_shas": {"node_00": "80b388d985ee4f720b9f2ca3cca16e725cbf8d6c47e553d7015535f22c15000e", "node_01": "493aca178d528c16ec681d5e109ba4e4500ff58bbef4f1e9c17dd6aa7ee50743", "node_02": "f9c054d2acbdc03972c86d1f6d5990aa75bdcee20033ee5eabc835bc69e58d95", "node_03": "41c6e176cda90bdc28fbfd1ca8ecb7077c1634616b5de4b5fd44445745c9a0a5"}, "shared_fanout_nodes": ["node_00", "node_01"], "sink_node_id": "node_03", "topo_order": ["node_00", "node_01", "node_02", "node_03"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef lin(f, y):\\n    return np.linalg.lstsq(f, y, rcond=1e-9)[0]\\n\\n\\ndef one(z):\\n    return np.c_[np.ones(len(z)), z]\\n\\n\\ndef _align(panel):\\n    u = np.asarray(panel[\'u\'], dtype=float)\\n    v = np.asarray(panel[\'v\'], dtype=float)\\n    offset, scale = [], []\\n    for a, b in zip(u.T, v.T):\\n        F = one(a)\\n        ij = np.triu_indices(len(a), 1)\\n        da = (a[:, None] - a)[ij]\\n        db = (b[:, None] - b)[ij]\\n        mask = abs(da) > 1e-8\\n        slope = np.median(db[mask] / da[mask])\\n        inter = np.median(b - slope*a)\\n        for _ in range(5):\\n            r = abs(b - inter - slope*a)\\n            keep = r <= max(5*np.median(r), 0.00035)\\n            inter, slope = lin(F[keep], b[keep])\\n        offset.append(inter)\\n        scale.append(slope)\\n    return np.asarray(offset), np.asarray(scale)\\n\\n\\ndef _panel_maps(panel, z):\\n    u = np.asarray(panel[\'u\'], dtype=float)\\n\\n    def softsign(w, q):\\n        return one(w / (1 + q[0]*abs(w)) + q[1]*w)\\n\\n    def curvature(w, q):\\n        w2 = w*w\\n        return np.c_[one(w), w2, w2/(1+q[0]*w2)]\\n\\n    def fit(features, target, initial, bounds, **options):\\n        def residual(q):\\n            f = features(u, q)\\n            return (f @ lin(f, target) - target).ravel()\\n        q = least_squares(residual, initial, bounds=bounds, **options).x\\n        return features(z, q) @ lin(features(u, q), target)\\n\\n    c = fit(softsign, panel[\'v1\'], [.85, -.01], ([.02, -1], [10, 1]))\\n    d = fit(curvature, panel[\'v2\'], [.25], (.08, .65), gtol=1e-13)\\n    return np.c_[c, d]\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    train = study[\'train\']\\n    n = len(train[\'x\'])\\n    offset, scale = _align(study[\'panel_a\'])\\n    z = (np.vstack((train[\'x\'], study[\'test\'][\'x\'])) - offset)/scale\\n    w = _panel_maps(study[\'panel_c\'], z)\\n    w = (w-w[:n].mean(0))/np.maximum(w[:n].std(0), .1)\\n    return {\'features\': w}\\n", "node_01": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef lin(f, y):\\n    return np.linalg.lstsq(f, y, rcond=1e-9)[0]\\n\\n\\ndef _time_basis(t, trajectory):\\n    def basis(r):\\n        return -np.expm1(-t[:, None]*r)\\n\\n    def residual(r):\\n        f = basis(r)\\n        return (f @ lin(f, trajectory.T) - trajectory.T).ravel()\\n\\n    rates = least_squares(residual, [.4, 1.8], bounds=([.05, .8], [.8, 4])).x\\n    return basis(rates)\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    train = study[\'train\']\\n    targets = np.column_stack((train[\'y\'], train[\'traj\']))\\n    basis = _time_basis(np.asarray(study[\'eval_times\']), train[\'traj\'])\\n    return {\'targets\': targets, \'basis\': basis}\\n", "node_02": "import numpy as np\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    w = inputs[\'node_00\'][\'features\']\\n    targets = inputs[\'node_01\'][\'targets\']\\n    n = len(targets)\\n    gram = w @ w.T / w.shape[1]\\n    norms = gram.diagonal()\\n    distance = np.maximum(norms[:, None] + norms - 2*gram, 0)\\n    quadratic = (1+.25*gram)**2\\n    eye = np.eye(n)\\n    fit = np.mean([\\n        kernel[n:, :n] @ np.linalg.solve(kernel[:n, :n] + ridge*eye, targets)\\n        for kernel, ridge in ((quadratic, 3e-5),\\n                              (quadratic + np.exp(-.06*distance), 3e-4))\\n    ], axis=0)\\n    return {\'fit\': fit}\\n", "node_03": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n# Calibration coefficients estimated from the public reference panels.\\nCAL = np.array([[0.033827479672666985, 0.07502994949014716, 0.03556305388690401, -0.10920335674376619], [0.9809307101862607, 0.8629990455768406, 1.1573998900759128, 1.2002882102717807], [-0.31999471283555536, -0.31999428914834527, -0.3199574603474531, -0.3199905103871267], [0.31999582315928554, 0.3199835435188928, 0.3200148733329416, 0.3200118232407754], [0.1600124998192269, 0.15996380454409756, 0.15999319730988887, 0.1599900206487814]])\\nQ1 = np.array([0.8444230770112555, -0.15774263453922568])\\nC1 = np.array([[0.08326011432424367, 0.06829789349557093, 0.04102278316560223, 0.030190593886713187], [-0.09090505019568829, -0.3794373611692894, -2.1183688571394956, 0.7529232648949882], [2.103052405384543, -0.6120794597854645, -0.19762994861113498, -0.6077585328589148], [0.696333714798857, 2.1244236743953153, -0.28223555699490804, 0.35714874324838525], [0.5392346967343538, -0.4152752361446725, 0.776606145736224, 2.035328363589207]])\\nQ2 = np.array([0.3048459992624531])\\nC2 = np.array([[-0.1207648052269579, 0.1360333597485881, 0.11783672982901112, -0.029565077901737787], [-0.20304870862171004, -0.02471274076998639, -0.2898835380063854, -0.32775246826751175], [-0.11059919195815693, -0.37286054754508563, 0.255458694723335, -0.1294096507969399], [-0.01617743152819192, -0.28148548428968023, -0.2777258505751167, 0.27689821932616077], [-0.42377776906819997, 0.11986388713245974, 0.08283979766434472, 0.1801610153684999], [-0.037660355995836454, -0.005043739307974342, -0.05307504446828029, -0.06270179361098874], [-0.020558102803073297, -0.06962559548665995, 0.0472061949417082, -0.02325426754559209], [-0.0026717784963122397, -0.053143690113138374, -0.05081724531673, 0.05031377363364097], [-0.07841490412041142, 0.022668912580626355, 0.015059750981176031, 0.033010160891762085], [-0.07420465741751027, -0.008180795646081856, -0.10688233259644535, -0.11723769188038459], [-0.0402948846823113, -0.135793892319241, 0.09366734585008932, -0.04838078767792691], [-0.006409742927532261, -0.10144049928549337, -0.10234195017860293, 0.10256147157866055], [-0.1551631385490583, 0.04307952266761627, 0.030664858413288018, 0.06634480135008766]])\\nOFFSET = np.array([0.06766971726814551, 0.007561306101974368, 0.08093317019163747, -0.08329542359914982])\\nSCALE = np.array([1.1500970782315993, 1.1491194763935682, 1.1602102016682996, 1.305159236516379])\\n\\n\\ndef _one(z):\\n    return np.column_stack((np.ones(len(z)), z))\\n\\n\\ndef _curvature(z):\\n    h = z*z\\n    return np.column_stack((_one(z), h, h/(1+Q2[0]*h))) @ C2\\n\\n\\ndef _decode(w, n):\\n    c, d = w[:, :4], w[:, 4:]\\n    inv = np.linalg.inv(C1[1:])\\n    def latent(p):\\n        a = (c*np.exp(p[:4])+p[4:]-C1[0])@inv\\n        # Invert the softsign map on its central monotone branch.\\n        b = abs(a)\\n        q, r = Q1\\n        B = 1+r-q*b\\n        disc = B*B+4*r*q*b\\n        return np.sign(a)*2*b/np.maximum(B+np.sqrt(np.maximum(disc, 1e-12)), .02), disc\\n    def residual(p):\\n        z, disc = latent(p)\\n        v = _curvature(z)\\n        v = (v-v[:n].mean(0))/np.maximum(v[:n].std(0), .1)\\n        return np.r_[(v-d).ravel(), 8*np.minimum(disc, 0).ravel()]\\n    low = np.r_[np.full(4,-3.4),C1[0]-.65]\\n    high = np.r_[np.full(4,.2),C1[0]+.65]\\n    best = None\\n    for scale in (.22, .12, .32, .45):\\n        p0 = np.r_[np.full(4,np.log(scale)), C1[0]]\\n        opt = least_squares(residual, p0, bounds=(low,high), max_nfev=110)\\n        if best is None or opt.cost < best.cost:\\n            best = opt\\n        if opt.cost < 1e-13:\\n            break\\n    # Resolve any rare folds of the softsign using both panel maps together.\\n    z, _ = latent(best.x)\\n    def full(p):\\n        zz = p[8:].reshape(-1,4)\\n        soft = _one(zz/(1+Q1[0]*abs(zz))+Q1[1]*zz)@C1\\n        v = _curvature(zz)\\n        v = (v-v[:n].mean(0))/np.maximum(v[:n].std(0), .1)\\n        return np.r_[(4*(soft-c*np.exp(p[:4])-p[4:8])).ravel(), (v-d).ravel()]\\n    if best.cost > 1e-12:\\n        opt = least_squares(full, np.r_[best.x,z.ravel()], max_nfev=45, ftol=1e-10, gtol=1e-10)\\n        z = opt.x[8:].reshape(-1,4)\\n        loss = np.sqrt(np.mean(opt.fun**2))\\n    else:\\n        loss = np.sqrt(np.mean(best.fun**2))\\n    return z, loss\\n\\n\\ndef _calibrated(z):\\n    x = z*SCALE+OFFSET\\n    z = z.copy()\\n    j = np.roll(np.arange(4),-1)\\n    for _ in range(15):\\n        v = z[:,j]\\n        r = CAL[0]+CAL[1]*z+CAL[2]*v+CAL[3]*z*v+CAL[4]*z**3-x\\n        jac = np.zeros((len(z), 4, 4))\\n        k = np.arange(4)\\n        jac[:,k,k] = CAL[1]+CAL[3]*v+3*CAL[4]*z*z\\n        jac[:,k,j] = CAL[2]+CAL[3]*z\\n        z -= np.clip(np.linalg.solve(jac,r[...,None])[...,0],-.65,.65)\\n    soft = _one(z/(1+Q1[0]*abs(z))+Q1[1]*z) @ C1\\n    return np.column_stack((soft,_curvature(z)))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    w = np.asarray(inputs[\'node_00\'][\'features\'], dtype=float)\\n    t = inputs[\'node_01\']\\n    targets, basis = np.asarray(t[\'targets\']), np.asarray(t[\'basis\'])\\n    baseline = np.asarray(inputs[\'node_02\'][\'fit\'])\\n    n = len(targets)\\n    z, loss = _decode(w, n)\\n    pred = baseline.copy()\\n    if loss < .025:\\n        f = _calibrated(z)\\n        # Quadratic response in the recovered panel coordinates.\\n        f = (f-f[:n].mean(0))/np.maximum(f[:n].std(0), .1)\\n        i, j = np.triu_indices(f.shape[1])\\n        F = np.column_stack((_one(f), f[:,i]*f[:,j]))\\n        U, D, V = np.linalg.svd(F[:n], full_matrices=False)\\n        ridge = .02\\n        model = ((F[n:] @ V.T)*(D/(D*D+ridge))) @ (U.T @ targets)\\n        pred = .998*model+.002*baseline\\n    y = pred[:,0]\\n    state = (pred[:,1:] @ np.linalg.pinv(basis).T) @ basis.T\\n    diag = np.array([y.mean(),y.std(),state[:,-1].mean(),state[:,-1].std()])\\n    return {\'y_hat\': y, \'state_pred\': state, \'diagnostics_vec\': diag}\\n"}'

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

def _collect_numeric_leaf_digests(obj) -> set[str]:
    out = set()
    def _walk(v):
        if isinstance(v, bool) or v is None:
            return
        if isinstance(v, (int, float, np.integer, np.floating)):
            out.add(_canon_digest(v))
        elif isinstance(v, np.ndarray):
            if v.size > 0 and v.dtype.kind in ('f', 'i', 'u'):
                out.add(_canon_digest(v))
        elif isinstance(v, dict):
            for sub in v.values():
                _walk(sub)
        elif isinstance(v, (list, tuple)):
            for sub in v:
                _walk(sub)
    _walk(obj)
    return out

def _collect_subtree_digests(obj) -> set[str]:
    out = {_canon_digest(obj)}
    if isinstance(obj, dict):
        for sub in obj.values():
            out.update(_collect_subtree_digests(sub))
    elif isinstance(obj, (list, tuple)):
        for sub in obj:
            out.update(_collect_subtree_digests(sub))
    return out

class _TrackedPayloadDict(dict):
    def __init__(self, data: dict, src_id: str, consumed_set: set[str]):
        super().__init__(data)
        self._src_id = src_id
        self._consumed_set = consumed_set
    def __getitem__(self, key):
        val = super().__getitem__(key)
        self._consumed_set.add(self._src_id)
        return val
    def get(self, key, default=None):
        if super().__contains__(key):
            self._consumed_set.add(self._src_id)
        return super().get(key, default)
    def values(self):
        if len(self) > 0:
            self._consumed_set.add(self._src_id)
        return super().values()
    def items(self):
        if len(self) > 0:
            self._consumed_set.add(self._src_id)
        return super().items()

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
    raw_input_copy = deepcopy(study)
    raw_input_dig = _canon_digest(raw_input_copy)
    outputs = {'__input__': raw_input_copy}
    exec_counts = {nid: 0 for nid in _SPEC_OBJ['topo_order']}
    non_identity_checks = {}
    edge_consumption_checks = {}
    is_multi_node = len(_SPEC_OBJ['topo_order']) > 1
    for nid in _SPEC_OBJ['topo_order']:
        srcs = _SPEC_OBJ['incoming'][nid]
        consumed_srcs: set[str] = set()
        if not is_multi_node and srcs == ['__input__']:
            node_in = _TrackedPayloadDict(deepcopy(outputs['__input__']), '__input__', consumed_srcs)
        else:
            node_in = {
                s: (_TrackedPayloadDict(deepcopy(outputs[s]), s, consumed_srcs) if isinstance(outputs[s], dict) else deepcopy(outputs[s]))
                for s in srcs
            }
        in_dig = _canon_digest(outputs['__input__'] if (not is_multi_node and srcs == ['__input__']) else {s: outputs[s] for s in srcs})
        in_num_leaves = set()
        for s in srcs:
            in_num_leaves.update(_collect_numeric_leaf_digests(outputs[s]))
        consumed_srcs.clear()
        out_v = _COMPILED_FNS[nid](node_in)
        exec_counts[nid] += 1
        if exec_counts[nid] != 1:
            raise RuntimeError(f'DAG node {nid} executed {exec_counts[nid]} times != 1')
        if not isinstance(out_v, dict) or not out_v:
            raise RuntimeError(f'DAG node {nid} must return a non-empty dict')
        if is_multi_node:
            unused_srcs = sorted(set(srcs) - consumed_srcs)
            if unused_srcs:
                raise RuntimeError(f'DAG node {nid} left declared incoming edges unconsumed: {unused_srcs}')
            edge_consumption_checks[nid] = sorted(consumed_srcs)
        else:
            edge_consumption_checks[nid] = sorted(consumed_srcs) if consumed_srcs else list(srcs)
        out_dig = _canon_digest(out_v)
        if nid != _SPEC_OBJ['sink_node_id']:
            # Strict anti-wrapper-shell & genuine numeric transformation check (`U06`):
            out_subtrees = _collect_subtree_digests(out_v)
            echoed = (out_dig == in_dig) or (raw_input_dig in out_subtrees) or any(out_dig == _canon_digest(outputs[s]) for s in srcs)
            out_num_leaves = _collect_numeric_leaf_digests(out_v)
            novel_num_leaves = out_num_leaves - in_num_leaves
            if echoed or not out_num_leaves or not novel_num_leaves:
                raise RuntimeError(f'DAG node {nid} failed anti-wrapper-shell / novel numeric transformation check')
            non_identity_checks[nid] = True
        # Convert any tracked dict views returned by user code into plain dicts before freezing in outputs
        outputs[nid] = {k: (dict(v) if isinstance(v, _TrackedPayloadDict) else v) for k, v in out_v.items()}
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
            'edge_consumption_checks': edge_consumption_checks,
            'all_declared_edges_consumed': all(set(edge_consumption_checks.get(n, [])) == set(_SPEC_OBJ['incoming'][n]) for n in _SPEC_OBJ['topo_order']) if is_multi_node else True,
            'node_code_shas': _SPEC_OBJ['node_code_shas'],
        }
        open(trace_fp, 'w', encoding='utf-8').write(json.dumps(trace_obj, sort_keys=True))
    return outputs[_SPEC_OBJ['sink_node_id']]
