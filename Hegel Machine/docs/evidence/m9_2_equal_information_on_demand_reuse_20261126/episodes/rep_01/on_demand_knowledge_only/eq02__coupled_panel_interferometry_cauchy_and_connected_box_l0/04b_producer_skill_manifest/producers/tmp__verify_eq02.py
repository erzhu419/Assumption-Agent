import importlib.util
import itertools
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import lsq_linear

root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('solution', root / 'coupled_interferometry_certified_l0.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def replay(x, y, l0, l2, box, cert):
    n, p = x.shape
    w = np.asarray(cert['weights'])
    leaves = cert['subcube_leaves']
    assert w.shape == (p,) and np.max(abs(w)) <= box and 1 <= len(leaves) <= 2048
    obj = .5 * (np.linalg.norm(x @ w - y)**2 + l2 * (w @ w)) + l0 * np.count_nonzero(abs(w) > 1e-8)
    assert_allclose(cert['objective'], obj, rtol=0, atol=2e-10)
    cube = np.arange(1 << p)
    covered = np.zeros(cube.size, bool)
    bits = 1 << np.arange(p)
    base = x.T @ x
    xy = x.T @ y
    eig = {}
    lower = np.inf
    min_slack = np.inf
    for leaf in leaves:
        zeros, actives = (leaf[k] for k in ('fixed_zero', 'fixed_active'))
        assert len(zeros) == len(actives) == p and all(type(i) is bool for i in zeros + actives)
        z, a = np.array(zeros), np.array(actives)
        assert not np.any(z & a)
        match = ((cube & bits[z].sum()) == 0) & ((cube & bits[a].sum()) == bits[a].sum())
        assert not covered[match].any()
        covered |= match
        free = ~(z | a)
        if 'dual_nu' in leaf:
            nu = np.asarray(leaf['dual_nu'])
            assert nu.shape == (n,)
            k, linear = l2, x.T @ nu
            constant = nu @ y - .5 * (nu @ nu)
        else:
            d, c = (np.asarray(leaf[k]) for k in ('minorant_diagonal', 'minorant_center'))
            assert d.shape == c.shape == (p,) and np.all(d >= 0)
            dt = tuple(d)
            if dt not in eig:
                eig[dt] = np.linalg.eigvalsh(base - np.diag(d))[0]
            assert eig[dt] >= -1e-10
            k, linear = d + l2, xy - (base - np.diag(d)) @ c
            constant = .5 * (y @ y - c @ (base - np.diag(d)) @ c)
        magnitude = abs(linear)
        amp = np.minimum(box, np.divide(magnitude, k, out=np.full(p, box), where=k > 0))
        active = l0 - amp * (magnitude - .5*k*amp)
        replayed = constant + active[a].sum() + np.minimum(active[free], 0).sum()
        assert leaf['lb'] <= replayed + 2e-9, (leaf['lb'], replayed)
        min_slack = min(min_slack, replayed - leaf['lb'])
        lower = min(lower, leaf['lb'])
    assert covered.all()
    assert cert['certified_lower_bound'] <= lower + 2e-9
    gap = obj - cert['certified_lower_bound']
    assert -2e-10 <= gap <= 1e-5
    return gap, len(leaves), min_slack

def design(rng, n, p, kind):
    x = rng.normal(size=(n,p))
    if kind == 'paired':
        x[:,1::2] = .97*x[:,:2*(p//2):2] + .245*x[:,1::2]
    elif kind == 'common':
        x = .3*x + rng.normal(size=(n,1))
    elif kind == 'ar':
        for j in range(1,p):
            x[:,j] = .88*x[:,j-1] + .475*x[:,j]
    elif kind == 'rankdef':
        x[:,-2:] = x[:,:2]
    return x / np.linalg.norm(x, axis=0)

def study_for(seed, p, kind):
    rng = np.random.default_rng(seed)
    cells = np.repeat(np.arange(15), rng.integers(15, 75, 15))
    rng.shuffle(cells)
    center, strata = cells // 3, cells % 3
    n = cells.size
    offsets = np.array([-2.0, .5, 1., -.8, 2.])
    gains = np.array([.5, .95, 1.6, .85, 1.2])
    true = rng.normal(size=(n,8)) + .4*strata[:,None]
    raw = true * gains[center,None] + offsets[center,None]
    raw[rng.random(raw.shape) < .025] = np.nan
    controls = rng.normal(size=(n,18)) * gains[center,None] + offsets[center,None]
    controls[rng.random(controls.shape) < .05] = np.nan
    phase = rng.uniform(-np.pi, np.pi, size=(n,2))
    carrier = rng.uniform(-np.pi, np.pi, size=(n,2))
    theta = rng.normal(size=13) * .4
    target = np.arange(n-110,n)
    cal = np.arange(n-110)
    x = design(rng, len(target), p, kind)
    w = np.zeros(p)
    w[rng.choice(p, 5, False)] = rng.choice([-1,1],5)*rng.uniform(.75,1.7,5)
    h = np.column_stack((np.ones(n), strata == 1, strata == 2, true, phase))
    y = h @ theta + rng.normal(0,.035,n)
    glitches = rng.choice(cal, int(.12*len(cal)), False)
    y[glitches] += np.clip(2*rng.standard_cauchy(len(glitches)), -11, 11)
    y[target] += x @ w
    study = dict(panel=dict(center_id=center, stratum=strata, controls=controls, X_panel_raw=raw), interferometry=dict(signal_phasor=np.exp(1j*(phase+carrier)), ref_phasor=np.exp(1j*carrier)), calibration_indices=cal, stage3_indices=target, y_assay_raw=y, X_cand=x, lam0=.14, lam2=.05, M=1.0)
    return study, theta, true

for seed, kind in enumerate(('iid','paired','ar','common','rankdef','paired','ar','common'), 31):
    p = 18 + seed % 2
    study, theta, true = study_for(seed, p, kind)
    t = perf_counter()
    try:
        result = mod.solve_coupled_interferometry_l0(study)
    except RuntimeError as error:
        print('FAIL', seed, kind, perf_counter()-t, str(error), flush=True)
        continue
    dt = perf_counter()-t
    cert = result['certified_selection']
    gap, leaves, slack = replay(study['X_cand'], result['deconfounded_target'], study['lam0'], study['lam2'], study['M'], cert)
    h = np.column_stack((np.ones(len(true)), study['panel']['stratum']==1, study['panel']['stratum']==2, result['aligned_panel'], np.angle(study['interferometry']['signal_phasor']*study['interferometry']['ref_phasor'].conj())))
    ix = study['stage3_indices']
    assert_allclose(result['deconfounded_target'], study['y_assay_raw'][ix] - h[ix] @ result['calibration_weights'], atol=1e-12)
    print(f'{seed} {kind} p={p}: panel_mae={np.mean(abs(result["aligned_panel"]-true)):.4f} theta_rel={np.linalg.norm(result["calibration_weights"]-theta)/np.linalg.norm(theta):.4f} gap={gap:.3g} leaves={leaves} time={dt:.4f} replay_slack={slack:.3g}', flush=True)

for seed in range(6):
    rng = np.random.default_rng(580+seed)
    x = design(rng, 25, 8, ('iid','ar','paired','common','rankdef','common')[seed])
    y = x @ rng.normal(size=8) + rng.normal(0,.2,25)
    l0, l2, box = (.07, .025, .4)
    cert = mod._BoxSearch(x, y, l0, l2, box).solve()
    gap, leaves, slack = replay(x, y, l0, l2, box, cert)
    values = [.5*y@y]
    for k in range(1,9):
        for s in itertools.combinations(range(8), k):
            aug = np.vstack((x[:,s], np.sqrt(l2)*np.eye(k)))
            sol = lsq_linear(aug, np.r_[y,np.zeros(k)], bounds=(-box,box), method='bvls').x
            values.append(.5*np.linalg.norm(aug@sol-np.r_[y,np.zeros(k)])**2 + l0*k)
    assert abs(cert['objective'] - min(values)) < 2e-10
    print(f'boxed exhaustive oracle {seed}: gap={gap:.3g} leaves={leaves}', flush=True)
