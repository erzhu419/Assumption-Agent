import importlib.util
import pathlib
import sys
import time
import numpy as np

root = pathlib.Path('/holdout_task')
spec = importlib.util.spec_from_file_location('eq02', root / 'coupled_interferometry_certified_l0.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
sys.path.insert(0, '/tmp')
from hegel_wormhole_skills.autogenous_prim_box_ridge_bounds import box_ridge_cube_lower_bound

for p, seed in [(18, 62), (19, 100)]:
    rng = np.random.default_rng(seed)
    counts = np.array([[120, 40, 30], [37, 149, 44], [45, 32, 120], [183, 37, 36], [35, 129, 44]])
    c, s = np.repeat(np.arange(5), counts.sum(axis=1)), np.repeat(np.tile(np.arange(3), 5), counts.ravel())
    n = len(c)
    latent = rng.normal(size=(n, 8)) + (s[:, None] - 1) * np.linspace(-0.4, 0.4, 8)
    offsets = np.array([-2.8, 0.9, 2.5, -1.5, 4.1])
    gains = np.array([0.56, 1.45, 2.15, 0.91, 1.75])
    raw = offsets[c, None] + gains[c, None] * latent
    standards = offsets[c, None] + gains[c, None] * rng.normal(size=(n, 18))
    raw[rng.random(raw.shape) < .025] = np.nan
    standards[rng.random(standards.shape) < .05] = np.nan
    phase = rng.uniform(-np.pi, np.pi, (n, 2))
    carrier = rng.uniform(-np.pi, np.pi, (n, 2))
    ref, sig = np.exp(1j*carrier), np.exp(1j*(carrier+phase))
    h = np.column_stack((np.ones(n), s == 1, s == 2, latent, phase))
    theta = rng.normal(size=13) * 0.5
    idx = rng.permutation(n)
    dst, cal = idx[:220], idx[220:]
    x = rng.normal(size=(len(dst), p)) / np.sqrt(len(dst))
    x = .42*x + .85*np.repeat(rng.normal(size=(len(dst), (p+2)//3))/np.sqrt(len(dst)), 3, axis=1)[:, :p]
    target = np.zeros(p)
    target[[1, 5, 8, 12, 16]] = [1.2, -.6, 1.05, -.69, .8]
    y = h@theta + .05*rng.normal(size=n)
    y[dst] += x@target
    glitches = rng.choice(cal, int(.12*len(cal)), replace=False)
    y[glitches] += rng.choice([-11., 11.], len(glitches))
    study = dict(panel=dict(center_id=c, stratum=s, X_panel_raw=raw, controls=standards),
                 interferometry=dict(signal_phasor=sig, ref_phasor=ref), y_assay_raw=y,
                 calibration_indices=cal, stage3_indices=dst, X_cand=x, lam2=.025, lam0=.115, M=.8)
    tick = time.monotonic()
    out = mod.solve_coupled_interferometry_l0(study)
    elapsed = time.monotonic() - tick
    z, th, dest, sol = (out[k] for k in ('aligned_panel', 'calibration_weights', 'deconfounded_target', 'certified_selection'))
    assert (z.shape, th.shape, dest.shape) == ((n, 8), (13,), (len(dst),))
    assert z.dtype == th.dtype == dest.dtype == np.float64
    aligned = raw.copy()
    for i in range(5):
        ref_i = standards[c == i]
        columns = [v[np.isfinite(v)] for v in ref_i.T]
        columns = [v for v in columns if len(v) >= 2]
        aligned[c == i] = (raw[c == i] - np.median([v.mean() for v in columns])) / np.median([v.std() for v in columns])
    cell_weight = 1/counts[c, s]
    known = np.isfinite(aligned)
    for j in range(8):
        take = known[:, j]
        aligned[~take, j] = np.average(aligned[take, j], weights=cell_weight[take])
    np.testing.assert_allclose(z, aligned, atol=1e-14)
    np.testing.assert_allclose(dest, y[dst]-np.column_stack((h[:, :3], aligned, phase))[dst]@th, atol=1e-13)
    panel_mae = np.mean(abs(z-latent))
    cal_rel = np.linalg.norm(th-theta)/np.linalg.norm(theta)
    ols = np.linalg.lstsq(np.column_stack((h[:, :3], aligned, phase))[cal], y[cal], rcond=None)[0]
    ols_rel = np.linalg.norm(ols-theta)/np.linalg.norm(theta)
    leaves = sol['subcube_leaves']
    w = np.array(sol['weights'])
    f = .5*np.linalg.norm(x@w-dest)**2 + .5*.025*(w@w) + .115*np.count_nonzero(abs(w)>1e-8)
    assert abs(f-sol['objective']) < 1e-9 and np.max(abs(w)) <= .8
    gap = sol['objective']-sol['certified_lower_bound']
    ids = np.arange(1<<p); bits = 1<<np.arange(p); covered = np.zeros(len(ids), dtype=np.int16)
    for leaf in leaves:
        a, zmask = (np.asarray(leaf[k]) for k in ('fixed_active','fixed_zero'))
        assert a.dtype == zmask.dtype == bool and a.shape == zmask.shape == (p,) and not (a&zmask).any()
        covered[(ids & (bits@(a|zmask))) == (bits@a)] += 1
        expected = box_ridge_cube_lower_bound(x, dest, np.asarray(leaf['minorant_center']), .8, .115,
                    ridge=.025, fixed_zero=zmask, fixed_active=a, diagonal=np.asarray(leaf['minorant_diagonal']))
        assert leaf['lb'] <= expected + 1e-9
    print(f'p={p} time={elapsed:.4f}s panel_MAE={panel_mae:.6f} calibration_relL2={cal_rel:.6f} OLS_relL2={ols_rel:.6f} gap={gap:.3g} leaves={len(leaves)}', flush=True)
    assert panel_mae <= .11 and cal_rel <= .14 and 0 <= gap <= 1e-5 and 1 <= len(leaves) <= 2048
    assert (covered == 1).all()
    assert sol['certified_lower_bound'] == min(leaf['lb'] for leaf in leaves)
    assert elapsed <= 9
