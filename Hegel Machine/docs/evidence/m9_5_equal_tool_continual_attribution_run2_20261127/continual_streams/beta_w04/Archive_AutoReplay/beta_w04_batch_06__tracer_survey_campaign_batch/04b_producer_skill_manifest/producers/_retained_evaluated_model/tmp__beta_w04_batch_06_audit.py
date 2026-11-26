import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, diags, eye, kron
from scipy.sparse.linalg import splu

root = Path('/holdout_task')
def load(name):
    spec = importlib.util.spec_from_file_location(name, root / (name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

m = load('beta_w04_batch_06_solver')
v = load('validate_public_schema')
archive = json.loads(Path('/tmp/hegel_continual_archive/archive_calibration_history.json').read_text())
full = []
for e in archive:
    b, x, y = (np.asarray(e[k]) for k in ('bank_id', 'ref_signal', 'raw_readout'))
    if min(np.bincount(b, minlength=4)) >= 2:
        full.append((b, x, y, np.array([np.polyfit(x[b == k], y[b == k], 1) for k in range(4)])))

# Use only two bank-0 points to select a profile, then score banks 1-3.
worst_cal = float(0)
for idx, (b, x, y, _) in enumerate(full):
    profiles = np.stack([f[3] for j, f in enumerate(full) if j != idx])
    err = np.abs(profiles[:, b, 0] * x + profiles[:, b, 1] - y)
    pairs = np.flatnonzero(b == 0)
    for j, i in enumerate(pairs):
        for k in pairs[j+1:]:
            pick = np.argmin(np.max(err[:, [i, k]], axis=1))
            worst_cal = max(worst_cal, float(np.max(err[pick, b != 0])))
assert worst_cal < 5e-6, worst_cal

# Reference stencil: accumulate face diffusion and row upwinding directly.
def axis(s, a):
    n = s['n'+a]
    h = s['L'+a] / n
    k, u, r = (s[t+'_'+a] for t in ('kappa','vel','reaction'))
    mat = np.diag(np.array(r, dtype=float))
    for i in range(n-1):
        q = 2 / (1/k[i] + 1/k[i+1]) / h**2
        mat[i:i+2, i:i+2] += q*np.array([[1,-1],[-1,1]])
    for i, speed in enumerate(u):
        adv = abs(speed)/h
        mat[i,i] += adv
        j = i-1 if speed >= 0 else i+1
        if 0 <= j < n:
            mat[i,j] -= adv
    for i in (0,n-1):
        robin = s['robin_'+a]
        mat[i,i] += robin*k[i]/(h*(robin*h/2 + k[i]))
    return csr_matrix(mat)

worst = dict(archived_unchecked_bank_abs_error=worst_cal, design_rel_l2=0., terminal_rel_l2=0., kkt_gap=0.)
scan = []
for entry in pickle.loads((root/'public_beta_w04_batch_06.pkl').read_bytes()):
    s = entry['study']
    nx, ny = s['nx'], s['ny']
    i, j = s['sensor_ij'].T
    V = s['V_reaction_2d']
    sep = V.mean(1)[:,None]+V.mean(0)[None,:]-V.mean()
    rel = np.linalg.norm(V-sep)/max(np.linalg.norm(V), 1e-30)
    assert rel <= 1e-6, rel
    op = kron(axis(s,'x'),eye(ny)) + kron(eye(nx),axis(s,'y')) + diags(V.ravel())
    rhs = s['source_modes'].reshape(-1,nx*ny).T
    modes = [splu((op + t*eye(nx*ny)).tocsc()).solve(rhs).reshape(nx,ny,-1) for t in s['mu_shifts']]
    ref = np.vstack([a[i,j] for a in modes])
    design, _ = m._forward(s)
    b = np.asarray(s['row_bank_id'])
    shift_idx = np.repeat(np.arange(len(modes)), len(i))
    for k in range(len(modes)):
        for wing in (0,1):
            cut = (shift_idx == k) & (b//2 == wing)
            if cut.any():
                worst['design_rel_l2'] = max(worst['design_rel_l2'], float(np.linalg.norm((design-ref)[cut])/np.linalg.norm(ref[cut])))
    sweep = s['sensor_calibration_sweeps']
    sb, sx, sr = (np.asarray(sweep[k]) for k in ('bank_id','ref_signal','raw_readout'))
    spot_errors = np.max(np.abs(m._ARCHIVE_PROFILES[:,sb,0]*sx + m._ARCHIVE_PROFILES[:,sb,1]-sr), axis=1)
    assert np.min(spot_errors) < 5e-6, spot_errors
    gain, offset = m._ARCHIVE_PROFILES[np.argmin(spot_errors)].T
    y = (s['y_obs_raw']-offset[b])/gain[b]
    z = s['Z_nuisance']
    ridge = s['nuisance_ridge']
    U, sig, _ = np.linalg.svd(z, full_matrices=False)
    d = np.column_stack((ref,y))
    d -= U @ ((sig**2/(sig**2+ridge))[:,None]*(U.T@d))
    out = m.solve_beta_w04_batch_06(s)
    w = out['weights']
    terminal = modes[-1] @ w
    worst['terminal_rel_l2'] = max(worst['terminal_rel_l2'], float(np.linalg.norm(terminal-out['terminal_field'])/np.linalg.norm(terminal)))
    cert = v.verify_box_simplex_kkt_certificate(d[:,:-1],d[:,-1],s['source_ridge'],s['u_box'],s['total_budget'],out['kkt_certificate'])
    assert cert['valid'] and np.array_equal(w,out['kkt_certificate']['weights']), cert
    worst['kkt_gap'] = max(worst['kkt_gap'], cert['gap'])
    # Fit nuisance only on the opposite wing; compare calibrated and raw transfer residuals.
    errors = []
    for wing in (0,1):
        test = b//2 == wing
        zz = z[~test]
        data = np.column_stack((ref,y,s['y_obs_raw']))
        fitted = np.linalg.solve(zz.T@zz + ridge*np.eye(z.shape[1]),zz.T@data[~test])
        residuals = data[test] - z[test]@fitted
        pred = residuals[:,:-2] @ w
        errors.append([float(np.sqrt(np.mean((pred-residuals[:,k])**2))) for k in (-2,-1)])
    scan.append(dict(separability_rel_l2=float(rel), spot_profile_max_abs=spot_errors.tolist(), heldout_wing_profiled_rmse_calibrated_vs_raw=errors))
assert worst['design_rel_l2'] < 1e-10 and worst['terminal_rel_l2'] < 1e-10, worst
print(json.dumps(dict(worst=worst, public_slices=scan)))
