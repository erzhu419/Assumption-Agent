import pickle
import runpy
import numpy as np
from scipy.sparse import csc_matrix, diags, eye, kron
from scipy.sparse.linalg import splu

p = runpy.run_path('/holdout_task/beta_w01_batch_07_solver.py')
worst_design = worst_field = worst_archive_rmse = 0.
wing_b_rmse_ratios = []
with open('/holdout_task/public_beta_w01_batch_07.pkl', 'rb') as f:
    cases = pickle.load(f)
for case in cases:
    s = case['study']
    S, fields = p['_responses'](s)
    src = s['source_modes']
    m, nx, ny = src.shape
    axes = [csc_matrix(p['_build_1d'](s['L'+d], n, s['kappa_'+d], s['vel_'+d], s['robin_'+d], s['reaction_'+d])) for d,n in zip('xy',(nx,ny))]
    op = kron(axes[0], eye(ny), format='csc') + kron(eye(nx), axes[1], format='csc') + diags(s['V_reaction_2d'].ravel(), format='csc')
    I = eye(nx*ny, format='csc')
    i, j = s['sensor_ij'].T
    for k, mu in enumerate(s['mu_shifts']):
        dense = splu(op + mu*I).solve(src.reshape(m,-1).T)
        expected = dense[i*ny+j]
        actual = S[k*len(i):(k+1)*len(i)]
        worst_design = max(worst_design, np.linalg.norm(expected-actual)/np.linalg.norm(expected))
    got = fields.reshape(nx*ny,m)
    worst_field = max(worst_field, np.linalg.norm(got-dense)/np.linalg.norm(dense))

    rb = np.asarray(s['row_bank_id'])
    raw = np.asarray(s['y_obs_raw'])
    Z = s['Z_nuisance']
    ridge = s['nuisance_ridge']
    gain, offset = p['_calibration'](s).T
    y = (raw-offset[rb])/gain[rb]
    a = s['sensor_calibration_sweeps']
    ab, x, r = (np.asarray(a[k]) for k in ('bank_id','ref_signal','raw_readout'))
    for b in np.unique(rb):
        held = rb == b
        z, design, target = Z[~held], S[~held], y[~held]
        prof = p['_profile'](z, np.column_stack((design,target)), ridge)
        w, _ = p['_box_simplex'](prof[:,:-1], prof[:,-1], s['source_ridge'], s['u_box'], s['total_budget'])
        beta = np.linalg.solve(z.T@z + ridge*np.eye(z.shape[1]), z.T@(target-design@w))
        predicted = S[held]@w + Z[held]@beta
        q = ab == b
        gl, ol = np.linalg.lstsq(np.column_stack((x[q], np.ones(q.sum()))), r[q], rcond=None)[0]
        rmse = lambda g,o: np.sqrt(np.mean((g*predicted+o-raw[held])**2))
        chosen, direct = rmse(gain[b],offset[b]), rmse(gl,ol)
        worst_archive_rmse = max(worst_archive_rmse, chosen)
        if b >= 2:
            wing_b_rmse_ratios.append(chosen/direct)
        print(f'held-out bank {b}: chosen raw RMSE={chosen:.6g}, current-only raw RMSE={direct:.6g}')
print(dict(worst_shift_design_rel=worst_design, worst_all_cell_terminal_modes_rel=worst_field, worst_held_out_bank_raw_rmse=worst_archive_rmse, worst_wing_b_chosen_to_direct_rmse=max(wing_b_rmse_ratios)))
assert max(worst_design, worst_field) < 1e-9
assert max(wing_b_rmse_ratios) < 1
