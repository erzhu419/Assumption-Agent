import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/beta_w01_s00_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
p = pickle.load(open('/holdout_task/public_beta_w01_s00.pkl', 'rb'))[0]
s, gt = p['study'], p['gt']
rows = s['row_bank_id']
sw = s['sensor_calibration_sweeps']
banks, ref, rawref = (sw[k] for k in ('bank_id','ref_signal','raw_readout'))
gs, rs, ridge = gt['S_tilde'], gt['r_y'], s['source_ridge']
w0, f0 = gt['w_opt'], gt['terminal_field_A']

def objective(w):
    e = gs @ w - rs
    return .5 * (e @ e + ridge * (w @ w))

for gain in (np.ones(4), np.array([.78, 1.23, .93, 1.14])):
    offset = np.array([.068,-.037,.083,-.095])
    for held_sets in ([(0,), (1,), (2,), (3,)], [(0,1),(2,3)]):
        worst = np.zeros(5)
        for held in held_sets:
            retained = ~np.isin(banks, held)
            refs = [np.flatnonzero(banks == b) for b in held]
            case = dict(s, y_obs_raw=gain[rows] * s['y_obs_raw'] + offset[rows])
            for j in range(max(map(len, refs))):
                keep = retained.copy()
                keep[[ids[j % len(ids)] for ids in refs]] = True
                case['sensor_calibration_sweeps'] = dict(bank_id=banks[keep], ref_signal=ref[keep], raw_readout=(gain[banks]*rawref+offset[banks])[keep])
                out = m.solve_beta_w01_s00(case)
                w, field, cert = (out[k] for k in ('weights','terminal_field','kkt_certificate'))
                g = gs.T @ (gs @ w - rs) + ridge * w
                nu = cert['nu_sum']
                true_gap = g @ w - nu*s['total_budget'] - s['u_box']*np.minimum(g-nu, 0.).sum()
                worst = np.maximum(worst, (np.linalg.norm(w-w0)/np.linalg.norm(w0), np.linalg.norm(field-f0)/np.linalg.norm(f0), objective(w)-objective(w0), abs(true_gap), abs(cert['primal_objective']-cert['dual_lower_bound'])))
        print('gain=',gain,'held=',len(held_sets[0]),'worst [weight rel, field rel, true excess, true tangent gap, fitted gap]:',worst)
        assert worst[0] < .011 and worst[1] < .011 and worst[-1] < 1e-9
