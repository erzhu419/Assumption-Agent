import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/beta_w01_s00_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
p = pickle.load(open('/holdout_task/public_beta_w01_s00.pkl', 'rb'))[0]
s, gt = p['study'], p['gt']
d, _ = m._responses(s)
z = s['Z_nuisance']
rows = s['row_bank_id']
sw = s['sensor_calibration_sweeps']
banks = sw['bank_id']
x, raw_ref = sw['ref_signal'], sw['raw_readout']
y0 = m._calibrated_observations(s)
lam, cap, budget = (s[k] for k in ('source_ridge', 'u_box', 'total_budget'))
truth = gt['S_tilde'] @ gt['w_opt'] - gt['r_y']
opt = .5 * (truth @ truth + lam * (gt['w_opt'] @ gt['w_opt']))

def fit(y):
    q = m._profile(z, np.column_stack((d,y)), s['nuisance_ridge'])
    a, r = q[:, :-1], q[:, -1]
    w, _ = m._box_simplex(a.T @ a + lam*np.eye(a.shape[1]), -a.T @ r, cap, budget)
    e = gt['S_tilde'] @ w - gt['r_y']
    return np.linalg.norm(w-gt['w_opt'])/np.linalg.norm(gt['w_opt']), .5*(e@e+lam*(w@w))-opt

for gain_case in ((1., 1., 1., 1.), (.78, 1.23, .93, 1.14)):
    gain = np.asarray(gain_case)
    offset = np.array([.068,-.037,.083,-.095])
    yraw = gain[rows]*y0+offset[rows]
    calraw = gain[banks]*raw_ref+offset[banks]
    for slices in (((0,), (1,), (2,), (3,)), ((0,1),(2,3))):
        worst = np.zeros((2,4))
        for held in slices:
            keep = ~np.isin(rows, held)
            q = m._profile(z[keep], np.column_stack((d[keep],y0[keep])), 0.)
            a,r = q[:,:-1],q[:,-1]
            w,_ = m._box_simplex(a.T@a+lam*np.eye(a.shape[1]), -a.T@r, cap,budget)
            coeff = np.linalg.lstsq(z[keep],y0[keep]-d[keep]@w,rcond=None)[0]
            pred = d@w+z@coeff
            refs = [np.flatnonzero(banks==b) for b in held]
            for j in range(max(map(len,refs))):
                for mode in range(2):
                    y=y0.copy()
                    errors=[]
                    for b,ids in zip(held,refs):
                        i=ids[j%len(ids)]; use=rows==b
                        a=pred[use]-x[i]; braw=yraw[use]-calraw[i]
                        g=(a@braw)/(a@a) if mode else 1.
                        y[use]=x[i]+braw/g
                        errors.append(abs(g-gain[b]))
                    worst[mode] = np.maximum(worst[mode], [max(errors),np.max(abs(y-y0)),*fit(y)])
        print('gain',gain_case,'held_count',len(slices[0]),'columns: gain abs, observation max abs, weight rel L2, true excess loss')
        print('unit',worst[0], 'profiled',worst[1])
