import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root+'beta_w01_batch_01_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

# Independent face-flux form, applied in array coordinates rather than via Kronecker assembly.
def axis_flux(u, length, k, vel, robin):
    h = length / len(k)
    k = np.asarray(k)
    face = (2 / (1/k[:-1] + 1/k[1:])) / h**2
    jump = u[1:] - u[:-1]
    flux = face[:, None] * jump
    result = np.zeros_like(u)
    result[:-1] -= flux
    result[1:] += flux
    backward = np.vstack((u[:1], jump))
    forward = np.vstack((-jump, u[-1:]))
    result += (np.maximum(vel, 0)[:, None]*backward + np.maximum(-vel, 0)[:, None]*forward) / h
    edge = k[[0,-1]]
    result[[0,-1]] += (robin*edge/(h*(edge+0.5*h*robin)))[:, None]*u[[0,-1]]
    return result

for index, case in enumerate(pickle.load(open(root+'public_beta_w01_batch_01.pkl','rb'))):
    s, gt = case['study'], case['gt']
    out = m.solve_beta_w01_batch_01(s)
    w = out['weights_by_budget'][1]
    u = out['terminal_field_mid_budget']
    rhs = np.einsum('m,mxy->xy',w,s['source_modes'])
    v = s['V_reaction_2d']
    interaction = v-v.mean(0)[None,:]-v.mean(1)[:,None]+v.mean()
    lhs = (axis_flux(u,s['Lx'],s['kappa_x'],s['vel_x'],s['robin_x']) +
           axis_flux(u.T,s['Ly'],s['kappa_y'],s['vel_y'],s['robin_y']).T +
           (v+s['reaction_x'][:,None]+s['reaction_y'][None,:]+s['mu_shifts'][-1])*u)
    equation = np.linalg.norm(lhs-rhs)/np.linalg.norm(rhs)
    certs = out['kkt_certificates']
    objective = max(abs(c['primal_objective']-g['primal_objective']) for c,g in zip(certs,gt['certs_by_budget']))
    sweeps = s['sensor_calibration_sweeps']
    row = s['row_bank_id']
    calibrated = m._calibrated_observations(s)
    print(index,'nonseparable interaction max',np.max(abs(interaction)),'terminal flux residual rel L2',equation,'public objective max abs error',objective)
    assert equation < 1e-10 and objective < 1e-7
    for wing in range(2):
        keep = sweeps['bank_id']//2 != wing
        reduced = dict(s,sensor_calibration_sweeps={k:np.asarray(a)[keep] for k,a in sweeps.items()})
        take = row//2 == wing
        transfer = np.max(abs(m._calibrated_observations(reduced)[take]-calibrated[take]))
        test = m.solve_beta_w01_batch_01(reduced)
        weights = np.max(abs(test['weights_by_budget']-out['weights_by_budget']))
        field = np.linalg.norm(test['terminal_field_mid_budget']-u)/np.linalg.norm(u)
        print('wing',wing,'historical transfer: observation abs / weights abs / field rel',transfer,weights,field)
        assert max(transfer,weights,field) < 2e-6
