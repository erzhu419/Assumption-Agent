import copy, importlib.util, pickle, time
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('candidate', root+'alpha_m96_w01_batch_01_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
p = pickle.load(open(root+'public_alpha_m96_w01_batch_01.pkl','rb'))[0]
d, gt = p['dataset'], p['gt']

def run(name, data):
    start = time.monotonic()
    o = m.solve_alpha_m96_w01_batch_01(data)
    e = o['predicted_strain'] - gt['te_clean']
    te = data['test']
    rmse = lambda x: float(np.sqrt(np.mean(x*x)))
    lab = [rmse(e[te['lab_id']==j]) for j in range(4)]
    missing = ~np.isfinite(te['coupon_features_raw']).all(axis=1)
    cert = o['kkt_certificate']
    print(name, 'secs', round(time.monotonic()-start, 2), 'taus', np.round(o['estimated_taus'],5), 'tau relative max', np.max(abs(o['estimated_taus']/gt['taus']-1)), 'rmse', round(rmse(e),5), 'lab', np.round(lab,5), 'worst horizon', round(max(map(rmse,e.T)),5), 'complete/missing', np.round([rmse(e[~missing]),rmse(e[missing])],5), 'alloc max error', round(np.max(abs(o['allocation_weights']-gt['w_alloc_opt'])),7), 'gap', cert['primal_objective']-cert['dual_lower_bound'])

run('public', d)
for j in range(4):
    case = dict(d)
    keep = d['train']['lab_id'] != j
    case['train'] = {k:v[keep] for k,v in d['train'].items()}
    run('exclude train lab '+str(j),case)

case = copy.deepcopy(d)
c = case['calibration_load_cells']
gain = np.array([1.065, .94, 1.09, .91])
bias = np.array([.06, -.08, .14, -.12])
for split in (case['train'], case['test'], c):
    ids = split['lab_id']
    for k in ('past_stress_cmd','stress_hold_cmd','cmd_stress'):
        if k in split:
            a,b = gain[ids],bias[ids]
            if split[k].ndim==2:
                a,b = a[:,None],b[:,None]
            split[k] = (split[k]-b)/a
for j in range(4):
    ix = np.flatnonzero(c['lab_id']==j)
    c['true_stress'][ix[:2]] += [.17,-.23]
run('affine drift plus EMI',case)
keep = np.ones(len(c['lab_id']), bool)
ix = np.flatnonzero(c['lab_id']==2)
keep[ix] = False
keep[ix[3]] = True
case['calibration_load_cells'] = {k:v[keep] for k,v in c.items()}
print('sparse fitted gain, bias, identified:', m._calibrate(case['calibration_load_cells']))
run('affine drift plus EMI and one lab-2 check',case)
