import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('candidate', root+'alpha_m96_w04_batch_04_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open(root+'public_alpha_m96_w04_batch_04.pkl', 'rb') as f:
    case = pickle.load(f)[0]
d, g = case['dataset'], case['gt']
tr, te = d['train'], d['test']
solver = m.solve_alpha_m96_w04_batch_04
coef = m._calibration(d['calibration_load_cells'])
print('calibration coefficient errors:', coef-np.column_stack((g['stress_gains'],g['stress_biases'])))
print('stress reconstruction max abs:', np.max(np.abs(coef[te['lab_id'],0,None]*te['past_stress_cmd']+coef[te['lab_id'],1,None]-g['te_hist_true'])))
out = solver(d)
for a,b in [('estimated_taus','taus'),('hereditary_weights','theta'),('viscoplastic_weights','vp_theta')]:
    print(a,'estimated',out[a],'truth',g[b],'relative L2',np.linalg.norm(out[a]-g[b])/np.linalg.norm(g[b]))
dh = np.max(np.abs(g['te_hist_true']),axis=1)>d['yield_stress']
ov = np.abs(g['te_hold_true'])>d['yield_stress']
e = out['predicted_strain']-g['te_clean']
r = out['initial_rate_pred']-g['te_true_rate0']
def report(label, mask, e=e, r=r):
    if mask.any():
        print(label,'n',mask.sum(),'curve/rate RMSE',np.sqrt(np.mean(e[mask]**2)),np.sqrt(np.mean(r[mask]**2)))
report('all test',np.ones(len(te['lab_id']),bool))
for lab in range(4):
    report('test lab '+str(lab), te['lab_id']==lab)
for i,j in [(0,0),(1,0),(0,1),(1,1)]:
    report('test past/hold above yield '+str((i,j)), (dh==i)&(ov==j))
print('error RMSE by hold time',np.sqrt(np.mean(e**2,axis=0)))
for lab in range(4):
    select = tr['lab_id']!=lab
    omitted = {k:v[select] for k,v in tr.items()}
    pred = solver({**d, 'train':omitted})
    report('clean test lab '+str(lab)+' with lab excluded from training labels',te['lab_id']==lab,pred['predicted_strain']-g['te_clean'],pred['initial_rate_pred']-g['te_true_rate0'])
    print('  taus',pred['estimated_taus'])
