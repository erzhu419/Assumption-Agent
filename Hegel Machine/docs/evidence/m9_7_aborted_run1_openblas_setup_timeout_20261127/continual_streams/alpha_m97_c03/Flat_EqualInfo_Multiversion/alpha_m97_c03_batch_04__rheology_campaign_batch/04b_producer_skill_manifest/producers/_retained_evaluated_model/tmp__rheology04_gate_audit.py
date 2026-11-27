import importlib.util
import pickle
import numpy as np
from pathlib import Path
root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('candidate', root / 'alpha_m97_c03_batch_04_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with (root / 'public_alpha_m97_c03_batch_04.pkl').open('rb') as f:
    d = pickle.load(f)[0]['dataset']
a, b = d['train'], d['test']
solve = m.solve_alpha_m97_c03_batch_04
from scipy.stats import spearmanr
lines = m._calibrate(d['calibration_load_cells'])
xc, _ = m._align(a,b)
stress,hold,_ = m._inputs(a,xc,lines)
dt,t,crit = (d[k] for k in ('past_dt','eval_times','crit_stress'))
damage = np.maximum(abs(stress)-crit,0)**2 @ dt
act = np.tanh(damage)
y = np.column_stack((a['strain_curves_obs'],a['instant_rate_obs']))
pred = np.empty_like(y)
for lab in np.unique(a['lab_id']):
    val = a['lab_id'] == lab
    fold = dict(d, train={k:v[~val] for k,v in a.items()}, test={k:np.concatenate((a[k][val],v)) for k,v in b.items()})
    o = solve(fold)
    pred[val] = np.column_stack((o['predicted_strain'][:val.sum()],o['initial_rate_pred'][:val.sum()]))
r = pred - y
for lab in np.unique(a['lab_id']):
    for name,slice_mask in [('inactive',act==0),('weak',(act>0)&(act<=0.2)),('strong',act>0.2)]:
        use = (a['lab_id']==lab) & slice_mask
        if use.any():
            print('lab/damage',lab,name,use.sum(),'curve/rate signed medians',np.round([np.median(r[use,:-1]),np.median(r[use,-1])],4),'curve/rate P90 abs',np.round([np.quantile(abs(r[use,:-1]),.9),np.quantile(abs(r[use,-1]),.9)],4))
# Probe coherent residual patterns, not isolated corrupted time cells.
z = np.zeros((len(hold),2))
o = solve(d)
for h,step in enumerate(dt):
    e = np.exp(-step/o['estimated_taus']); z = z*e + stress[:,h,None]*(1-e)
bases = {'hold': hold,'gap difference': z[:,1]-z[:,0], 'damage':act, 'activated memory': act*z[:,0], 'saturating memory':act*z[:,0]/(1+z[:,0]**2)}
for name,v in bases.items():
    rr = np.column_stack((np.median(r[:,:-1],axis=1),r[:,-1]))
    by_lab = [np.round([spearmanr(v[a['lab_id']==lab],q[a['lab_id']==lab]).statistic for q in rr.T],3).tolist() for lab in np.unique(a['lab_id'])]
    print('residual rank correlation',name,'pooled',np.round([spearmanr(v,q).statistic for q in rr.T],3),'by lab',by_lab)
# Measure the physical sensitivity to the published calibration discrepancy.
reference = np.array([[1.28994604,.12001392],[.81974036,-.14964349],[1.17001889,.1400143],[.7600149,-.11004836]])
_,xb = m._align(a,b)
u,v = (m._inputs(b,xb,c) for c in (lines,reference))
w = np.r_[o['hereditary_weights'],o['regime_mode_weights']]
args = (dt,t,crit,d['shear_banding_index']<=.5)
delta = (m._design(np.log(o['estimated_taus']),u,*args)-m._design(np.log(o['estimated_taus']),v,*args)) @ w
print('max calibration delta in physical stress',np.max(abs(u[0]-v[0])),'in strain',np.max(abs(delta[:,:-1])),'in rate',np.max(abs(delta[:,-1])))
