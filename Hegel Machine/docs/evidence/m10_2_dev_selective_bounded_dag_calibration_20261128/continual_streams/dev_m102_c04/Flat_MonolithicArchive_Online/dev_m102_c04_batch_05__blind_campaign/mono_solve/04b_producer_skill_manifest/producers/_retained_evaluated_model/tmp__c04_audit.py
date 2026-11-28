import importlib.util, pickle
import numpy as np
from scipy.optimize import least_squares
path='/holdout_task/'
spec=importlib.util.spec_from_file_location('candidate',path+'batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open(path+'public_batch.pkl','rb'))[0]['study']; t=s['train']; x=t['x']; time=s['eval_times']; n=len(x); order=np.arange(n)
D=m._design; fit=m._fit
z=m._calibrate(s['panel_a'],x); p,r=m._operators(s['panel_c'],z)
dot=(p*r).sum(1); cross=p[:,0]*r[:,1]; base=D(p,r,dot); both=D(p,r,dot,cross)
i,j=np.triu_indices(4,1)
diff=np.column_stack(((z[:,i]-z[:,j])**2,(z[:,i]+z[:,j])**2)); triple=np.column_stack([z[:,a]*z[:,b]*z[:,c] for a,b,c in [(0,1,2),(0,1,3),(0,2,3),(1,2,3)]])
def scalar_candidates():
    yield 'baseline', base
    yield 'no bilinear contraction',D(p,r)
    yield 'sum/difference squares',np.c_[base,diff]
    yield 'triple products',np.c_[base,triple]
    for name,q in [('dot',dot),('cross',cross),('difference',z[:,0]-z[:,1]),('norm',np.linalg.norm(z,axis=1))]:
        for a in [0.01,0.1,0.5,1.]:
            yield f'quotient {name} {a}', base/(1+a*q[:,None]**2)
features=list(scalar_candidates()); y=t['y']; trj=t['traj']; errors=[]; traj_errors=[]; row_resid=np.zeros(n); row_traj=np.zeros(n)
print('observed fields',list(s),'train fields',list(t),'telemetry shape',s['telemetry_vec'].shape,'times',time)
print('No observed cohort IDs, relay states, probability density, or control/value observations; scoring regression and relaxation, not balanced accuracy or PDE/control models.')
folds=[('required',order[52:])]+[(f'order{i}',v) for i,v in enumerate(np.array_split(order,6))]+[('high-norm',np.argsort(np.linalg.norm(z,axis=1))[-20:])]
for name,va in folds:
    tr=np.setdiff1d(order,va)
    def err(A, i, j): return np.mean(abs(A[j]@fit(A[i],y[i])-y[j]))
    inner=np.array_split(np.random.default_rng(345).permutation(tr),4)
    choose=lambda fs:min(fs,key=lambda kv:np.mean([err(kv[1],np.setdiff1d(tr,h),h) for h in inner]))
    rn,ra=choose([k for k in features if k[0].startswith('quotient')]); an,aa=choose(features[2:4])
    out=m.solve_batch(dict(s,train={k:t[k][tr] for k in ('x','y','traj')},test={'x':x[va]}))
    ye=out['y_hat']-y[va]; te=out['state_pred']-trj[va]; errors.append([np.mean(abs(ye)),err(features[1][1],tr,va),err(aa,tr,va),err(ra,tr,va)])
    if name.startswith('order'): row_resid[va]=ye; row_traj[va]=te[:,-1]
    ti=np.r_[tr,va]; fast=D(p,dot)[ti]; slow=D(r)[ti]; no_cross=m._trajectory(fast,slow,time,trj[tr]); free=both[va]@fit(both[tr],trj[tr]);
    # Alternative: allow an unconstrained initial offset in a two-rate relaxation.
    target=trj[tr].T
    def basis(k): return np.column_stack((np.ones(len(time)),-np.expm1(-time[:,None]*k)))
    k=least_squares(lambda k:(basis(k)@fit(basis(k),target)-target).ravel(),[1.8,.42],bounds=([1.2,.12],[2.6,.9])).x
    coef=fit(both[tr],fit(basis(k),target).T)
    free_initial=both[va]@coef@basis(k).T
    traj_errors.append([np.mean(abs(te)),np.mean(abs(no_cross-trj[va])),np.mean(abs(free-trj[va])),np.mean(abs(free_initial-trj[va]))]); print(name,'y base/no-dot/symmetry/quotient',np.round(errors[-1],6),an,rn,'trajectory base/no-cross/free-horizon/free-initial',np.round(traj_errors[-1],6))
print('worst y',np.max(errors,axis=0),'worst trajectory',np.max(traj_errors,axis=0))
for name,e in [('y',row_resid),('terminal',row_traj)]:
    print('order-block OOF bias',name,np.round([e[k].mean() for k in np.array_split(order,6)],6),'order correlation',round(np.corrcoef(order,e)[0,1],4))
# Reference-panel block jackknife: refit calibration and readouts, then score actual held-out outcomes.
tr,va=order[:52],order[52:]; a=s['panel_a']; c=s['panel_c']; panel_errors=[]
for key,panel in [('panel_a',a),('panel_c',c)]:
    idx=np.arange(len(panel['u']))
    for hold in np.array_split(idx,6):
        keep=np.setdiff1d(idx,hold); sub=dict(s,train={k:t[k][tr] for k in ('x','y','traj')},test={'x':x[va]}); sub[key]={k:v[keep] for k,v in panel.items()}; out=m.solve_batch(sub); panel_errors.append([np.mean(abs(out['y_hat']-y[va])),np.mean(abs(out['state_pred']-trj[va]))])
print('panel-block jackknife worst 52/20',np.max(panel_errors,axis=0)); assert np.max(panel_errors)<.038 and np.max(np.array(errors)[:,0])<.038 and np.max(np.array(traj_errors)[:,0])<.038
