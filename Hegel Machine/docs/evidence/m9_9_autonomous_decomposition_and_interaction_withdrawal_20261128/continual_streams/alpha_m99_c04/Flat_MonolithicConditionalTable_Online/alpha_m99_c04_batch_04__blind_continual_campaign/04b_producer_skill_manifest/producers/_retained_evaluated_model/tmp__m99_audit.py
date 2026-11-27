import importlib.util, pickle
import numpy as np
from scipy.optimize import least_squares
p='/holdout_task/'
s=importlib.util.spec_from_file_location('solver',p+'batch_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
with open(p+'public_batch.pkl','rb') as f: item=pickle.load(f)[0]
d=item['dataset']; cal=d['calibration_load_cells']; x,y=cal['true_channel_loads'],cal['bridge_raw']
# Hold out high-load sweeps as well as random folds. Learn support and reject
# gross calibration faults on the fitting rows only.
strength=np.linalg.norm(x,axis=1)
folds=list(np.array_split(np.random.default_rng(4).permutation(len(x)),4))+[np.argsort(strength)[-16:]]
for j,te in enumerate(folds):
    tr=np.setdiff1d(np.arange(len(x)),te)
    b=m._fit_bridge({k:cal[k][tr] for k in ('true_channel_loads','bridge_raw')})
    def forward(z):
        return b[0]+z@b[1]+sum(z[:, ii].prod(axis=1)[:,None]*c for ii,c in b[2])
    e=y-forward(x); good=tr[np.linalg.norm(e[tr],axis=1)<.01]
    assert len(good)>len(tr)-5
    # The full polynomial dictionary already searches pairwise sum/difference
    # powers, since their expansions lie in this span. Challenge the retained
    # sparse support with the full cubic and with rational denominator terms.
    cubic=np.column_stack([x[:,v].prod(axis=1) for v in m._TERMS]); full=cubic@np.linalg.lstsq(cubic[good],y[good],rcond=None)[0]
    rat=np.empty_like(y)
    for k in range(4):
        z=x[:,k]; w=x[:,(k+1)%4]; B=np.column_stack([np.ones(len(x)),x,z*w,z**3]); q=np.column_stack([z*z,w*w,(z-w)**2,(z+w)**2]); start=np.r_[np.linalg.lstsq(B[good],y[good,k],rcond=None)[0],np.zeros(q.shape[1])]
        def pred(co): return (B@co[:B.shape[1]])/(1+q@co[B.shape[1]:])
        fit=least_squares(lambda co:(pred(co)-y[:,k])[good],start,bounds=(np.r_[np.full(B.shape[1],-np.inf),np.zeros(q.shape[1])],np.full(len(start),np.inf)),gtol=1e-11)
        rat[:,k]=pred(fit.x)
    clean=te[np.linalg.norm(e[te],axis=1)<.01]
    load=np.abs(m._decode(y[clean],b).sum(1)-x[clean].sum(1))
    print('bridge slice',j,'n / faults',len(te),len(te)-len(clean),'P90 sparse/full/rational raw',*[float(np.quantile(np.linalg.norm(r[clean],axis=1),.9)) for r in (e,y-full,y-rat)],'max total load error',load.max())
# Check chamber initial-rate consistency and actual dependence on chronology.
assert item['campaign_diagnostics'] is None and m._history_withdrawn()
out=m.solve_batch(d); tau,theta=out['estimated_taus'],out['hereditary_weights']; b=m._fit_bridge(cal); tr=m._calibrate(d['train'],b); xt,xe=m._features(d['train'],d['test']); hold=m._decode(d['test']['hold_channel_raw'],b).sum(1); drift=theta[2]+xe@theta[3:]
zero,rate=m._withdrawn_response(hold,drift,np.array([0.,1e-6]),tau,theta)
assert np.all(zero[:,0]==0)
np.testing.assert_allclose(rate,out['initial_rate_pred'],rtol=0,atol=1e-12)
np.testing.assert_allclose(zero[:,1]/1e-6,rate,rtol=0,atol=1e-5)
dt,t=d['past_dt'],d['eval_times']; state=np.zeros((len(xt),2))
for past,h in zip(tr['past_stress'].T,dt):
    state=np.exp(-h/tau)*state-np.expm1(-h/tau)*past[:,None]
c,r=m._design(tr,xt,dt,t,tau)
np.testing.assert_allclose(r[:,:2],(tr['hold_stress'][:,None]-state)/tau,rtol=0,atol=1e-12)
np.testing.assert_allclose(c[:,:,:2],(tr['hold_stress'][:,None]-state)[:,None,:]*(-np.expm1(-t[:,None]/tau)),rtol=0,atol=1e-12)
print('chamber rate derivative max error',np.max(np.abs(zero[:,1]/1e-6-rate)))
for lab in np.unique(tr['lab_id']):
    mask=tr['lab_id']==lab; e=c[mask]@theta+tr['strain_0'][mask,None]-tr['strain_curves_obs'][mask]; re=r[mask]@theta-tr['instant_rate_obs'][mask]
    print('lab',lab,'signed strain residual median by horizon',np.median(e,axis=0).round(5),'signed rate median',round(float(np.median(re)),5))
