import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
s=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=d['calibration_load_cells']; x,y=c['true_channel_loads'],c['bridge_raw']; axis=np.count_nonzero(x,axis=1)<=1
# Known gross common-mode readout jumps are held apart from the clean-load accuracy test.
valid=np.max(np.abs(y-np.median(y,axis=0)),axis=1)<8
rng=np.random.default_rng(71); pools=[np.array_split(rng.permutation(np.flatnonzero(valid & mask)),5) for mask in (axis,~axis)]
result=[]
for j in range(5):
 f=np.r_[pools[0][j],pools[1][j]]; tr=np.setdiff1d(np.arange(len(x)),f)
 b=m._axial_load_map({'true_channel_loads':x[tr],'bridge_raw':y[tr]}); z=m._decode(y[f],b); yhat=np.sum(m._bridge_design(x[f])*b,axis=-1)
 a=np.c_[np.ones(len(x)),x]; keep=tr.copy(); err=np.linalg.norm(y[tr]-a[tr]@np.linalg.lstsq(a[tr],y[tr],rcond=None)[0],axis=1); keep=keep[err<=8*np.median(err)]
 aff=np.linalg.lstsq(a[keep],y[keep],rcond=None)[0]; za=np.linalg.solve(aff[1:].T,(y[f]-aff[0]).T).T
 # Challenge the sparse numerator with rational attenuation depending on an adjacent-channel sum or difference.
 qtr=np.stack(((x[keep]+np.roll(x[keep],-1,axis=1))**2,(x[keep]-np.roll(x[keep],-1,axis=1))**2),axis=-1)
 qte=np.stack(((x[f]+np.roll(x[f],-1,axis=1))**2,(x[f]-np.roll(x[f],-1,axis=1))**2),axis=-1)
 gtr,gte=m._bridge_design(x[keep]),m._bridge_design(x[f]); rational=np.empty_like(yhat)
 for k in range(4):
  def resid(p): return (gtr[:,k]@p[:6])/(1+qtr[:,k]@p[6:])-y[keep,k]
  fit=least_squares(resid,np.r_[b[k],0.,0.],bounds=(np.r_[[-np.inf]*6,0.,0.],np.inf),gtol=1e-12)
  rational[:,k]=(gte[:,k]@fit.x[:6])/(1+qte[:,k]@fit.x[6:])
 for label,mask in [('axis',axis[f]),('mixed',~axis[f])]:
  av,bv=((v[mask]-x[f][mask]).sum(axis=1) for v in (za,z)); raw=[np.sqrt(np.mean((v[mask]-y[f][mask])**2,axis=0)).max() for v in (yhat,rational)]
  result.append((label,np.sqrt(np.mean(av**2)),np.sqrt(np.mean(bv**2)),np.max(abs(bv)),*raw))
for label in ('axis','mixed'):
 values=np.array([r[1:] for r in result if r[0]==label]); print(label,'worst-fold metrics [robust affine axial RMSE, cyclic axial RMSE, cyclic axial max, cyclic raw RMSE, rational raw RMSE]:',values.max(0))
 assert values[:,2].max()<0.002
# Check for signed lab bias rather than classification metrics on a continuous-response task.
tr,te=d['train'],d['test']; xt,_=m._features(tr,te); sp=m._specimen(tr,xt,m._axial_load_map(c)); t,dt=d['eval_times'],d['past_dt']; obs=tr['strain_curves_obs']-tr['strain_0'][:,None]; rate=tr['instant_rate_obs']; lab=tr['lab_id']
for k in np.unique(lab):
 test=lab==k; at=lambda f:tuple(v[f] for v in sp); tau,w=m._fit(at(~test),dt,t,obs[~test],rate[~test]); curves,rates=m._branches(at(test),dt,t,tau); drift=sp[2][test]@w[2:]; print('held-out lab',k,'median signed strain/rate residual',np.median(curves@w[:2]+drift[:,None]*t-obs[test]),np.median(rates@w[:2]+drift-rate[test]))
