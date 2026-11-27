import pickle, sys, importlib.util
import numpy as np
from itertools import combinations_with_replacement
spec=importlib.util.spec_from_file_location('archive','/tmp/hegel_continual_archive/alpha_m98_v2_c02_batch_03_solver.py'); archive=importlib.util.module_from_spec(spec); spec.loader.exec_module(archive)
it=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=it['dataset']; tr,te,g=d['train'],d['test'],it['campaign_diagnostics']; c=d['calibration_load_cells']; x,y=c['true_channel_loads'],c['bridge_raw']; dt,t=d['past_dt'],d['eval_times']
terms=[(0,0,0,0)]
for k in (1,2,3):
 for ids in combinations_with_replacement(range(4),k): terms.append(tuple(ids.count(i) for i in range(4)))
terms=np.array(terms); deriv=np.maximum(terms[None,:,:]-np.eye(4,dtype=int)[:,None,:],0)
def basis(u): return np.prod(u[:,None,:]**terms,axis=2)
def calibrate(rows):
 a=basis(x[rows]); b=np.linalg.lstsq(a,y[rows],rcond=None)[0]
 return b
keep=np.flatnonzero(np.max(abs(y),axis=1)<8); b=calibrate(keep)
if __name__=='__main__':
 for j in range(4): print('channel',j,[(tuple(term),round(v,6)) for term,v in zip(terms,b[:,j]) if abs(v)>.0006])

def invert(raw, b=b):
 z=raw.reshape(-1,4); u=np.linalg.solve(b[1:5].T,(z-b[0]).T).T
 for _ in range(32):
  r=basis(u)@b-z
  if np.max(abs(r))<1e-10: break
  jac=np.einsum('nik,ik,kj->nji', np.prod(u[:,None,None,:]**deriv,axis=-1), terms.T,b)
  step=np.linalg.solve(jac,r[...,None])[...,0]; h=np.ones((len(u),1)); nxt=u-step
  for __ in range(16):
   wrong=np.sum((basis(nxt)@b-z)**2,axis=1)>np.sum(r*r,axis=1)
   if not wrong.any(): break
   h[wrong]*=.5; nxt=u-h*step
  u=nxt
 return u.reshape(raw.shape)
def stress(raw): return invert(raw).sum(-1)

def score(e): return np.round([np.sqrt(np.mean(e**2)), np.median(abs(e)), np.max(abs(e))],6)

if __name__=='__main__':
 for q in np.array_split(np.random.default_rng(6).permutation(keep),5):
  ep=invert(y[q],calibrate(np.setdiff1d(keep,q)))-x[q]; print('bridge inverse fold', score(ep.sum(1)))
 for name,s in [('train',tr),('test',te),('diagnostic',g)]:
  for key in ('past_channel_raw','hold_channel_raw'):
   raw=s[key]; v=invert(raw); print('bridge closure/range',name,key,np.max(abs(basis(v.reshape(-1,4))@b-raw.reshape(-1,4))),np.round(np.quantile(v,[0,1]),3))
 xtr,xte=archive._standardize(tr,te); _,xg=archive._standardize({k:np.concatenate((tr[k],te[k])) for k in ('coupon_features_raw','ref_coupons','lab_id','stratum')},g)
 trainmat=archive._design(tr,xtr,stress,dt,t); yt=tr['strain_curves_obs']-tr['strain_0'][:,None]; rt=tr['instant_rate_obs']; tau,th=archive._fit(trainmat,yt,rt); yp,rp=trainmat(tau); print('intrinsic',tau,th,'residual curve/rate',score(yp@th-yt),score(rp@th-rt))
 gm=archive._design(g,xg,stress,dt,t); gy=g['strain_curves_obs']-g['strain_0'][:,None]; gr=g['instant_rate_obs']; a,r=gm(tau)
 for name,A,R in [('benchtop',a@th,r@th),('gate1',*archive._chamber(gm,tau,th,t))]:
  print('chamber',name,score(A-gy),score(R-gr),'by time mean',np.round(np.mean(gy-A,axis=0),4)); print('first six rows late and rate meas-pred',np.round(np.c_[(gy-A)[:6,::3],(gr-R)[:6]],4))
 print('chamber drift range',np.round(np.quantile(r[:,2:]@th[2:],[0,.5,1]),5))
