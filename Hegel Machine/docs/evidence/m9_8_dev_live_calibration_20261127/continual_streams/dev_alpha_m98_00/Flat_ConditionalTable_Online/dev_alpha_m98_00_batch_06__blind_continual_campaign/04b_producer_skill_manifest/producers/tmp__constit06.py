import pickle, numpy as np
import base06 as m
from nonlinear06 import *
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; g=i['campaign_diagnostics']; tr=d['train']; te=d['test']; dt,t=d['past_dt'],d['eval_times']; c=d['calibration_load_cells']; raw=c['bridge_raw']; loads=c['true_channel_loads']
co,keep=bridge_fit(loads,raw);print('excluded',np.flatnonzero(~keep)); print('mixed terms:',[(ix,np.round(v,5)) for ix,v in zip(MONOMIALS,co[1:]) if len(set(ix))>1 and max(abs(v))>.001]); print('forward max good',abs(co[0]+basis(loads[keep],POWERS)@co[1:]-raw[keep]).max()); print('inverse max good',abs(invert(raw[keep],co)-loads[keep]).max())
xx=m._aligned_features(tr,te); xt,xe=np.split(xx,[len(tr['lab_id'])]); xg=m._aligned_features(tr,g)[len(tr['lab_id']):]
def coupon(q,x):
 p,h=[invert(q[k],co).sum(-1) for k in ('past_channel_raw','hold_channel_raw')]; return p,h,x
def obs(q):return q['strain_curves_obs']-q['strain_0'][:,None],q['instant_rate_obs']
ct=coupon(tr,xt); cg=coupon(g,xg); tau,theta=m._fit(dt,t,ct,*obs(tr)); print('train fit',tau,theta)
for a,b in zip(obs(tr),[a@theta for a in m._design(tau,dt,t,ct)]): print('train error',np.quantile(abs(a-b),[.5,.9,.95,1]))
p,h,x=cg; dr=x@theta[2:]; Z=np.zeros((len(h),2));
for a,ps in zip(-np.expm1(-dt[:,None]/tau),p.T): Z+=a*(ps[:,None]-Z)
F=-np.expm1(-t[:,None]/tau); y,r=obs(g); base=h[:,None]*theta[:2]; b=np.linalg.lstsq(np.c_[F,t],y.T,rcond=None)[0].T; print('temporal noise and rate prediction',np.sqrt(np.mean((y-b@np.c_[F,t].T)**2)),np.sqrt(np.mean((r-b@np.r_[1/tau,1])**2)))
for j in range(2): print('branch unconstrained',j,np.linalg.lstsq(np.c_[base[:,j],Z[:,j]*theta[j],dr*tau[j]],b[:,j],rcond=None)[0]);
print('slope ratio',dr@b[:,-1]/(dr@dr))
for name,amp,slope in [('reset',base,dr),('05',base+dr[:,None]*tau*np.array([.52,.43]),.05*dr),('baseline',base-Z*theta[:2],dr)]:
 print(name,'curve rate RMS',np.sqrt(np.mean((y-amp@F.T-slope[:,None]*t)**2)),np.sqrt(np.mean((r-amp@(1/tau)-slope)**2)))
print('initial rate residual for reset',np.round(r-base@(1/tau)-dr,4))
