import pickle,numpy as np
import rheo_base as m
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=p['dataset']; g=p['campaign_diagnostics']; tr=d['train']; te=d['test']; tt=d['eval_times']; dt=d['past_dt']; q=m._calibration(d['calibration_load_cells']); n=len(tr['lab_id']); x=m._features(tr,te); joint={k:np.concatenate([te[k],g[k]]) for k in ['lab_id','stratum','ref_coupons','coupon_features_raw']}; gx=m._features(tr,joint)[-len(g['lab_id']):]
def coupon(s,x): return m._stress(s['past_channel_raw'],q),m._stress(s['hold_channel_raw'],q),np.c_[np.ones(len(x)),x]
def y(s): return np.c_[s['strain_curves_obs']-s['strain_0'][:,None],s['instant_rate_obs']]
tc=coupon(tr,x[:n]); gc=coupon(g,gx); taus,b=m._fit(tc,dt,tt,y(tr)); h,H,X=gc; drift=X@b[2:]; design=m._design(gc,dt,tt,taus); pred=design@b; obs=y(g); gap=design[:,-1,:2]*taus
from scipy.optimize import least_squares
F=np.vstack((-np.expm1(-tt[:,None]/taus),1/taus))
z=H[:,None]-design[:,-1,:2]*taus
reset=H[:,None]*(F@b[:2]); base=design[:,:,:2]@b[:2]
def sat(tau): return np.r_[-np.expm1(-tt/tau),1/tau]
def fit(mask,kind):
 def pred(v):
  if kind=='reset_fixed': return reset+v[0]*drift[:,None]*sat(1)
  if kind=='reset_free': return reset+v[0]*drift[:,None]*sat(np.exp(v[1]))
  if kind=='memory_free': return base+v[0]*drift[:,None]*sat(np.exp(v[1]))
  if kind=='reset_two': return reset+drift[:,None]*(F@v)
  if kind=='reset_two_stress': return reset+(drift[:,None]*np.c_[np.ones(len(H)),H-3.3])@v.reshape(2,2)@F.T
  if kind=='reset_scaled': return (H[:,None]*(F@v[:2]) + drift[:,None]*(F@v[2:]))
  if kind=='memory_scaled': return reset+(z*v[:2])@F.T+drift[:,None]*(F@v[2:])
  if kind=='reset_fastslow': return reset+drift[:,None]*(v[0]*F[:,0]+v[1]*sat(np.exp(v[2])))
  if kind=='reset_offset': return reset+np.c_[np.ones(len(H)),drift,H-3.3]@v.reshape(3,2)@F.T
  if kind=='reset_linear': return reset+v[0]*drift[:,None]*np.r_[tt,1.]
  if kind=='drift_features': return reset+(X@v.reshape(5,2))@F.T
  if kind=='reset_quadratic': return reset+np.c_[drift,drift**2]@v.reshape(2,2)@F.T
 size={'reset_fixed':1,'reset_free':2,'memory_free':2,'reset_two':2,'reset_two_stress':4,'reset_scaled':4,'memory_scaled':4,'reset_fastslow':3,'reset_offset':6,'reset_linear':1,'drift_features':10,'reset_quadratic':4}[kind]
 v=np.ones(size)*.3
 if kind=='reset_scaled': v[:2]=b[:2]
 free=kind in ('reset_free','memory_free','reset_fastslow')
 lo=np.full(size,-np.inf); hi=-lo
 if free: lo[-1],hi[-1]=np.log([.05,40.]); v[-1]=np.log(3.)
 r=least_squares(lambda p:(pred(p)[mask]-obs[mask]).ravel(),v,bounds=(lo,hi),loss='soft_l1',f_scale=.012)
 return r.x,pred(r.x)
print('times',tt,'dt',dt,'diagnostic missing',np.isnan(g['coupon_features_raw']).sum(0))
print('reset without drift q90',np.quantile(abs(reset[:,:-1]-obs[:,:-1]),.9),np.quantile(abs(reset[:,-1]-obs[:,-1]),.9))
from itertools import product
R=obs-reset
u,s,v=np.linalg.svd(R,full_matrices=False); k=v[0]/v[0,-1]; amp=R@k/(k@k)
print('waveform:', k.round(5))
print('physical basis at fitted taus:',np.linalg.lstsq(F,k,rcond=None)[0], 'max mismatch',abs(F@np.linalg.lstsq(F,k,rcond=None)[0]-k).max())
print('fit physical waveform allowing taus to vary:')
def shape(p):
 t=np.exp(p[:2]); return np.r_[-np.expm1(-tt[:,None]/t)@p[2:], (1/t)@p[2:]]
w=least_squares(lambda p:shape(p)-k,np.r_[np.log(taus),np.linalg.lstsq(F,k,rcond=None)[0]]).x
print(np.exp(w[:2]),w[2:],np.max(abs(shape(w)-k)))
def features(sources, mode):
 keys=['lab_id','stratum','ref_coupons','coupon_features_raw']; p={j:np.concatenate([z[j] for z in sources]) for j in keys}; l=p['lab_id']; x=p['coupon_features_raw'].copy(); r=p['ref_coupons']; stats=[]
 for j in np.unique(l):
  a=l==j; z=r[a]; c=np.median(z.mean(0)); sc=np.median(z.std(0)); stats.append([c,sc]); x[a]=(x[a]-c)/sc
 _,i,n=np.unique(np.c_[l,p['stratum']],axis=0,return_inverse=True,return_counts=True)
 weights=1/n[i] if mode=='cell' else np.ones(len(i)); mu=np.nansum(x*weights[:,None],axis=0)/np.sum(np.isfinite(x)*weights[:,None],axis=0)
 return np.where(np.isnan(x),mu,x)[-len(amp):],np.asarray(stats)
for label,src in [('tr+te+g',[tr,te,g]),('tr+g',[tr,g]),('te+g',[te,g]),('g',[g])]:
 for mode in ['cell','pool']:
  x,stats=features(src,mode); D=x@b[3:]+b[2]; print(label,mode,stats.round(3).tolist());
  for kind,V in [('gain',D[:,None]),('gain+offset',np.c_[D,np.ones(len(D))])]:
   def coeff(mask): return least_squares(lambda v:V[mask]@v-amp[mask],np.ones(V.shape[1]),loss='soft_l1',f_scale=.018).x
   cv=np.zeros(len(D)); ss=[]
   for l in np.unique(g['lab_id']):
    msk=g['lab_id']==l; cv[msk]=V[msk]@coeff(~msk)-amp[msk]; ss.append(np.quantile(abs(cv[msk]),.9))
   print(kind,'coef',coeff(np.ones(len(D),bool)).round(4),'pooled/worst amp 90',np.quantile(abs(cv),.9).round(4),np.max(ss).round(4),'largest signed',cv[np.argsort(abs(cv))[-4:]].round(3))
for i in [6,13,15]: print('verification',i, 'lab',g['lab_id'][i],'amp',amp[i],'features',g['coupon_features_raw'][i], 'ref',g['ref_coupons'][i])
