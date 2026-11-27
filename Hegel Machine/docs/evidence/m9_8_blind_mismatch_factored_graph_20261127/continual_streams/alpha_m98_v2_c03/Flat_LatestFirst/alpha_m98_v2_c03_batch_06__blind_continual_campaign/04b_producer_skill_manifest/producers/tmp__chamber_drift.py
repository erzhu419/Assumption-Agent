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
v,p=fit(np.ones(len(H),bool),'reset_two'); k=F@v
print('coupon: lab stratum missing columns, drift, implied drift, projection error, median feature residual relative to lab center')
imp=(obs-reset)@k/(k@k)
for i in range(len(H)):
 ir=least_squares(lambda a:reset[i]+a[0]*k-obs[i],[imp[i]],loss='soft_l1',f_scale=.009).x[0]; print(i,g['lab_id'][i],g['stratum'][i],np.where(np.isnan(g['coupon_features_raw'][i]))[0],np.round([drift[i],ir,ir-drift[i],np.median(abs(reset[i]+ir*k-obs[i]))],4),X[i,1:].round(3))
print('benchtop missing and drift estimates:')
a=m._design(tc,dt,tt,taus); target=y(tr); err=target-a@b; tv=np.r_[tt,1.]; driftres=[]
for row in err:
 driftres.append(least_squares(lambda u:row-u[0]*tv,[0.],loss='soft_l1',f_scale=.02).x[0]);
for k0 in [0,1,2]:
 mask=np.isnan(tr['coupon_features_raw']).sum(1)==k0; print(k0,mask.sum(),np.quantile(abs(np.array(driftres)[mask]),[.5,.9,1]) if mask.any() else [])
print('ref example',np.vstack([tr['ref_coupons'][:3],g['ref_coupons'][:3]]).round(3))
print('keys',list(d)); print('telemetry summary',[(s,np.mean(d[s]['aux_chamber_telemetry'],axis=0),np.std(d[s]['aux_chamber_telemetry'],axis=0)) for s in ('train','test')]); print('test missing columns/by row', np.isnan(te['coupon_features_raw']).sum(0),np.unique(np.isnan(te['coupon_features_raw']).sum(1),return_counts=True))
