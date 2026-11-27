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
from scipy.optimize import least_squares
GX=m._features(g,g)[len(H):]; L=g['lab_id']; pool=np.nanmean(np.where(np.isnan(g['coupon_features_raw']),np.nan,GX),axis=0); GX=np.where(np.isnan(g['coupon_features_raw']),pool,GX); D=b[2]+GX@b[3:]; R=obs-reset

def basis(tau): return np.r_[-np.expm1(-tt/tau),1/tau]
def law(p, kind):
 if kind=='intrinsic': return F@p
 if kind=='double_exp': return p[0]*basis(np.exp(p[2]))+p[1]*basis(np.exp(p[3]))
 if kind=='exp_linear': return p[0]*basis(np.exp(p[2]))+p[1]*np.r_[tt,1.]
 if kind=='exp_log': return p[0]*basis(np.exp(p[2]))+p[1]*np.r_[np.log1p(tt*np.exp(p[3])),np.exp(p[3])]
 if kind=='exp_rational': return p[0]*basis(np.exp(p[2]))+p[1]*np.r_[tt/(1+tt*np.exp(p[3])),1.]
 if kind=='log': return p[0]*np.r_[np.log1p(tt*np.exp(p[1])),np.exp(p[1])]
 if kind=='rational': return p[0]*np.r_[tt/(1+tt*np.exp(p[1])),1.]
 if kind=='exp1_exp': return p[0]*basis(1)+p[1]*basis(np.exp(p[2]))
import sys
sys.path.insert(0,'/holdout_task')
import batch_solver as candidate
train,test=candidate._prepare(d); target=y(tr); t,w=candidate._fit(train,dt,tt,target)
fb=lambda t: np.vstack((-np.expm1(-tt[:,None]/t),1/t))
for name,tau,theta in [('joint train/test',t,w),('train-only fit with archived preprocessing',taus,b)]:
  B=fb(tau); Y=obs-H[:,None]*(B@theta[:2]); DD=theta[2]+GX@theta[3:]; log=np.r_[np.log1p(tt),1.]; cur=B@np.array([.219,2.30]); print(name,tau,theta)
  for n,k in [('fixed log1p',log),('current candidate basis',cur)]:
    er=abs(DD[:,None]*k-Y); rows=[]
    for lab in np.unique(L):
      e=er[L==lab]; rows.append([np.quantile(e[:,:-1],.9),np.quantile(e[:,-1],.9)])
    print(n,'worst lab p90',np.max(rows,0),'rms',np.sqrt(np.mean(er**2)))
  p=[]
  for lab in np.unique(L):
    mask=L!=lab; v=least_squares(lambda q:(DD[mask,None]*law(q,'log')-Y[mask]).ravel(),[1.,0.],loss='soft_l1',f_scale=.004).x; p.append(v)
  print('log gain, log inverse time scale by fold',np.array(p).round(5))
print('benchtop signed errors median / absolute p90 by horizon'); e=candidate._design(train,dt,tt,t)@w-target; print(np.median(e,axis=0).round(4)); print(np.quantile(abs(e),.9,axis=0).round(4))
