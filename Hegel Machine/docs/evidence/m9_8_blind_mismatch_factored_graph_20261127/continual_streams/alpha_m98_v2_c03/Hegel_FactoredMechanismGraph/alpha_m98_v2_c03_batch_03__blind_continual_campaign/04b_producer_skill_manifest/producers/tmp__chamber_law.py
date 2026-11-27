exec(open('/tmp/chamber_probe.py').read().split("print('parameters'")[0])
from scipy.optimize import least_squares
n,k=v.shape
z=hold[:,None]-m._design(ch,dt,np.array([1.]),taus)[:,-1,:2]*taus
J=np.vstack([E,1/taus]); H=hold[:,None]*(J@theta[:2]); M=z[:,None,:]*J[None,:,:]*theta[:2]
q=np.r_[-np.expm1(-t),1.]; D=drift[:,None]*q

def report(name,p,r=None):
 e=p-v if r is None else r
 print(name,'med',np.round(np.median(abs(e),axis=0),4),'rate_rmse',round(np.sqrt(np.mean(e[:,-1]**2)),5),'curve_rmse',round(np.sqrt(np.mean(e[:,:-1]**2)),5))

p=H-M.sum(2)+1.5*D; report('baseline memory + 1.5 saturation(tau=1)',p)
print('lab and missing diagnostics'); print(np.round(np.c_[c['lab_id'],np.isnan(c['coupon_features_raw']).sum(1),hold,z,drift,((v-H+M.sum(2))@q)/(q@q),np.median(v-p,axis=1),v[:,-1]-p[:,-1]],4))
print('imputations and reference statistics');print(fill)
for labj in np.unique(lab):
 print(labj,[(np.median(rr.mean(0)),np.median(rr.std(0))) for rr in (ref[lab==labj],c['ref_coupons'][c['lab_id']==labj])])
for extra in (np.empty((n,k,0)),q[None,:,None]*feat[:,None,:],q[None,:,None]*np.ones((n,1,1))):
 a=np.concatenate([H[:,:,None],-M,D[:,:,None],extra],2)
 y=v
 for loss in ['linear','soft_l1']:
  coef=least_squares(lambda b:(a@b-y).ravel(),np.r_[1.,1.,1.,1.5,np.zeros(extra.shape[-1])],loss=loss,f_scale=.013).x
  print('coef',np.round(coef,5));report(loss,a@coef)
# Fit a common drift time constant, retardation memory time multiplier, and deployed feature response on complete diagnostics.
ok=~np.isnan(c['coupon_features_raw']).any(1)
for free in [False,True]:
 def pred(p):
  zz=np.zeros_like(z)
  for s,dur in enumerate(dt): zz+=-np.expm1(-dur/(taus*p[4:6]))*(past[:,s,None]-zz)
  return H-(zz[:,None,:]*J[None,:,:]*theta[:2]*p[1:3]).sum(2)+(feat@(p[6:] if free else theta[2:]*p[3]))[:,None]*np.r_[-np.expm1(-t/p[0]),1/p[0]]+(p[0]*0)
 start=np.r_[1,1,1,1.5,1,1,theta[2:]*1.5] if free else np.r_[1,1,1,1.5,1,1]
 for use in [np.ones(n,dtype=bool),ok]:
  fit=least_squares(lambda p:(pred(p)-v)[use].ravel(),start,loss='soft_l1',f_scale=.012,bounds=(np.r_[.1,-10,-10,-10,.15,.15,np.full(len(start)-6,-10)],np.r_[10,10,10,10,8,8,np.full(len(start)-6,10)])); print('nonlinear',free,use.sum(),np.round(fit.x,5));report('all',pred(fit.x));report('complete',None,(pred(fit.x)-v)[ok])
