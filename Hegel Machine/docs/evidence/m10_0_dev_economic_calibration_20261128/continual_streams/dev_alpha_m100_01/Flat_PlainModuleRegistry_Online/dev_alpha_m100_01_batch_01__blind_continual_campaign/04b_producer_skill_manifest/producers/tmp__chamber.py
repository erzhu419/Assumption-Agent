import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from explore import *
from base import _design
A,B=_design(taus,pc,hc,xc,dt,t); Ar,Br=_design(taus,pc[:,::-1],hc,xc,dt[::-1],t); Af,Bf=_design(taus,pc[:,::-1],hc,xc,dt,t)
y=np.c_[qc,rc]; bb=np.vstack([np.c_[-np.expm1(-t[:,None]/taus),t],np.r_[1/taus,1]]); c=lstsq(bb,y.T)[0].T
np.set_printoptions(precision=4,linewidth=190)
print('past / hold / lab / stratum and recovered [branch1 branch2 drift]');print(np.c_[pc,hc,ch['lab_id'],ch['stratum'],c]); print('refs',[(g['ref_coupons'].min(),g['ref_coupons'].max()) for g in [tr,te,ch]])
for name,a,b in [('baseline',A,B),('reverse paired',Ar,Br),('reverse loads',Af,Bf)]:
 coef=np.c_[b[:,:2]*taus*theta[:2],b[:,2:]@theta[2:]];e=y-np.c_[a@theta,b@theta]; print(name,'coef rmse',rms(c-coef,0),'curve & rate rmse',rms(e[:,:-1]),rms(e[:,-1])); print('residual coeff vs predictors',np.corrcoef(np.c_[c-coef,pc,hc,xc].T)[:3,3:])
for name,m in [('past',np.c_[np.ones(len(pc)),pc]),('past+hold',np.c_[np.ones(len(pc)),pc,hc]),('past+hold+x',np.c_[np.ones(len(pc)),pc,hc,xc]),('hold+x',np.c_[np.ones(len(pc)),hc,xc])]:
 for k in range(3):
  b=lstsq(m,c[:,k])[0]; h=(np.linalg.qr(m)[0]**2).sum(1); loo=(c[:,k]-m@b)/(1-h);print(name,k,'loo',rms(loo),'weights',b)
