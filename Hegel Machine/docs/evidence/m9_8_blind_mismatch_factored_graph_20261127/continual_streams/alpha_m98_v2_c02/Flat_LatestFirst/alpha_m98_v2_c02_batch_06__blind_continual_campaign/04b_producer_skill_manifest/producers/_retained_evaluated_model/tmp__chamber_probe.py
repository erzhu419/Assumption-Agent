from campaign_probe import *
from scipy.optimize import least_squares
xtr,xte=archive._standardize(tr,te); _,xg=archive._standardize({k:np.concatenate((tr[k],te[k])) for k in ('coupon_features_raw','ref_coupons','lab_id','stratum')},g)
ytr=tr['strain_curves_obs']-tr['strain_0'][:,None]; rtr=tr['instant_rate_obs']; tau,th=archive._fit(archive._design(tr,xtr,stress,dt,t),ytr,rtr)
Y=g['strain_curves_obs']-g['strain_0'][:,None]; R=g['instant_rate_obs']; P=stress(g['past_channel_raw']); H=stress(g['hold_channel_raw']); X=np.c_[np.ones(len(Y)),xg]; D=X@th[2:]; tail=np.cumsum(dt[::-1])[::-1]-dt
print('t/dt',t,dt); print('obs range',Y.min(),Y.max(),R.min(),R.max()); print('diag labs',np.unique(g['lab_id'],return_counts=True)); print('train feature range',xtr.min(),xtr.max()); print('diag feature range',xg.min(),xg.max())
def z(mem,reverse=False):
 if reverse: return P@((-np.expm1(-dt[:,None]/mem))*np.exp(-np.r_[0.,np.cumsum(dt)[:-1]][:,None]/mem))
 return P@((-np.expm1(-dt[:,None]/mem))*np.exp(-tail[:,None]/mem))
Z=z(tau); G=H[:,None]-Z
for label,A in [('full',np.c_[H,P,X]),('intrinsic',np.c_[H,Z,X]),('intrinsic drift',np.c_[H,Z,D,np.ones(len(D))]),('alternate drives',np.c_[H,P@dt,P[:,-1],D,np.ones(len(D))])]:
 b=np.linalg.lstsq(A,np.c_[R,Y],rcond=None)[0]; print(label,'R and curves rmse',np.round(np.sqrt(np.mean((A@b-np.c_[R,Y])**2,axis=0)),5)); print('coeff rate, t early/mid/late:',np.round(b[:,[0,1,6,12]],4))
def kernel(tau):return -np.expm1(-t[:,None]/tau)
def evaluate(mode,p,ids):
 if mode=='freeall':
  q=np.exp(p[:2]); mem=np.exp(p[2:4]); h=p[4:6]; m=p[6:8]; a=p[8]; b=p[9]; dur=np.exp(p[10]); off=p[11]; zz=z(mem)
 elif mode=='fixedmemory':
  q=np.exp(p[:2]); h=p[2:4]; m=p[4:6]; a=p[6]; b=p[7]; dur=np.exp(p[8]); off=p[9]; zz=Z
 elif mode=='fixedkernel':
  q=tau; mem=np.exp(p[:2]); h=p[2:4]; m=p[4:6]; a=p[6]; b=p[7]; dur=np.exp(p[8]); off=p[9]; zz=z(mem)
 elif mode=='reverse':
  q=np.exp(p[:2]); h=p[2:4]; m=p[4:6]; a=p[6]; b=p[7]; dur=np.exp(p[8]); off=p[9]; zz=z(tau,True)
 E=(H[:,None]*h-zz*m)*th[:2]; F=dur*-np.expm1(-t/dur); cur=E@kernel(q).T+D[:,None]*(a*t+b*F)+off*t; rate=E@(1/q)+D*(a+b)+off
 return np.r_[(cur[ids]-Y[ids]).ravel(),rate[ids]-R[ids]]
for mode in ['freeall','fixedmemory','fixedkernel','reverse']:
 p=np.r_[np.log(tau),np.log(tau) if mode=='freeall' else [],[1.,1.,1.,1.,.7,.3,0.,0.]]
 ids=np.arange(len(Y)); opt=least_squares(lambda p:evaluate(mode,p,ids),p,loss='soft_l1',f_scale=.012,max_nfev=350); e=opt.fun; print(mode,'params',np.round(opt.x,5),'curve/rate',score(e[:-len(ids)]),score(e[-len(ids):])); print('median level err',np.round(np.median(e[:-len(ids)].reshape(len(ids),-1),axis=0),4))
