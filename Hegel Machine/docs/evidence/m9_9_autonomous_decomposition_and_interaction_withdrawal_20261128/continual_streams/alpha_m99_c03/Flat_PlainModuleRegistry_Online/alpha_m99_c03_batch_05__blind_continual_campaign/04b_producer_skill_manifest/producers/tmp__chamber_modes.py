from chamber_probe import *
from scipy.optimize import least_squares
A,G=m._design(tau,inp_d,dt,t)
past,hold,F=inp_d; z=hold[:,None]-G[:,:2]*tau; r0=F@th[2:]; n=len(F)
Y=np.c_[yd-A@th,rd-G@th]; tt=np.r_[t,0.]; rr=np.r_[t,1.]
def exp(k): return np.r_[-np.expm1(-t/k),1/k]
def basis(ks): return np.column_stack([*[exp(k) for k in ks],rr])
for scales in [(0,0),(1,0),(1,1),(0,1)]:
 R=Y-(z*th*1 if False else z*(th[:2]*scales))@basis(tau)[:,:2].T
 U,S,V=np.linalg.svd(R,full_matrices=False); print('removed',scales,'svals',S[:6].round(5)); print('corr',np.corrcoef(np.c_[U[:,:3],z,r0,F[:,1:]].T)[:3,3:].round(3))
 if scales not in [(1,0),(0,0)]: continue
 for name,k0 in [('one',[4.]),('two',[.65,4.])]:
  def prediction(p):
   b=basis(np.exp(p)); return np.linalg.lstsq(b,R.T,rcond=None)[0].T@b.T
  opt=least_squares(lambda p:(prediction(p)-R).ravel(),np.log(k0),bounds=(np.log(.06),np.log(100.)))
  print(name,np.exp(opt.x),score(opt.fun)); print('temporal modes',V[:2].round(4))
 print('missing',np.isnan(d['coupon_features_raw']).sum(axis=0),'diag vs deployed transform',np.max(abs(xd-xd2)))
 print('r0/slow amplitude/fast after removal/net slow rate/missing'); B=basis(tau); P=np.linalg.lstsq(B,R.T,rcond=None)[0].T; print(np.c_[r0,P[:,1],P[:,0],P[:,2]+P[:,1]/tau[1],np.isnan(d['coupon_features_raw']).sum(axis=1)].round(4))
