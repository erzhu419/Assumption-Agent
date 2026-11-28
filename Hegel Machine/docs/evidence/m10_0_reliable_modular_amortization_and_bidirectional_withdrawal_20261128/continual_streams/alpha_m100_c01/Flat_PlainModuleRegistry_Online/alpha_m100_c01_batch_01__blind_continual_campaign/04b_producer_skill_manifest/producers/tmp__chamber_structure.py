exec(open('/tmp/chamber_probe.py').read().split('for name,v')[0])
from scipy.optimize import least_squares
np.set_printoptions(precision=5,suppress=True,linewidth=150)
Y=np.c_[y,yr]; Y0=np.c_[cg@theta,rg@theta]
def temporal(tau):
 return np.vstack([np.c_[-np.expm1(-t[:,None]/tau),t],np.r_[1/tau,1]])
def ampfit(tau):
 T=temporal(tau); A=Y@np.linalg.pinv(T.T); return A,Y-A@T.T
for name,tau in [('intrinsic',taus),('free',np.exp(least_squares(lambda u: ampfit(np.exp(u))[1].ravel(),np.log(taus)).x))]:
 A,res=ampfit(tau); print(name,tau,'temporal rmse',np.sqrt(np.mean(res**2)),'per-time',np.sqrt(np.mean(res**2,0)))
 past=pg; B=np.c_[np.ones(len(hg)),hg,past,fg[:,1:]]
 cb=np.linalg.lstsq(B,A,rcond=None)[0]
 print('full weights rows [1,hold,past6,X4]',cb,'rmse',np.sqrt(np.mean((B@cb-A)**2,0)))
 age=np.cumsum(dt[::-1])[::-1]-dt; mem=np.exp(-age[:,None]/taus)*-np.expm1(-dt[:,None]/taus); gap=hg[:,None]-past@mem
 for j in range(3):
  if j<2:
   print('base memory',j,-mem[:,j]*theta[j], 'amp corrections', cb[2:8,j]+mem[:,j]*theta[j]);
   for desc,Q in [('gap,last',np.c_[gap[:,j],past[:,-1]]),('gap,step-last',np.c_[gap[:,j],hg-past[:,-1]]),('hold,last',np.c_[hg,past[:,-1]]),('gap,hold,last',np.c_[gap[:,j],hg,past[:,-1]])]:
    q=np.linalg.lstsq(Q,A[:,j],rcond=None)[0]; print(j,desc,q,np.sqrt(np.mean((Q@q-A[:,j])**2)))
  else:
   for desc,Q in [('features',fg),('features+hold',np.c_[fg,hg])]:
    q=np.linalg.lstsq(Q,A[:,j],rcond=None)[0]; print(j,desc,q,np.sqrt(np.mean((Q@q-A[:,j])**2)))
print('raw Y sv',np.linalg.svd(Y,compute_uv=False)); print('diag labs and counts',np.unique(g['lab_id'],return_counts=True)); print('joint missing',np.isnan(xp).sum(0),'diag missing',np.isnan(g['coupon_features_raw']).sum(0))
