exec(open('/tmp/check_downstream.py').read().split('for name,m in')[0].replace("    print('benchtop held-out lab'", "    print('benchtop held-out lab'"))
# Check whether retained history or nonlinear load terms explain residuals on complete coupons.
from itertools import combinations
age = np.cumsum(dt[::-1])[::-1] - dt
Z = P @ (np.exp(-age[:,None]/tau)*-np.expm1(-dt[:,None]/tau))
phi = np.vstack((-np.expm1(-t[:,None]/tau), 1/tau))
k = np.r_[dn._drift_integral(t,dn._DRIFT),1.]
def modes(mask):
 def kernel(q): return np.r_[dn._drift_integral(t,q),1.]
 return least_squares(lambda q:(D[mask,None]*kernel(q)-E[mask]).ravel(),[.8,.3,8.5],bounds=([.1,0,2],[2,1,50]),loss='soft_l1',f_scale=.005).x
families={'reset': np.empty((len(H),len(k),0)), 'retained history': Z[:,None,:]*phi[None,:,:]*theta[:2], 'hold-history cross term': H[:,None,None]*Z[:,None,:]*phi[None,:,:]*theta[:2], 'saturating history': (Z/(1+Z**2))[:,None,:]*phi[None,:,:]*theta[:2], 'pairwise difference': (Z[:,0]-Z[:,1])[:,None,None]*k[None,:,None], 'three-way product': (H*Z[:,0]*Z[:,1])[:,None,None]*k[None,:,None]}
for label,A in families.items():
 errors=[]
 for l in np.unique(L):
  learn=(L!=l)&complete; val=(L==l)&complete
  q=modes(learn); kk=np.r_[dn._drift_integral(t,q),1.]; err=E-D[:,None]*kk
  if A.shape[-1]:
   c=np.linalg.lstsq(A[learn].reshape(-1,A.shape[-1]),err[learn].ravel(),rcond=None)[0];err=err-A@c
  errors.append((rms(err[val,:-1]),rms(err[val,-1])))
 print(label,'complete worst LOLO curve/rate',np.round(np.max(errors,axis=0),6))
# Compare clean-label fit sensitivity to the heavy-tailed strain channel.
P,H,F=dn._inputs(tr,xt);Y=tr['strain_curves_obs']-tr['strain_0'][:,None];R=tr['instant_rate_obs'];LT=tr['lab_id']
for loss,scale in [('soft_l1',.008),('cauchy',.018),('cauchy',.03)]:
 print(loss,scale,end=' ')
 for l in np.unique(LT):
  m=LT!=l
  def pred(q,m):
   c,r=dn._design(q[:2],P[m],H[m],F[m],dt,t);return c@q[2:],r@q[2:]
  def res(q):
   c,r=pred(q,m);return np.r_[(c-Y[m]).ravel(),r-R[m]]
  q=least_squares(res,np.r_[tau,theta],loss=loss,f_scale=scale).x
  c,r=pred(q,~m);print(tuple(np.round([np.quantile(abs(c-Y[~m]),.9),rms(r-R[~m])],6)),end=' ')
 print()
print('Available cohort keys:',sorted(tr),'eval time bounds:',t[0],t[-1])
