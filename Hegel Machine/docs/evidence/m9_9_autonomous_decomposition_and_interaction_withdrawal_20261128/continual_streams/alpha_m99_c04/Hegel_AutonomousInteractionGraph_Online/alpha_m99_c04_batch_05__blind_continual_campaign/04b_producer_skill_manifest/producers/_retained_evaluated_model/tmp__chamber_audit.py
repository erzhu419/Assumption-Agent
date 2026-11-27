from chamber_probe import *
import downstream_solver as m
from scipy.optimize import lsq_linear
_,xd=m._features(g,g)
D=theta[2]+xd@theta[3:]
L=H*np.sum(theta[:2]/taus)
obs=np.c_[g['strain_curves_obs'],R]
labs=g['lab_id']; missing=np.isnan(g['coupon_features_raw']).any(1)
# Use the same integrated kernel for each strain column and its derivative at zero.
K=np.r_[-taus[None,:]*np.expm1(-t[:,None]/taus[None,:]),[[1.,1.]]]
T=np.r_[t,1.]; B=.2*K[:,0]+.55*K[:,1]+.25*T
q,s=m._chamber_response(g,xd,t,taus,theta); base=np.c_[q,s]
def product(x,k): return x[:,None]*k[None,:]
# Physically motivated alternatives, fit only in the outer training labs.
terms={
 'drift partition':np.stack([product(D,K[:,0]-T),product(D,K[:,1]-T)],-1),
 'partition and history':np.stack([product(D,K[:,0]-T),product(D,K[:,1]-T),product(Z[:,0]*theta[0]/taus[0],K[:,0]),product(Z[:,1]*theta[1]/taus[1],K[:,1])],-1),
 'partition and nonlinear load':np.stack([product(D,K[:,0]-T),product(D,K[:,1]-T),product(L*(H-np.median(H)),K[:,0]-K[:,1]),product(L/(1+(H/np.median(H))**2),K[:,0]-K[:,1])],-1),
 'shared drift bias':np.broadcast_to(B[None,:,None],(n,len(T),1)),
}
for tag,A in [('candidate',None),*terms.items()]:
 pred=base.copy(); coeff=[]
 for lab in np.unique(labs):
  hold=labs==lab; sel=~hold & ~missing
  if A is not None:
   b=np.linalg.lstsq(A[sel].reshape(-1,A.shape[-1]),(obs-base)[sel].ravel(),rcond=None)[0]
   coeff.append(b); pred[hold]+=A[hold]@b
 E=abs(pred-obs)
 slices=[labs==j for j in np.unique(labs)]+[g['stratum']==j for j in np.unique(g['stratum'])]+[~missing,missing]
 def table(slices): return np.max([[E[a,:-1].mean(),E[a,-2].mean(),E[a,-1].mean()] for a in slices],axis=0).round(6)
 print(tag,'worst curve/late/rate',table(slices),'worst complete by lab',table([(labs==j)&~missing for j in np.unique(labs)]),'fold coefficients',np.round(coeff,4))
# Check whether the missing-feature residual follows the constrained drift kernel.
e=obs-base
for a in [~missing,missing]:
 drift_delta=(e[a]@B)/(B@B); after=e[a]-product(drift_delta,B)
 print('missing',bool(missing[a][0]),'signed kernel correction',round(drift_delta.mean(),6),'curve residual orthogonal to drift',round(abs(after[:,:-1]).mean(),6),'rate residual after projection',round(abs(after[:,-1]).mean(),6))
# Withheld strain/rate labels only; the specified unlabeled joint panel is retained.
for lab in np.unique(tr['lab_id']):
 hold=tr['lab_id']==lab
 sub=lambda mask:{k:v[mask] for k,v in tr.items()}
 tau,th=m._fit(sub(~hold),x[~hold],dt,t)
 c,r=m._design(sub(hold),x[hold],dt,t,tau)
 ec=abs(c@th-(tr['strain_curves_obs'][hold]-tr['strain_0'][hold,None])); er=abs(r@th-tr['instant_rate_obs'][hold])
 print('train held-out lab',lab,'curve/rate median',np.median(ec),np.median(er),'curve p95',np.quantile(ec,.95),'taus',tau)
