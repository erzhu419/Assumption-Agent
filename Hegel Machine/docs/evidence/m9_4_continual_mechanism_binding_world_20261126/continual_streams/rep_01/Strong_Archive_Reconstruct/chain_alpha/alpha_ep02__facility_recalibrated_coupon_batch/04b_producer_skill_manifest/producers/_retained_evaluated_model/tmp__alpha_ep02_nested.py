exec(open('/tmp/alpha_ep02_cv.py').read().split('for lab in [-1,')[0])
from scipy.optimize import minimize_scalar
from itertools import combinations
labs_u=np.unique(labs); cache={}
for size in (1,2):
 for excluded in combinations(labs_u,size):
  keep=~np.isin(labs,excluded)
  tau,w=m._fit(a,y,complete & keep)
  cache[excluded]=y-a.matrix(np.log(tau))@w

def score(e,k):
 err=e[:,:-1]-k*e[:,[-1]]*times
 # Clipping the loss derivative limits the influence of Cauchy glitches.
 s=0.04
 return np.mean(s*s*(np.sqrt(1+(err/s)**2)-1))

for lab in labs_u:
 e0=cache[(lab,)]; mask=labs==lab
 inner=np.vstack([cache[tuple(sorted((lab,j)))][(labs==j)&complete] for j in labs_u if j!=lab])
 opt=minimize_scalar(lambda k:score(inner,k),bounds=(0,1),method='bounded').x
 print('lab',lab,'inner complete blend',round(opt,3))
 for name,k in [('archive',.25),('nested',opt),('half',.5)]:
  e=e0[mask]; krow=np.where(complete[mask],k,1.0); res=e[:,:-1]-krow[:,None]*e[:,[-1]]*times
  late=np.abs(res[:,-1]); comp=complete[mask]; g=tr['stratum'][mask]; err=np.abs(res)
  # A robust per-row slope estimated from held-out curves is an independent proxy for latent initial rate.
  slope=np.median(e[:,2:-1]/times[2:],axis=1)
  print(name,'complete late median/p90',np.round([np.median(late[comp]),np.quantile(late[comp],.9)],5),'worst complete stratum late median',round(max(np.median(late[comp&(g==s)]) for s in np.unique(g)),5),'curve abs90',round(np.quantile(err,.9),5),'rate-vs-curve slope median',round(np.median(abs(krow*e[:,-1]-slope)),5),'imputed late median',round(np.median(late[~comp]),5))
