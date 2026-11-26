exec(open('/tmp/alpha_ep02_cv.py').read().split('for lab in [-1,')[0])
from scipy.optimize import least_squares
from itertools import combinations
from collections import defaultdict

# Corrections affect only linear drift, so they preserve the zero-preload recurrence.
x=a.drift[:,-1,1:]; hold=a.hold
scaled=lambda x: x/np.sqrt(1+x*x)
z=scaled(x); h=scaled(hold)
terms=[z[:,i]*z[:,j] for i,j in combinations(range(z.shape[1]),2)]
F={
 'pairwise':np.column_stack(terms+[z*z,h[:,None]*z]),
 'rational':np.column_stack([z/(1+.5*(z[:,i]-z[:,j])[:,None]**2)-z for i,j in combinations(range(z.shape[1]),2)]),
}
labs_u=np.unique(labs); cache={}; stats=defaultdict(list)
for n in (1,2):
 for no in combinations(labs_u,n):
  tau,w=m._fit(a,y,complete & ~np.isin(labs,no)); cache[no]=y-a.matrix(np.log(tau))@w
  if n==1: print('fold',no[0],'taus',np.round(tau,5),'minimum bound/order margin',round(min(tau[0]-.2,tau[1]-tau[0],8-tau[1]),5))
for l in labs_u:
 e=cache[(l,)]; g=labs==l
 inner_y=np.zeros(len(labs))
 for j in labs_u[labs_u!=l]:
  r=cache[tuple(sorted((l,j)))]; msk=labs==j
  inner_y[msk]=np.median(r[msk,2:-1]/times[2:],axis=1)-.5*r[msk,-1]
 fit=~g & complete
 for name in ('current','pairwise','rational'):
  correction=np.zeros(len(labs))
  if name!='current':
   f=F[name]; center=f[fit].mean(axis=0); spread=np.maximum(f[fit].std(axis=0),.01); f=(f-center)/spread
   # Fixed shrinkage prevents the small panel from fitting measurement glitches.
   q=least_squares(lambda q:np.r_[f[fit]@q-inner_y[fit],np.sqrt(8)*q],np.zeros(f.shape[1]),loss='soft_l1',f_scale=.012).x
   correction[complete]=f[complete]@q
  delta=np.where(complete,.5,1)*e[:,-1]+correction
  er=np.abs(e[:,:-1]-delta[:,None]*times)
  late=max(np.median(er[(g&complete)&(tr['stratum']==s),-1]) for s in np.unique(tr['stratum'][g&complete]))
  proxy=np.median(np.abs(np.median(e[g,2:-1]/times[2:],axis=1)-delta[g]))
  stats[name].append((np.quantile(er[g],.9),late,proxy))
  print(l,name,'curve p90 / worst complete stratum late / rate slope median',np.round(stats[name][-1],5))
for name,v in stats.items(): print('worst',name,np.round(np.max(v,axis=0),5))
