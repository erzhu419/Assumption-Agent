import numpy as np,pickle,sys
from scipy.optimize import least_squares
sys.path.insert(0,'/tmp/hegel_continual_archive')
from dev_m102_c03_batch_10_solver import _coordinates,_ls,_bias
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u=s['panel_c']['u']; x=s['train']['x']; z=_coordinates(s,x); Y=np.column_stack([s['train']['y'],s['train']['traj']]); print('coordinate range',z.min(0),z.max(0))
ff={'sin':np.sin,'cos':np.cos,'tanh':np.tanh,'atan':np.arctan,'sat':lambda t:t/(1+abs(t)),'gauss':lambda t:np.exp(-t*t),'linear':lambda t:t,'quad':lambda t:t*t,'cubic':lambda t:t**3,'rational':lambda t:t/(1+t*t)}
def score(make,p,v):
 f=_bias(make(u,p)); return (f@_ls(f,v)-v).ravel()
results={}
for k in ('v1','v2'):
 v=s['panel_c'][k]; res=[]
 for a,fa in ff.items():
  for desc,make,p0,lo,hi in [(a,lambda t,p:fa(p[0]*t),[1.7],[.03],[5.]),(a+'+self',lambda t,p:fa(p[0]*t)+p[1]*t,[1.7,.2],[.03,-3],[5,3])]+[(f'{a}+{b}:roll{r}',lambda t,p,fb=fb,r=r:fa(p[0]*t)+p[2]*np.roll(fb(p[1]*t),r,axis=1),[1.7,2.1,.25],[.03,.03,-3],[5,5,3]) for b,fb in ff.items() for r in (0,1,-1,2)]:
   p=least_squares(lambda p:score(make,p,v),p0,bounds=(lo,hi),max_nfev=75).x
   err=np.sqrt(np.mean(score(make,p,v)**2));res.append((err,desc,p))
   if err<1e-7: print('exact',k,desc,p,flush=True)
 res.sort(key=lambda j:j[0]); print(k,*res[:14],sep='\n',flush=True)
