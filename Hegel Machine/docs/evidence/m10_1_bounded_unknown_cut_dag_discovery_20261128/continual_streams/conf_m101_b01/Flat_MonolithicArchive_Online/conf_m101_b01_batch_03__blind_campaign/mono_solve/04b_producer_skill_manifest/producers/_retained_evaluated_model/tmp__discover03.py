import pickle,itertools as it
import numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x,y,T=tr['x'],tr['y'],tr['traj']; t=s['eval_times']; L=np.linalg.lstsq

def D(x): return np.c_[np.ones(len(x)),x]
def F(x,p): return np.column_stack([np.ones(len(x))]+[np.prod(x[:,a],1) for k in range(1,p+1) for a in it.combinations_with_replacement(range(4),k)])
def robust(A,v):
 n,p=A.shape; rg=np.random.default_rng(8); best=None
 for _ in range(400):
  i=rg.choice(n,p,replace=False); w=L(A[i],v[i],rcond=None)[0]; e=np.linalg.norm(A@w-v,axis=1); q=np.median(e)
  if best is None or q<best[0]: best=(q,e)
 q,e=best; mask=e<max(0.001,5*q)
 for _ in range(3):
  w=L(A[mask],v[mask],rcond=None)[0]; e=np.linalg.norm(A@w-v,axis=1); mask=e<max(.001,5*np.median(e))
 return w,e
pa,pb,pc=[s['panel_'+k] for k in 'abc']; u,v=pa['u'],pa['v']; A,e=robust(D(u),v); B=L(D(pb['u']),pb['v'],rcond=None)[0]; C=L(F(pc['u'],2),pc['v2'],rcond=None)[0];
if __name__=='__main__':
 np.set_printoptions(precision=5,suppress=True); print('times',t,'telemetry',s['telemetry_vec'],'summary',tr['summary']); print('A',A,'err percentiles',np.percentile(e,[0,25,50,75,85,90,95,100])); print('robust quad pct',np.percentile(robust(F(u,2),v)[1],[25,50,75,85])); print('C2 coefficient map',C)
z=(x-A[0])@np.linalg.inv(A[1:]); b=D(z)@B; cs={'raw':x,'cal':z,'b':b,'braw':D(x)@B};
for k in ['cal','b']:
 q=cs[k]; cs[k+'c1']=D(q)@L(D(pc['u']),pc['v1'],rcond=None)[0]; cs[k+'c2']=F(q,2)@C

def projected(q,T): return (q @ L(q,T.T,rcond=None)[0]-T.T).ravel()
def timefit(T,case):
 def make(p):
  r=np.exp(p); h=-np.expm1(-t[:,None]*r)
  if case=='2exp':return h
  if case=='linear-exp':return np.c_[t,h]
  if case=='exp-decay':return np.c_[h[:,0],np.exp(-t*r[1])]
  if case=='two-rational':return t[:,None]/(1+t[:,None]*r)
  if case=='poly-exp':return np.c_[h,t] 
  if case=='two-damped':return np.exp(-t[:,None]*r)
 n=1 if case=='linear-exp' else 2; f=least_squares(lambda p:projected(make(p),T),np.linspace(-1,1,n),bounds=(-8,5)); return np.exp(f.x),make(f.x),np.std(f.fun)
if __name__=='__main__':
 for k in ['2exp','linear-exp','exp-decay','two-rational','two-damped']:
  r,H,err=timefit(T[:52],k); print('timefit',k,r,err,np.std(projected(H,T[52:])))
 r,H,_=timefit(T[:52],'2exp'); W=T@np.linalg.pinv(H.T); V=np.c_[y,W]; print('std targets',V.std(0),'corrcoef',np.corrcoef(V.T)); print('std x cal b',x.std(0),z.std(0),b.std(0));
 for k,q in cs.items():
  for typ,G in [('lin',D(q)),('quad',F(q,2)),('cub',F(q,3)),('tanh',F(np.tanh(q),2)),('sin',F(np.sin(q),2))]:
   p=G@L(G[:52],V[:52],rcond=None)[0]; print(k,typ,'fit',np.round(np.mean(abs(p[:52]-V[:52]),0),4),'hold',np.round(np.mean(abs(p[52:]-V[52:]),0),4),'traj',round(np.mean(abs(p[52:,1:]@H.T-T[52:])),4))
