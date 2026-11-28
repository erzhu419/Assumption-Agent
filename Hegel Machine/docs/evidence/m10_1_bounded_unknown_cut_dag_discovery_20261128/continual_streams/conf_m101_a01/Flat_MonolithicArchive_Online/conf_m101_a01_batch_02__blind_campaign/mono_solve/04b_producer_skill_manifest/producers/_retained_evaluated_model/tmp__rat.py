from panels import *
from sklearn.model_selection import KFold
from scipy.optimize import least_squares
splits=[(np.arange(52),np.arange(52,72)),(np.arange(56),np.arange(56,72)),*KFold(4,shuffle=True,random_state=13).split(X)]
print('A',ab,'B',B,'t',s['eval_times']); print('C1 params',C1.__defaults__[1:]); print('C2 params',C2.__defaults__[1:]); print('raw quadratic y',ls(PolynomialFeatures(2).fit_transform(X),Y))
def run(A,i,j,mode,out):
 q=PolynomialFeatures(2).fit_transform(A); l=aug(A); a=q[i]; b=l[i]; yy=out[i]; m=a.shape[1];
 def make(p,a,b):
  if mode=='r1': return a/(1+(b@p)**2)[:,None]
  if mode=='r2': return a/(1+np.sum((b@p.reshape(-1,b.shape[1]).T)**2,axis=1))[:,None]
  if mode=='s1': return a/np.sqrt(1+(b@p)**2)[:,None]
  if mode=='rplus': return np.c_[a,a/(1+(b@p)**2)[:,None]]
  if mode=='sinplus': return np.c_[a,b*np.sin(b@p)[:,None]]
  if mode=='cos': return a*np.cos(b@p)[:,None]
  if mode=='escale': return a*np.exp(-np.minimum((b@p)**2,20))[:,None]
  if mode=='r4': return a/(1+(A0(a)@p)**2).sum(1)[:,None]
 def A0(a):return a[:,1:5]
 def cf(D):
  reg=1e-6 if mode!='rplus' else .001
  return np.linalg.solve(D.T@D+reg*np.eye(D.shape[1]),D.T@yy)
 rng=np.random.default_rng(4); fits=[]
 size=2*b.shape[1] if mode=='r2' else (4 if mode=='r4' else b.shape[1]);
 for p in np.r_[np.eye(size)*.7,rng.normal(0,.6,(6,size))]:
  def err(p):
   D=make(p,a,b); c=cf(D); return np.r_[D@c-yy,np.sqrt(1e-6 if mode!='rplus' else .001)*c,.004*p]
  r=least_squares(err,p,max_nfev=110); fits.append(r)
 r=min(fits,key=lambda r:r.cost);D=make(r.x,a,b);c=cf(D);return make(r.x,q[j],l[j])@c,r.cost,r.x,c
ans=[]
for n,d in routes(X):
 if n not in ['X','Ai','AiB','AiBi']:continue
 for k,A in [('d',d),('C1',C1(d)),('C2',C2(d)),('diff',C1(d)-C2(d)),('sum',C1(d)+C2(d)),('d+z',d+C1(d)),('d+w',d+C2(d))]:
  for mode in ['r1','r2','s1','rplus','sinplus','cos','escale']:
   i,j=splits[0]; h,c,p,a=run(A,i,j,mode,Y); err=abs(h-Y[j]).mean(); ans.append((err,c,n,k,mode,A));
for e,c,n,k,mode,A in sorted(ans,key=lambda r:r[1])[:12]:print('fit rank',round(e,6),round(c,7),n,k,mode)
for e,c,n,k,mode,A in sorted(ans,key=lambda r:r[0])[:16]:
 es=[]
 for i,j in splits[1:]:h,_,p,a=run(A,i,j,mode,Y);es.append(abs(h-Y[j]).mean())
 print('ration',n,k,mode,round(c,6),np.round([e]+es,5),flush=True)
