import numpy as np,pickle
from scipy.optimize import minimize_scalar,least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u=np.load('/tmp/probe.npz')['u']; n=len(u)
from itertools import combinations_with_replacement

def feat(x,kind,k):
 if kind=='cos': return np.cos(k*x)-1
 if kind=='sin': return np.sin(k*x)
 if kind=='rat': return x/(1+(k*x)**2)
 return np.tanh(k*x)
def design(x,kind,k,mode):
 return np.column_stack((np.ones(len(x)), *((x,) if mode&1 else ()), feat(x,kind,k), *((x*x,) if mode&2 else ())))
def panel(p,key):
 X,Y=p['u'],p[key]; out=[]
 for kind in ('cos','sin','rat','tanh'):
  for mode in (0,1,2,3):
   if kind=='cos' and not mode&1: continue
   def fit(k,full=False):
    h=design(X,kind,k,mode); q=np.linalg.qr(h,mode='reduced')[0]; b=np.linalg.lstsq(h,Y,rcond=None)[0]; r=h@b-Y; loo=np.mean(abs(r/(1-(q*q).sum(1))[:,None])); return (loo,np.max(abs(r)),design(u,kind,k,mode)@b) if full else loo
   opts=[minimize_scalar(fit,bounds=(a,b),method='bounded') for a,b in ((.15,1),(1,2),(2,3.5),(3.5,5))]; k=min(opts,key=lambda o:o.fun).x; l,r,pred=fit(k,True); out.append((l,r,kind,mode,k,pred))
 for fun in ('tanh','sin','rat'):
  pp=[]; errs=[]; loos=[]
  H=np.column_stack((np.ones(len(X)),X)); Hu=np.column_stack((np.ones(n),u))
  for j,y in enumerate(Y.T):
   f=lambda h,b: b[0]+b[1]*feat(h@b[2:7],fun,1)+(h[:,1:]@b[7:] if len(b)>7 else 0)
   init=np.linalg.lstsq(H,y,rcond=None)[0]; a=np.r_[0.,1.,init]; o=min([least_squares(lambda b:f(H,b)-y,a*c,max_nfev=300) for c in (.7,1,1.5)],key=lambda o:o.cost)
   pp.append(f(Hu,o.x)); errs.extend(o.fun); q=np.linalg.qr(o.jac,mode='reduced')[0]; loos.extend(o.fun/np.maximum(.01,1-(q*q).sum(1)))
  out.append((np.mean(abs(np.array(loos))),max(abs(np.array(errs))),fun+'-mix',0,1,np.array(pp).T))
 return sorted(out,key=lambda v:v[0])
D={}
for lab,key in [('b','v'),('c','v1'),('c','v2')]:
 label=lab+key; opts=panel(s['panel_'+lab],key); print(label,[(round(l,6),round(r,6),k,m,round(v,4)) for l,r,k,m,v,_ in opts[:7]]); D[label]=opts[0][-1]
np.savez('/tmp/panelprobe.npz',**D)
y=np.column_stack((s['train']['y'],s['train']['traj'])); rng=np.random.default_rng(15); splits=[(np.arange(52),np.arange(52,n))]+[(np.setdiff1d(np.arange(n),k),k) for k in np.array_split(rng.permutation(n),4)]
def score(g,alpha=.002):
 out=[]
 for tr,te in splits:
  h=g[tr]; p=alpha*np.eye(h.shape[1]); p[0,0]=0; d=abs(g[te]@np.linalg.solve(h.T@h+p,h.T@y[tr])-y[te]); out.append([d[:,0].mean(),d[:,1:].mean()])
 return np.round([out[0],np.max(out[1:],axis=0)],5)
def withq(x): return np.column_stack([np.prod(x[:,i],axis=1) for i in combinations_with_replacement(range(x.shape[1]),2)])
base=np.column_stack((np.ones(n),u)); core=np.column_stack((base,*D.values()));
for label,m in [('core',core),('bq',np.column_stack((core,withq(D['bv'])))),('c1q',np.column_stack((core,withq(D['cv1'])))),('c2q',np.column_stack((core,withq(D['cv2'])))),('uq',np.column_stack((core,withq(u)))),('allq',np.column_stack((core,withq(u),*[withq(v) for v in D.values()]))),('fusion',np.column_stack((core,*(v*u for v in D.values()),D['bv']*D['cv1'],D['cv1']*D['cv2'])) )]: print(label,score(m).tolist())
