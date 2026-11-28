import pickle, numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import cdist
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
fit=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
def one(x):return np.c_[np.ones(len(x)),x]
def poly(x,d):return np.column_stack([np.ones(len(x))]+[np.prod(x[:,i],axis=1) for j in range(1,d+1) for i in combinations_with_replacement(range(4),j)])
def a_features(x,j):
 k=(j+1)%4
 return np.c_[np.ones(len(x)),x[:,j],x[:,k],x[:,j]*x[:,k],x[:,j]**3]
def cal(x,inverse=False):
 p=s['panel_a'];u=p['u'];v=p['v'];g=np.linalg.norm(v-u,axis=1)<3; C=np.array([fit(a_features(u[g],j),v[g,j]) for j in range(4)])
 def f(z): return np.column_stack([a_features(z,j)@C[j] for j in range(4)])
 if not inverse:return f(x)
 z=x.copy();k=np.arange(4);q=(k+1)%4
 for _ in range(18):
  J=np.zeros((len(x),4,4));J[:,k,k]=C[:,1]+C[:,3]*z[:,q]+3*C[:,4]*z*z;J[:,k,q]=C[:,2]+C[:,3]*z
  z-=np.linalg.solve(J,(f(z)-x)[...,None])[...,0]
 return z

def candidates(u,v):
 def loo(a):
  q=np.linalg.qr(a)[0];r=v-a@fit(a,v);return np.sqrt(np.mean((r/(1-(q*q).sum(1))[:,None])**2))
 fs={'lin':one,'quad':lambda x:poly(x,2),'odd':lambda x:np.c_[one(x),x**3,x**5]}
 funcs={'sin':np.sin,'tanh':np.tanh,'atan':np.arctan,'rat':lambda a:a/(1+a*a),'exp':np.exp,'sincos':None,'linSin':None}
 for key,f in funcs.items():
  def fe(x,w,key=key,f=f):
   if key=='sincos': return np.c_[np.ones(len(x)),np.sin(w*x),np.cos(w*x)]
   if key=='linSin': return np.c_[one(x),np.sin(w*x)]
   return one(f(w*x))
  grid=np.linspace(.1,5,80); best=np.argmin([loo(fe(u,w)) for w in grid]); w=minimize_scalar(lambda w:loo(fe(u,w)),bounds=(grid[max(best-1,0)],grid[min(best+1,79)]),method='bounded').x
  fs[key+':'+str(round(w,4))]=lambda x,w=w,fe=fe:fe(x,w)
 return sorted([(loo(f(u)),name,lambda x,f=f,c=fit(f(u),v):f(x)@c) for name,f in fs.items()],key=lambda a:a[0])

def get():
 panels={}
 for p,k in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
  rr=candidates(s[p]['u'],s[p][k]);print(p,k,[(round(r,5),n) for r,n,_ in rr]);panels[p+k]=rr[0][2]
 return panels

def run(coords):
 t=s['train'];Y=np.c_[t['y'],t['traj']];splits=[(np.arange(56),np.arange(56,72))]+[(p[:56],p[56:]) for p in (np.random.default_rng(k).permutation(72) for k in range(5))];out=[]
 for name,x in coords.items():
  for deg in (1,2,3):
   a=poly(x,deg);ee=[]
   for tr,te in splits:
    pred=a[te]@fit(a[tr],Y[tr]);r=abs(pred-Y[te]);ee.append([r[:,0].mean(),r[:,1:].mean()])
   e=np.array(ee);out.append((name,'poly'+str(deg),e[0],e.max(0),e.mean(0)))
  for sc in (.4,.8,1.6,3.2):
   ee=[]
   for tr,te in splits:
    z=x/np.std(x[tr],axis=0);d=cdist(z,z,'sqeuclidean');K=np.exp(-d/(2*sc*sc)); a=one(z);rho=.003
    m=np.block([[K[np.ix_(tr,tr)]+rho*np.eye(len(tr)),a[tr]],[a[tr].T,np.zeros((5,5))]]);w=np.linalg.solve(m,np.r_[Y[tr],np.zeros((5,Y.shape[1]))]);r=abs(np.c_[K[np.ix_(te,tr)],a[te]]@w-Y[te]);ee.append([r[:,0].mean(),r[:,1:].mean()])
   e=np.array(ee);out.append((name,'gp'+str(sc),e[0],e.max(0),e.mean(0)))
 for i in range(2):
  print('best output',i)
  for name,m,o,w,a in sorted(out,key=lambda e:e[-1][i])[:13]:print(name,m,'56/16',o.round(5),'worst',w.round(5),'avg',a.round(5))
 print('linear all',[(n,o.round(4).tolist()) for n,m,o,w,a in out if m=='poly1'])
if __name__=='__main__':
 P=get();X=s['train']['x'];z={'raw':X,'Ainv':cal(X,True),'Afwd':cal(X)}
 coords=z|{a+'>'+b:g(x) for a,x in z.items() for b,g in P.items()};run(coords)
