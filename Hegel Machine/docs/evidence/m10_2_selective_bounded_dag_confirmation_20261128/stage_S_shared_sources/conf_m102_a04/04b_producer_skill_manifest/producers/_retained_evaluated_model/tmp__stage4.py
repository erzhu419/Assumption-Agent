import numpy as np,pickle
from scipy.optimize import least_squares
from itertools import product,combinations
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];x=s['train']['x'];T=s['train']['traj'];y=s['train']['y'];t=s['eval_times'];np.set_printoptions(precision=5,suppress=True,linewidth=150)
def aff(p,key='v'):
 u=p['u'];v=p[key];f=np.c_[np.ones(len(u)),u];i=np.arange(len(u))
 for _ in range(6):
  b=np.linalg.lstsq(f[i],v[i],rcond=None)[0];i=np.argsort(np.linalg.norm(f@b-v,axis=1))[:int(.85*len(u))]
 return b[0],b[1:]
a,A=aff(s['panel_a']);b,B=aff(s['panel_b']);c,C=aff(s['panel_c'],'v1');d,D=aff(s['panel_c'],'v2');print(a,A,b,B,sep='\n'); print('sv traj',np.linalg.svd(T,compute_uv=False));
def hat(F,Y):return np.linalg.lstsq(F,Y,rcond=None)[0]
def basis(p):return -np.expm1(-t[:,None]*p[None,:])
def err(p):
 f=basis(p);return (f@hat(f,T.T)-T.T).ravel()
r=least_squares(err,[.5,2],bounds=(.001,15));F=basis(r.x); Z=hat(F,T.T).T;print('time rates',r.x,'rmse',np.std(err(r.x)),'basis',F,'coef',Z[:8]);print('other time fits');
for fam in [lambda p:np.c_[t,-np.expm1(-p[0]*t)],lambda p:np.c_[-np.expm1(-p[0]*t),t*np.exp(-p[0]*t)],lambda p:np.c_[-np.expm1(-p[0]*t),-np.expm1(-p[0]*t)**2]]:
 r0=least_squares(lambda p:(fam(p)@hat(fam(p),T.T)-T.T).ravel(),[1.]);print(r0.x,np.std(r0.fun))
from scipy.linalg import polar
from sklearn.model_selection import KFold
from scipy.spatial.distance import cdist
p=s['panel_c']; u=p['u']; m=len(u)
for name in ('v1','v2'):
 v=p[name]; print('panel',name)
 for tag,f in [('lin',np.c_[np.ones(m),u]),('square',np.c_[np.ones(m),u,u*u]),('cube',np.c_[np.ones(m),u,u**3]),('quad',PolynomialFeatures(2).fit_transform(u)),('quad+cube',np.c_[PolynomialFeatures(2).fit_transform(u),u**3]),('lin+sin',np.c_[np.ones(m),u,np.sin(u)]),('lin+tanh',np.c_[np.ones(m),u,np.tanh(u)]),('lin+v1',np.c_[np.ones(m),u,p['v1']])]:
  if name=='v1' and tag=='lin+v1':continue
  q=hat(f,v); e=v-f@q; lev=np.sum(f*np.linalg.pinv(f).T,axis=1); print(tag,'rmse/loo',np.sqrt(np.mean(e**2)),np.sqrt(np.mean((e/(1-lev[:,None]))**2))); print(q.round(3)) if tag=='quad' else None
 print('lin singular',np.linalg.svd(hat(np.c_[np.ones(m),u],v)[1:],compute_uv=False))
print('B singular',np.linalg.svd(B,compute_uv=False))

def cf(u,name,kind):
 v=p[name]; f=lambda u:PolynomialFeatures(2).fit_transform(u)
 if kind=='quad':return f(u)@hat(f(p['u']),v)
 if kind=='aff':return np.c_[np.ones(len(u)),u]@hat(np.c_[np.ones(m),p['u']],v)
 if kind=='mixed':
  def fun(u):return np.c_[np.ones(len(u)),u,u*u,u**3]
  return fun(u)@hat(fun(p['u']),v)
frames={};
for aa,z in {'x':x,'Ai':(x-a)@np.linalg.inv(A),'Af':x@A+a}.items():
 frames[aa]=z
 for bb,zz in [('B',z@B+b),('Bi',(z-b)@np.linalg.inv(B))]:frames[aa+bb]=zz
for key,z in list(frames.items()):
 for nm in ('v1','v2'):
  for kind in ('quad','mixed'):
   zz=cf(z,nm,kind);frames[key+nm+kind]=zz;frames[key+'+'+nm+kind]=np.c_[z,zz]
 for kind in ('quad','mixed'): frames[key+'C'+kind]=np.c_[cf(z,'v1',kind),cf(z,'v2',kind)]
Y=np.c_[y,T];ii=np.arange(52);splits=list(KFold(4,shuffle=True,random_state=813).split(ii)); rows=[]
for name,z in frames.items():
 z=z/np.maximum(z[:52].std(0),.1);D2=cdist(z,z,'sqeuclidean')/z.shape[1]
 for typ in ('rbf','inv','poly','mixed'):
  for l in (.12,.3,.6,1.2,2.4):
   zc=z-z[:52].mean(0); G=zc@zc.T/z.shape[1]; K={'rbf':lambda:np.exp(-l*D2),'inv':lambda:1/(1+l*D2),'poly':lambda:(1+l*G)**3,'mixed':lambda:(1+.25*G)**2+np.exp(-l*D2)}[typ]()
   for ridge in (.00003,.0003,.003,.03):
    E=np.vstack([Y[j]-K[np.ix_(j,i)]@np.linalg.solve(K[np.ix_(i,i)]+ridge*np.eye(len(i)),Y[i]) for i,j in splits]); ec=np.mean(abs(E),0);obj=max(ec[0],ec[1:].mean())
    ep=np.mean(abs(Y[52:]-K[52:,:52]@np.linalg.solve(K[:52,:52]+ridge*np.eye(52),Y[:52])),0)
    rows.append((obj,name,typ,l,ridge,ec[0],ec[1:].mean(),ep[0],ep[1:].mean()))
print('CV-ranked kernel benchmarks');print(*sorted(rows)[:25],sep='\n');print('best per target');print(*sorted(rows,key=lambda k:k[5])[:6],*sorted(rows,key=lambda k:k[6])[:6],sep='\n')
