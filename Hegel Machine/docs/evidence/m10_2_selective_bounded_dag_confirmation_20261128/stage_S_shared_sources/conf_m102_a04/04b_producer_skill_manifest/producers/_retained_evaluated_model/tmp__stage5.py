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
from cfit import fitc
from sklearn.model_selection import KFold
from scipy.spatial.distance import cdist
p=s['panel_c'];u=p['u'];v=p['v2'];m=len(u);cs=fitc(p,True)
def sq(z):return np.c_[np.ones(len(z)),z,z*z]
def cub(z):return np.c_[sq(z),z**3]
for fn in [sq,cub,lambda z:np.c_[sq(z),np.tanh(z)],lambda z:np.c_[sq(z),np.sin(z)],lambda z:np.c_[sq(z),np.tanh(z)**2]]:
 f=fn(u);w=hat(f,v);E=v-f@w;h=np.sum(f*np.linalg.pinv(f).T,axis=1);print('v2 fit loo',np.std(E),np.std(E/(1-h[:,None])))
q=hat(sq(u),v);print('v2 square coefficients',q);print('v2 residual',np.linalg.svd(v-sq(u)@q,compute_uv=False))
Y=np.c_[y,T];splits=list(KFold(4,shuffle=True,random_state=813).split(np.arange(52)));rows=[]
for name,z in [('raw',x),('Ai',(x-a)@np.linalg.inv(A)),('AiB',((x-a)@np.linalg.inv(A))@B+b),('Af',x@A+a)]:
 for k,cf in enumerate(cs):
  c=cf(z);d=sq(z)@q
  for key,w in [('C',c),('D',d),('CD',np.c_[c,d]),('ACD',np.c_[z,c,d]),('CDB',np.c_[c,d,z@B+b]),('CBDB',np.c_[c@B+b,d@B+b]),('CofD',np.c_[c,cf(d)]),('DofC',np.c_[c,sq(c)@q])]:
   w=(w-w[:52].mean(0))/np.maximum(w[:52].std(0),.1);G=w@w.T/w.shape[1];D=cdist(w,w,'sqeuclidean')/w.shape[1]
   for typ in range(4):
    for l in (.06,.12,.25,.5):
     K=[(1+l*G)**3,np.exp(-l*D),(1+.25*G)**2+np.exp(-l*D),(1+l*G)**4][typ]
     for alp in (.00003,.0003,.003):
      def e(i,j):return Y[j]-K[np.ix_(j,i)]@np.linalg.solve(K[np.ix_(i,i)]+alp*np.eye(len(i)),Y[i])
      r=np.mean(abs(np.vstack([e(i,j) for i,j in splits])),axis=0);h=np.mean(abs(e(np.arange(52),np.arange(52,72))),axis=0)
      rows.append((max(r[0],r[1:].mean()),name,k,key,typ,l,alp,r[0],r[1:].mean(),h[0],h[1:].mean()))
print('joint');print(*sorted(rows)[:12],sep='\n');print('y');print(*sorted(rows,key=lambda z:z[-4])[:8],sep='\n');print('traj');print(*sorted(rows,key=lambda z:z[-3])[:8],sep='\n')
