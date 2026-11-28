import numpy as np,pickle,itertools
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from scipy.linalg import solve
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; t=s['eval_times']; X=tr['x']; Y=tr['traj']; y=tr['y']; n=len(X)
def aff(x):return np.c_[np.ones(len(x)),x]
def panel(u,v,model):
 def mat(x,r):
  h={'sin':lambda:np.sin(x*r),'tanh':lambda:np.tanh(x*r),'sq':lambda:x*x,'cube':lambda:x**3}[model]()
  return np.c_[aff(x),h]
 def err(r):
  a=mat(u,r);return (a@np.linalg.lstsq(a,v,rcond=None)[0]-v).ravel()
 r=least_squares(err,[2.],bounds=(.25,6.)).x if model in ['sin','tanh'] else 1.
 a=mat(u,r); q=np.linalg.qr(a)[0]; c=np.linalg.lstsq(a,v,rcond=None)[0]; e=v-a@c
 return np.sqrt(np.mean((e/(1-(q*q).sum(1))[:,None])**2)),r,lambda x:mat(x,r)@c,c
u,v=s['panel_a'].values(); a=aff(u); i=np.arange(len(u))
for _ in range(6):
 c=np.linalg.lstsq(a[i],v[i],rcond=None)[0]; i=np.argsort(np.linalg.norm(v-a@c,axis=1))[:40]
Z=(X-c[0])/np.diag(c[1:]); W=aff(X)@c
print('cal',c.round(5));print('svd',np.linalg.svd(Y,compute_uv=False)); b=lambda r:-np.expm1(-t[:,None]*r); err=lambda r:(b(r)@np.linalg.lstsq(b(r),Y.T,rcond=None)[0]-Y.T).ravel(); r=least_squares(err,[1.8,.4],bounds=([.7,.03],[5,.68])).x; print('rates',r,'time rmse',np.std(err(r)), 'time noise',np.std(err(r).reshape(8,-1),axis=1)); M=np.linalg.lstsq(b(r),Y.T,rcond=None)[0].T
print('y from time',np.std(y-aff(M)@np.linalg.lstsq(aff(M),y,rcond=None)[0]));print('y from modes and inputs',np.std(y-np.c_[aff(X),M]@np.linalg.lstsq(np.c_[aff(X),M],y,rcond=None)[0]))
ops=[]
for k,o in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 p=s[k]; fits=sorted((panel(p['u'],p[o],model) for model in ['sin','tanh','sq','cube']),key=lambda p:p[0]); e,r,f,c=fits[0];print(o,[(round(g[0],6),g[1]) for g in fits], '\n',c.round(3));ops.append(f)
F={'raw':X,'inverse':Z,'forward':W}
for name,P in list(F.items()):
 B,C,D=[f(P) for f in ops]; F.update({name+nm:A for nm,A in [('B',B),('C',C),('D',D),('BC',np.c_[B,C]),('CD',np.c_[C,D]),('BD',np.c_[B,D]),('BCD',np.c_[B,C,D]),('xB',np.c_[P,B]),('xC',np.c_[P,C]),('xD',np.c_[P,D]),('xBCD',np.c_[P,B,C,D]),('CB',ops[0](C)),('BC1',ops[1](B))]})
idx=[(np.arange(52),np.arange(52,72))]+[(j[:52],j[52:]) for j in [np.random.default_rng(seed).permutation(n) for seed in range(4)]]
T=np.c_[y,Y]; results=[]
for name,A in F.items():
 L=aff(A); Q=np.c_[L,np.column_stack([A[:,i]*A[:,j] for i,j in itertools.combinations_with_replacement(range(A.shape[1]),2)])]; A=A/np.sqrt(A.shape[1]/4); d=cdist(A,A,'sqeuclidean'); kernels={'lin':L@L.T,'quad':Q@Q.T,'rbf':np.exp(-.4*d),'rbf+lin':np.exp(-.25*d)+L@L.T,'rbf+quad':np.exp(-.25*d)+Q@Q.T}
 for k,K in kernels.items():
  for reg in [1e-5,.001,.01,.1]:
   es=[]
   for i,j in idx:
    e=abs(K[np.ix_(j,i)]@solve(K[np.ix_(i,i)]+reg*np.eye(len(i)),T[i],assume_a='pos')-T[j]);es.append([e[:,0].mean(),e[:,1:].mean()]);
   es=np.asarray(es);results.append((name,k,reg,*es[0],*es.mean(0),*es.max(0)))
for col in (5,6):
 print('best',col);print('\n'.join(str(q[:3]+tuple(round(f,5) for f in q[3:])) for q in sorted(results,key=lambda k:k[col])[:15]))
