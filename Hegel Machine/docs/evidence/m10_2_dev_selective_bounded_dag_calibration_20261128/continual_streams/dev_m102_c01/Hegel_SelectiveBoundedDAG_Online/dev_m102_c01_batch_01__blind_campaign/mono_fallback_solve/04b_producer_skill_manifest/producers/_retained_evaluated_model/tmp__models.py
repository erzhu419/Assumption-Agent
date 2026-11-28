exec(open('/tmp/explore.py').read().split("a=s['panel_a']")[0])
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from scipy.linalg import solve
from itertools import combinations_with_replacement
from scipy.special import expit
p=s['panel_a']; u=p['u']; v=p['v']; A=fit(design(u),v); keep=np.linalg.norm(design(u)@A-v,axis=1)<3; A=fit(design(u[keep]),v[keep]); d=np.diag(A[1:]); a=A[0]; p=s['panel_b']; B=fit(design(p['u']),p['v']); p=s['panel_c']; u=p['u']; v=p['v1'];
for kind in ['sep','post','shift','sepquad']:
 if kind=='post':
  def calc(th,u): return np.tanh(design(u)@th[:20].reshape(5,4))*th[20:24]+th[24:28]
  th=np.r_[fit(design(u),v).ravel(),np.ones(4),np.zeros(4)]; g=lambda h: (calc(h,u)-v).ravel()
 else:
  def f(h,u):
   f0=np.tanh(u*np.exp(h[:4])+ (h[4:] if kind=='shift' else 0)); return np.c_[design(f0),u*u] if kind=='sepquad' else design(f0)
  th=np.r_[np.ones(4)*.5,np.zeros(4)] if kind=='shift' else np.ones(4)*.5
  g=lambda h: (f(h,u)@fit(f(h,u),v)-v).ravel()
 o=least_squares(g,th,max_nfev=500); print('C1',kind,np.std(o.fun),o.x)
 if kind=='shift': W=fit(f(o.x,u),v); h=o.x.copy(); C1=lambda q: design(np.tanh(q*np.exp(h[:4])+h[4:]))@W
v=p['v2']; f2=lambda q,k:np.c_[design(q),q+.0] if False else np.c_[design(q),q*q/(1+k*q*q)]; o=minimize_scalar(lambda k:np.std(f2(u,k)@fit(f2(u,k),v)-v),bounds=(.001,.6),method='bounded'); k=o.x; W2=fit(f2(u,k),v); C2=lambda q:f2(q,k)@W2
z=(x-a)/d; b=lambda q:design(q)@B
base={'X':x,'A-':z,'A+':x*d+a}; base.update({'B'+key:b(q) for key,q in list(base.items())}); base.update({'C'+str(i)+key:f(q) for key,q in list(base.items()) for i,f in [(1,C1),(2,C2)]}); print('C2 k',k)
Y=np.c_[y,T]; U,S,V=np.linalg.svd(T,full_matrices=False); Y=np.c_[y,U[:,0]*S[0],U[:,1]*S[1]]; splits=[(np.arange(52),np.arange(52,72))]+[(j[:52],j[52:]) for j in [np.random.default_rng(i).permutation(n) for i in range(8)]]
def check(K, reg):
 out=[]
 for j,l in splits:
  P=K[np.ix_(l,j)]@solve(K[np.ix_(j,j)]+reg*np.eye(len(j)),Y[j],assume_a='pos');out.append(np.mean(abs(P-Y[l]),axis=0))
 return np.array(out)
def poly(q,m): return np.column_stack([np.prod(q[:,i],axis=1) for d in range(1,m+1) for i in combinations_with_replacement(range(q.shape[1]),d)])
rows=[]
for name,q in base.items():
 for deg in range(1,5):
  for mode in range(2):
   for rem in ([0] if mode==0 else [.25,1,4]):
    for reg in [1e-6,.001,.025,.15]:
     D=poly(q,deg); K=1+D@D.T
     if mode: K+=2*np.exp(-cdist(q,q,'sqeuclidean')*rem)
     out=check(K,reg); rows.append((f'{name}/{deg}/{mode}/{rem}/{reg}',out[0],out.mean(0),out.max(0)))
for j in [0,1]:
 for rank in [1,2,3]:
  print('ranking',j,rank)
  for row in sorted(rows,key=lambda r:r[rank][j])[:9]: print(row)
# trajectory time laws
for typ in range(4):
 def curve(k):
  p=-np.expm1(-k*t); return [p,(t/(t+k)),p*(1+k*t),p/(1+k*t)][typ]
 def f(k):
  c=curve(k); return np.linalg.norm(T- T@c[:,None]/(c@c)*c)
 o=minimize_scalar(f,bounds=(.01,4),method='bounded'); print('mode',typ,o.x,o.fun)
