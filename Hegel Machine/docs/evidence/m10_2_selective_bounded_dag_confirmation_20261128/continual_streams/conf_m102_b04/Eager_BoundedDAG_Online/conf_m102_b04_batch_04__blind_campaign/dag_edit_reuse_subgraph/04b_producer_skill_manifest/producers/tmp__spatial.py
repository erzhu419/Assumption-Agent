exec(open('/tmp/probe_dag.py').read().split("print('times'")[0])
from scipy.spatial.distance import cdist
from scipy.linalg import solve
poly=lambda q,d:np.column_stack([np.ones(len(q))]+[np.prod(q[:,j],1) for i in range(1,d+1) for j in comb(range(q.shape[1]),i)])
p=s['panel_a']; U=a(p['u']); j=np.arange(len(U));
for _ in range(12):
 W=fit(p['u'][j],p['v'][j]); j=np.argsort(np.linalg.norm(U@W-p['v'],axis=1))[:int(.82*len(U))]
xc=(x-W[0])/W[1:].diagonal(); Y=np.c_[y,z]; ss=[(np.arange(52),np.arange(52,n))]+[(np.setdiff1d(np.arange(n),j),j) for j in np.array_split(np.random.default_rng(11).permutation(n),4)]
coords={'raw':x,'cal':xc};
for pn in ['panel_b','panel_c']:
 p=s[pn]; u=p['u']; print(pn,len(u)); ii=np.arange(int(.8*len(u))); jj=np.arange(len(ii),len(u))
 for vn in sorted(set(p)-{'u'}):
  v=p[vn]
  for d in [1,2,3]:
   G=poly(u,d); C=np.linalg.lstsq(G[ii],v[ii],rcond=None)[0]; print(vn,d,'rank',np.linalg.matrix_rank(G[ii]),'train/test',np.sqrt(np.mean((G[ii]@C-v[ii])**2)),np.sqrt(np.mean((G[jj]@C-v[jj])**2))); C=np.linalg.lstsq(G,v,rcond=None)[0];
   if d==2: print('quadratic coefficients',C)
   for k,q in [('x',x),('c',xc)]: coords[f'{vn}{d}{k}']=poly(q,d)@C

def metrics(K,l):
 ms=[]
 for i,j in ss:
  P=K[np.ix_(j,i)]@solve(K[np.ix_(i,i)]+l*np.eye(len(i)),Y[i],assume_a='pos'); E=abs(P-Y[j]); ms.append([E[:,0].mean(),E[:,1:].mean()])
 return np.array(ms)
res=[]
for key,Q in coords.items():
 for s0 in [0,.09,.4,1]:
  q=Q/np.sqrt(1+s0*Q**2); d2=cdist(q,q,'sqeuclidean'); D=q@q.T
  for name,K in [(f'rbf{g}',1+np.exp(-g*d2)) for g in [.02,.06,.15,.3,.6,1.2]]+[(f'poly{d}',(1+D/4)**d) for d in [2,3,4,5,6]]:
   for l in [1e-7,1e-5,.0003,.003,.02]:
    M=metrics(K,l); res.append((M.mean(),M[0].max(),key,s0,name,l,M[0],M.max(0)))
for k in [0,1]:
 print('best by',k)
 for r in sorted(res,key=lambda r:r[k])[:15]: print(r)
