exec(open('/tmp/readout_probe.py').read().split('nodes=')[0])
p=s['panel_a']; U,V=p['u'],p['v']; e=np.linalg.norm(D(U)@L(D(U),V)-V,axis=1); ii=np.argsort(e)[:45]; u,v=U[ii],V[ii]
from scipy.optimize import least_squares
from itertools import product
pairs=list(combinations_with_replacement(range(4),2))
for j in range(4):
 q=np.c_[u,p2(u),u**3]; design=D(q); beta=L(design,v[:,j]); print('A',j,'rmse',np.std(design@beta-v[:,j]),'coef',list(zip(['int']+[f'x{i}' for i in range(4)]+[f'x{i}x{k}' for i,k in pairs]+[f'x{i}^3' for i in range(4)],np.round(beta,5))))
j=lambda a:D(a,a*a/np.sqrt(1+.39*a*a))
def structured(w):return D(w,w**3,w*np.roll(w,-1,axis=1))
AA=L(structured(u),v); print('struct',np.round(AA,5),'rmse',np.std(structured(u)@AA-v));
def inv(fun,w):return np.array([least_squares(lambda t:fun(t[None])[0]-r,np.clip(r,-1.6,1.6),max_nfev=50).x for r in w])
cal={'structured':lambda w:structured(w)@AA,'full':a}; F={}
for name,op in cal.items():
 aa=inv(op,x); print(name,'latent range',aa.min(0),aa.max(0)); F[name]=aa
 if name=='structured':
  at=inv(op,s['test']['x']); print('test range',at.min(0),at.max(0))
F['raw']=x
G={}
for name,f in F.items():
 for route,w in [('id',f),('B',b(f)),('BT',(f-B[0])@np.linalg.inv(B[1:]))]:
  g,k=c(w),d(w)
  base=D(f,w,g,k); G[name+'/'+route+'/linear']=base
  G[name+'/'+route+'/dot']=D(f,w,g,k,(f*g).sum(1),(w*g).sum(1),(f*k).sum(1),(w*k).sum(1),(g*k).sum(1));
  G[name+'/'+route+'/quad']=D(f,p2(f),g,k)
  for gn,ga in [('f',f),('w',w),('g',g),('k',k)]:
   for hn,ha in [('g',g),('k',k)]:G[name+'/'+route+'/'+gn+hn]=np.c_[base,cross(ga,ha)]
  G[name+'/'+route+'/all']=np.c_[base,p2(f),cross(f,g),cross(f,k),cross(g,k)]
  G[name+'/'+route+'/wonly']=D(w,g,k,cross(w,g),cross(w,k),cross(g,k))
ix=np.arange(72); rng=np.random.default_rng(18); fs=[(ix[:52],ix[52:]),(ix[:56],ix[56:])]+[(q[:52],q[52:]) for q in [rng.permutation(72) for _ in range(5)]]
R=[]
for name,g in G.items():
 for lam in (.000001,.001,.03,.3):
  es=[]
  for tr,te in fs:
   T=g[tr]; Q=np.eye(g.shape[1])*lam;Q[0,0]=0; P=g[te]@np.linalg.solve(T.T@T+Q,T.T@Y[tr]); er=abs(Y[te]-P); es.append([er[:,0].mean(),er[:,1:].mean()])
  es=np.array(es);R.append((name,lam,*es[0],*es[1],*es.max(0),*es.mean(0)))
for j in (2,3,6,7):
 print('SORT',j); print('\n'.join(str((r[0],r[1],*np.round(r[2:],5))) for r in sorted(R,key=lambda r:r[j])[:6]))
