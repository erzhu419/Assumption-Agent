exec(open('/tmp/probe.py').read().split("print('times'")[0])
from scipy.spatial.distance import cdist
from scipy.linalg import solve
from scipy.optimize import least_squares
from itertools import combinations
z=(x-a[0])/a[1]; w=A(z)@b; p=s['panel_c']; u=p['u']
def phi(q,mode):
 d=q.shape[1]; pairs=list(combinations(range(d),2)); cross=np.column_stack([q[:,i]*q[:,j] for i,j in pairs]); cube21=np.column_stack([q[:,i]**2*q[:,j] for i in range(d) for j in range(d) if i!=j]); cube111=np.column_stack([np.prod(q[:,ij],axis=1) for ij in combinations(range(d),3)])
 return {'l':A(q),'s':q*q,'p':cross,'c':q**3,'m':cube21,'t':cube111,'4':q**4,'5':q**5,'6':q**6}
def basis(q,mode):
 tab=phi(q,mode); return np.column_stack([tab[k] for k in mode])
def loo(f,y):
 m=np.linalg.pinv(f); e=y-f@m@y; l=np.maximum(1-np.sum(f*m.T,axis=1),1e-8); return np.mean(abs(e/l[:,None]),axis=0), np.sqrt(np.mean(e**2,axis=0)), m@y
for k in ('v1','v2'):
 v=p[k]; print('C',k)
 for mode in ['l','ls','lsc','lsc4','lsc45','lsc456','lc','lp','lsp','lpc','lspc','lt','lst','lstm','lscm']:
  f=basis(u,mode)
  if f.shape[1]>=len(f): continue
  score,rm,c=loo(f,v); print(mode,np.round(score,5),np.round(rm,5));
  if mode in ('lsc','lspc','lsc45'): print(' coeff',np.round(c.T,4))
 def smooth(kval,kind):
  if kind=='cos': return (np.cos(u*kval)-1)/kval**2
  if kind=='rational': return u*u/(1+kval*u*u)
  if kind=='sin': return (np.sin(u*kval)-u*kval)/kval**3
  return np.tanh(u*kval)
 for kind in ['cos','rational','sin','tanh']:
  for add in ['l','ls','lc']:
   vals=[(.08,0)]
   vals=[(loo(np.c_[basis(u,add),smooth(h,kind)],v)[0].mean(),h) for h in np.geomspace(.04,4,50)]
   print(kind,add,min(vals))
print('regularized y'); rng=np.random.default_rng(42); splits=[(np.arange(52),np.arange(52,72))]+[(p[:52],p[52:]) for p in (rng.permutation(72) for _ in range(20))];
def eval_kernel(k):
 vals=[]
 for i,j in splits:
  f=A(z); qi,ri=np.linalg.qr(f[i]); m=np.eye(len(i))-qi@qi.T
  ka=k[np.ix_(i,i)]; kb=k[np.ix_(j,i)]
  outs=[]
  for lam in (1e-5,.0001,.001,.01,.05,.2,1):
   al=solve(m@ka@m+np.eye(len(i))*lam,m@y[i],assume_a='pos'); pred=f[j]@fit(f[i],y[i]-ka@al)+kb@al; outs.append(np.mean(abs(pred-y[j])))
  vals.append(outs)
 return np.array(vals)
cands={}
for name,q in [('z',z),('w',w)]:
 for mode in ('lspcmt','lspcmt45','lscmt','lpct','lspmt','lspc','lsmt','lstmt','lspmt45'):
  f=basis(q,mode); cands[name+mode]=f@f.T
 for d in (.4,.8,1.2,1.8,3):
  R=cdist(q,q)/d; cands[name+str(d)+'rbf']=np.exp(-.5*R**2); cands[name+str(d)+'matern']=(1+R+R**2/3)*np.exp(-R)
 for d in (3,4,5,6): cands[name+str(d)+'poly']=(1+q@q.T)**d
out=[]
for name,k in cands.items():
 val=eval_kernel(k); col=np.argmin(val[1:].mean(0)); out.append((val[0,col],val[1:,col].mean(),val[1:,col].max(),name,col,val[0].min()))
for r in sorted(out,key=lambda r:r[1])[:20]: print(r)
