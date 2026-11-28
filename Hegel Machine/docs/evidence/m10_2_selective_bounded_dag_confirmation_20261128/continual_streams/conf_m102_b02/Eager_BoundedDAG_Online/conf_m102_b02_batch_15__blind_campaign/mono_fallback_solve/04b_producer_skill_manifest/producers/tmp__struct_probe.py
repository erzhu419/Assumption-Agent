import numpy as np
from scipy.optimize import least_squares
from itertools import combinations
p=np.load('/tmp/design.npz'); z,b,c,h=[p[k][:72] for k in 'zbch']; y=p['y']; M=p['m']; T=p['T']; f=p['f']; Y=np.c_[y,M]; n=52
D=lambda *a:np.column_stack((np.ones(len(a[0])),*a)); fit=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-9)[0]
def report(nm,F,j):
 v=fit(F[:n],Y[:n,j]); e=F@v-Y[:,j]; print(nm,j,np.round([np.mean(abs(e[:n])),np.mean(abs(e[n:])),np.corrcoef(F@v,Y[:,j])[0,1]],4)); return e,v
print('corr',np.corrcoef(Y.T)); print('summary',__import__('pickle').load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']['train']['summary'])
u=z[:,2]+z[:,3]; v=z[:,0]-z[:,1];
for q in [.5,1,1.25,1.7,2.15]:
 report(('two tanh',q),D(np.tanh(q*u),np.tanh(q*v)),1)
for A, nm in [(b,'b'),(z,'z'),(h,'h')]:
 Q=np.array([A[:,i]*A[:,j] for i,j in combinations(range(4),2)]).T
 for j in range(3):
  e,w=report(nm+' quad',D(A,A*A,Q),j);print('w',np.round(w,2))
# variable projection of two-index heads with shared low-dimensional coordinate groups
rng=np.random.default_rng(88)
ops={'t':np.tanh,'s':np.sin,'c':np.cos,'r':lambda x:x/(1+x*x),'q':lambda x:x*x,'e':lambda x:np.exp(-x*x)}
for j,keys in [(1,'z'),(2,'b'),(0,'z'),(0,'h'),(0,'zh'),(1,'zh'),(2,'zb')]:
 X=np.c_[*[dict(z=z,b=b,c=c,h=h)[k] for k in keys]]; dn=X.shape[1]; ans=[]
 for mode in ['t','tt','ts','tc','tr','tq','ss','sc','qq','tct','ttt','sct']:
  best=None
  for it in range(11 if dn==4 else 5):
   def F(w):
    g=D(X)@w.reshape(-1,dn+1).T
    return D(X,*[ops[a](g[:,i]) for i,a in enumerate(mode)])
   def E(w):
    g=F(w)[:n];return g@fit(g,Y[:n,j])-Y[:n,j]
   w=rng.normal(0,.9,(len(mode),dn+1));
   if it==0 and keys=='z':w[:min(2,len(w))]=[[0,0,0,1,1],[0,1,-1,0,0]][:len(w)]
   res=least_squares(E,w.ravel(),max_nfev=110); score=np.mean(res.fun**2)
   if best is None or score<best[0]:best=score,res.x
  w=best[1]; g=F(w); coef=fit(g[:n],Y[:n,j]); ans.append((np.sqrt(best[0]),np.mean(abs(g[n:]@coef-Y[n:,j])),mode,np.round(w,2),np.round(coef,2)))
 print('IDX',j,keys,sorted(ans,key=lambda x:x[0])[:4], 'bestval',sorted(ans,key=lambda x:x[1])[:2],flush=True)
