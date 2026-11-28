from graph import *
from scipy.linalg import solve
from scipy.optimize import minimize_scalar
p=s['panel_b'];u,v=p['u'],p['v']; X=poly(u,1); ops=[]
funs={'tanh':np.tanh,'atan':np.arctan,'sin':np.sin,'rat':lambda x:x/(1+x*x),'erf':__import__('scipy.special',fromlist=['erf']).erf,'softsign':lambda x:x/(1+abs(x)),'sat':lambda x:x/np.sqrt(1+x*x)}
for name,f in funs.items():
 for power in [0,1,3,5]:
  for kind in [0,1]:
   def Z(a,q):
    g=f(q[0]*a)
    h=a**power if power else np.zeros_like(a)
    return np.c_[np.ones(len(a)),g+q[-1]*h] if kind else np.c_[np.ones(len(a)),g,h]
   def resid(q):
    H=Z(u,q); return (H@fit(H,v)-v).ravel()
   for start in [.7,1.3,2.5]:
    opt=least_squares(resid,[start,.3] if kind else [start],bounds=([.1,-20] if kind else [.1],[5,20] if kind else [5]),max_nfev=70)
    m=fit(Z(u,opt.x),v); ops.append((np.std(opt.fun),(name,power,kind,opt.x),lambda a,Z=Z,f=f,power=power,kind=kind,m=m,q=opt.x: (np.c_[np.ones(len(a)),f(q[0]*a)+q[-1]*(a**power if power else 0)] if kind else np.c_[np.ones(len(a)),f(q[0]*a),a**power if power else np.zeros_like(a)])@m,m))
print('B',[(round(e,7),n) for e,n,_,_ in sorted(ops,key=lambda a:a[0])[:20]]); print('best B',min(ops,key=lambda a:a[0])[-1]);
a,ai=Amap(s['panel_a']); x=ai(s['train']['x']); y=np.c_[s['train']['y'],s['train']['traj']]; rng=np.random.default_rng(32); folds=[(np.arange(52),np.arange(52,72))]+[(p[:52],p[52:]) for p in [rng.permutation(72) for _ in range(3)]]
def eval(F,reg):
 E=[]
 for i,j in folds:
  W=F[i]; E.append(np.abs(F[j]@solve(W.T@W+reg*np.eye(W.shape[1]),W.T@y[i],assume_a='pos')-y[j]).mean(0));
 return np.array(E)
# Readout mechanism candidates: additive saturated terms and products, with compact cubic and rational transformations.
R=[]; i,j=np.triu_indices(4); ci,cj=np.triu_indices(4,1)
for name,f in funs.items():
 for q in np.linspace(.6,3.2,27):
  z=f(q*x); P=np.c_[np.ones(len(x)),x,z]; Q=np.c_[P,z[:,i]*z[:,j]]; linear=np.c_[np.ones(len(x)),z]; XZ=(x[:,:,None]*z[:,None,:]).reshape(-1,16); X3=(z[:,:,None]*z[:,i,None].transpose(0,2,1)*z[:,j,None].transpose(0,2,1)).reshape(-1,40); unique=np.unique(np.array([sorted([p,a,b]) for p in range(4) for a,b in zip(i,j)]),axis=0); X3=np.prod(z[:,unique],axis=2)
  for typ,F in enumerate([Q,np.c_[Q,XZ],poly(z,3),np.c_[Q,x**2,x**3],np.c_[Q,X3,x],np.c_[Q,x**2,z**3],np.c_[Q,(z*x),x**2],np.c_[Q,z[:,ci]*z[:,cj]*(z[:,ci]+z[:,cj]),z**3],np.c_[Q,z[:,ci]*z[:,cj]*(z[:,ci]-z[:,cj]),z**3],np.c_[Q,z**3],poly(z,2),np.c_[poly(x,2),poly(z,2)[:,1:]],np.c_[Q,z[:,ci]*z[:,cj]*(z[:,ci]**2+z[:,cj]**2),z**4],np.c_[Q,np.prod(z,axis=1),z**3]]):
   for reg in [.0001,.003,.015]:
    e=eval(F,reg); R.append((e[:,0].mean(),e[0,0],e[0,1:].mean(),e[:,0].max(),name,q,typ,reg))
print('dict AVG',sorted(R)[:14]);print('dict FIX',sorted(R,key=lambda a:a[1])[:14]);
for e,l,b,_ in sorted(ops,key=lambda a:a[0])[:3]:
 z=b(x); print('B coeff',l,np.corrcoef(np.c_[z,y[:,0],y[:,-1]].T)[:4,4:]);
 for F in [poly(z,2),np.c_[poly(x,2),poly(z,2)[:,1:]],np.c_[poly(z,2),np.tanh(z),np.tanh(z)**2]]:print(eval(F,.001)[:,[0,8]])
