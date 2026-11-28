import numpy as np,pickle,itertools
from scipy.optimize import least_squares
from scipy.linalg import solve
from scipy.spatial.distance import cdist
np.set_printoptions(precision=4,suppress=True,linewidth=160)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; t=s['eval_times']; T=tr['traj']; Y=np.c_[tr['y'],T]; n=len(Y)
def lin(a,b):return np.linalg.lstsq(a,b,rcond=1e-9)[0]
def poly(x,k):return np.column_stack([np.ones(len(x))]+[np.prod(x[:,j],axis=1) for m in range(1,k+1) for j in itertools.combinations_with_replacement(range(x.shape[1]),m)])
u=s['panel_a']['u']; v=s['panel_a']['v']; f=poly(u,1); best=None
for p in np.random.default_rng(1).integers(len(u),size=(500,5)):
 c=lin(f[p],v[p]); r=np.linalg.norm(f@c-v,axis=1); score=np.median(r)
 if best is None or score<best[0]:best=(score,c)
c=best[1]
for _ in range(5):
 r=np.linalg.norm(f@c-v,axis=1); m=r<max(5*np.median(r),.0006);c=lin(f[m],v[m]);
print('A',c,'error',np.sort(r)); z=np.linalg.solve(c[1:].T,(tr['x']-c[0]).T).T
b=s['panel_b']; cb=lin(poly(b['u'],1),b['v']);print('B',cb); bx=poly(z,1)@cb
p=s['panel_c']; c1=lin(poly(p['u'],2),p['v1']);print('C1',c1); cx=poly(z,2)@c1
print('C2 quadratic',lin(poly(p['u'],2),p['v2']));
for name,f in [('cube',lambda x:np.c_[poly(x,2),x**3]),('sparse3',lambda x:np.c_[poly(x,2),np.prod(x[:,list(itertools.combinations(range(4),3))],axis=2)]),('trig',lambda x:np.c_[poly(x,2),np.sin(x),np.cos(x)])]:
 F=f(p['u']); C=lin(F,p['v2']); resid=(F@C-p['v2'])/(1-np.diag(F@np.linalg.pinv(F)))[:,None]; print(name,'loocv',np.mean(abs(resid),axis=0),C[-4:]);
Y0=Y.copy(); U,S,V=np.linalg.svd(T,full_matrices=False); Y=np.c_[Y[:,0],U[:,:3]*S[:3]]
folds=[(np.arange(52),np.arange(52,72))]+[(o[:52],o[52:]) for o in (np.random.default_rng(i).permutation(n) for i in range(5))]
def cv(K,l):
 e=[]
 for a,b in folds:
  P=K[np.ix_(b,a)]@solve(K[np.ix_(a,a)]+l*np.eye(len(a)),Y[a],assume_a='pos'); E=P-Y[b];e.append([np.mean(abs(E[:,0])),np.mean(abs(E[:,1:]@V[:3]))]);
 return np.array(e)
for name,X in [('z',z),('b',bx),('c1',cx),('c1(b)',poly(bx,2)@c1),('c1(x)',poly(tr['x'],2)@c1),('raw',tr['x']),('z+c1',np.c_[z,cx]),('b+c1(b)',np.c_[bx,poly(bx,2)@c1])]:
 print(name,'linear MAE',np.mean(abs(poly(X,1)@lin(poly(X,1),Y)-Y),0)); out=[]
 for tf in [lambda x:x,np.tanh,lambda x:x/(1+abs(x)),lambda x:np.c_[x,np.tanh(x)]]:
  X0=tf(X);X0=(X0-X0.mean(0))/np.maximum(.06,X0.std(0)); G=X0@X0.T/X0.shape[1];D=cdist(X0,X0,'sqeuclidean')/X0.shape[1]
  for j in [1,2,3]:
   for q,g in [(0,1),(.1,.25),(.1,1),(1,.25),(1,1)]:
    for l in [1e-6,1e-4,.002,.02]:
     e=cv((1+.4*G)**j+q*np.exp(-g*D),l);out.append((e[1:].mean(),j,q,g,l,e[0],e[1:].max(0)))
 print('best',sorted(out,key=lambda row:row[0])[:2])
