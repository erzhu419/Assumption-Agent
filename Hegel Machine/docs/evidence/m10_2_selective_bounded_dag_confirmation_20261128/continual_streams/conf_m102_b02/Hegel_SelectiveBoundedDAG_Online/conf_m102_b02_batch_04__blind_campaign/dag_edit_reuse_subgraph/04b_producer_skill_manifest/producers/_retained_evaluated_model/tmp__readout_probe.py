from panel_probe import *
from itertools import combinations, product
from scipy.optimize import least_squares
rng=np.random.default_rng(7)
x=np.r_[s['train']['x'],s['test']['x']].copy(); u,v=(s['panel_a'][k] for k in ('u','v'))
for j in range(4):
 a=d(u[:,j]); keep=np.ones(len(u),bool)
 for _ in range(8):
  w=fit(a[keep],v[keep,j]); r=abs(a@w-v[:,j]); keep=r<max(.0006,4*np.median(r))
 x[:,j]=(x[:,j]-w[0])/w[1]
u,v=(s['panel_b'][k] for k in ('u','v')); bo=bfit(u,v)
for score,key,mode,p,w,e in bo[:14]:print('B',score,key,mode,np.round(p,5),e,flush=True)
z=bf(x,bo[0][3],bo[0][1],bo[0][2])@bo[0][4]
pairs=[(0,1),(0,3),(1,2),(2,3)]
def p1(q):return d(np.column_stack([q[:,i]*q[:,j] for i,j in pairs]))
u=s['panel_c']['u']; c=s['panel_c']; w=fit(p1(u),c['v1']); print('C1',err(p1(u),c['v1']))
def C(q):return p1(q)@w
def v2(q,p):
 k=np.exp(p[0]); power=p[1]; t=np.log1p(k*q*q); h=t if abs(power)<1e-7 else np.expm1(power*t)/power
 return d(q+p[2]*h/k)
def r(p):
 a=v2(u,p);return (a@fit(a,c['v2'])-c['v2']).ravel()
p=least_squares(r,[-.1,.5,.5],bounds=([-5,-3,-10],[3,3,10]),max_nfev=500).x
print('C2',p,err(v2(u,p),c['v2']),flush=True)
w2=fit(v2(u,p),c['v2'])
def E(q):return v2(q,p)@w2
A={'x':x,'b':z,'cx':C(x),'cb':C(z),'ex':E(x),'eb':E(z),'ecx':E(C(x)),'ecb':E(C(z)),'cex':C(E(x)),'ceb':C(E(z))}
y=np.c_[s['train']['y'],s['train']['traj']]; n=len(y)
folds=np.array_split(rng.permutation(n),6)
def cv(f,alpha=1e-6):
 f=f[:n]; out=[]
 for te in folds:
  tr=np.setdiff1d(np.arange(n),te); scale=np.maximum(f[tr].std(0),1e-9); scale[0]=1; a=f[tr]/scale; b=f[te]/scale
  pred=b@np.linalg.solve(a.T@a+alpha*np.eye(a.shape[1]),a.T@y[tr]); e=abs(pred-y[te]); out.append([e[:,0].mean(),e[:,1:].mean()])
 return np.mean(out,axis=0),np.max(out,axis=0)
def prod(a,b):return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
out=[]
for ka,kb in combinations(A,2):
 a,b=A[ka],A[kb]; pr=prod(a,b)
 for mode,f in [('lin',d(a,b)),('cross',d(a,b,pr)),('squares',d(a,b,pr,a*a,b*b)),('cross+x+b',d(x,z,a,b,pr))]:
  for reg in (1e-5,.01,.25):
   av,mx=cv(f,reg);out.append((sum(av),ka,kb,mode,reg,np.round(av,5),np.round(mx,5)))
for ka,a in A.items():
 for deg in (1,2,3):
  f=np.column_stack([np.prod(a[:,cs],axis=1) for i in range(deg+1) for cs in combinations_with_replacement(range(4),i)]); av,mx=cv(f,.005);out.append((sum(av),ka,'-',deg,.005,np.round(av,5),np.round(mx,5)))
for o in sorted(out,key=lambda a:a[0])[:38]:print(o)
np.savez('/tmp/campaign_coords.npz',**A)
