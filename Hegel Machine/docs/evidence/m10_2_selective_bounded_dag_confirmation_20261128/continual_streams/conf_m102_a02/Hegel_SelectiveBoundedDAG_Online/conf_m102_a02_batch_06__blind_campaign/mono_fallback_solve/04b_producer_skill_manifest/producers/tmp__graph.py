exec(open('/tmp/probe.py').read().split('np.set_printoptions')[0])
from scipy.optimize import least_squares
from sklearn.preprocessing import PolynomialFeatures
from scipy.linalg import lstsq
ls=lambda f,v: lstsq(f,v)[0]
np.set_printoptions(precision=4,suppress=True)
W={'q':q}
models={}
for pn,key in [('b','v'),('c','v1'),('c','v2')]:
 P=s['panel_'+pn]; U,V=P['u'],P[key]
 families={'tanh':(lambda a,k:np.tanh(k*a),1.25),'rational':(lambda a,k:a/(1+k*a*a),1.25),'cos':(lambda a,k:np.cos(k*a),1.25),'sin':(lambda a,k:np.sin(k*a),1.25),'softabs':(lambda a,k:np.sqrt(1+k*a*a),1.25),'exp':(lambda a,k:np.exp(-k*a*a),1.25),'cube':(lambda a,k:a**3,1.25)}
 opts=[]
 for nm,(h,k0) in families.items():
  for affine in [False,True]:
   def f(a,k,h=h,affine=affine):return np.c_[np.ones(len(a)),a,h(a,k)] if affine else np.c_[np.ones(len(a)),h(a,k)]
   def r(k):return (f(U,k)@ls(f(U,k),V)-V).ravel()
   k=least_squares(r,[k0],bounds=(.02,5),max_nfev=100).x; F=f(U,k); B=ls(F,V); err=F@B-V; loo=err/(1-np.einsum('ij,ji->i',F,np.linalg.pinv(F)))[:,None]; opts.append((np.std(loo),np.std(err),nm,affine,k,B,f))
 opts.sort(key=lambda p:p[0]); name='b' if pn=='b' else key
 print(name,[(p[:5]) for p in opts[:5]]); _,_,_,_,k,B,f=opts[0]; print(B); models[name]=lambda a,k=k,B=B,f=f:f(a,k)@B
for n,model in models.items():W[n]=model(q)
for n,model in models.items():
 for p in list(models):
  if n!=p: W[p+n]=model(W[p])
A=np.c_[y,z@np.linalg.svd(z[:52],full_matrices=False)[2][0]]; A[:,1]/=2

def check(F):
 B=ls(F[:52],A[:52]); e=F@B-A
 return np.mean(abs(e[52:]),axis=0),np.mean(abs(e[:52]),axis=0)
R=[]
for n,a in W.items():
 for p,b in [('none',a[:,:0]),*W.items()]:
  if p==n:continue
  comb=np.c_[a,b];
  for d in [1,2]:
   r0,r1=check(PolynomialFeatures(d).fit_transform(comb)); R.append((n,p,d,*r0,*r1))
for i in [3,4]:
 print('best',i)
 for r in sorted(R,key=lambda r:r[i])[:14]:print(r)
np.savez('/tmp/graph.npz',**W)
