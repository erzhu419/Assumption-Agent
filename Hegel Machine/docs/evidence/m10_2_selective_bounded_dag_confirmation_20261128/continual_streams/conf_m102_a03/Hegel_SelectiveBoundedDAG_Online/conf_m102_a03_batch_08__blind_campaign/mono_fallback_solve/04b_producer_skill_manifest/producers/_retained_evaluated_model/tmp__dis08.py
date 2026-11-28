import numpy as np,pickle
from scipy.optimize import least_squares
from itertools import combinations_with_replacement as cwr
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u,v=[s['panel_b'][k] for k in ('u','v')]; d=np.load('/tmp/feat08.npz'); z,y,t=[d[k] for k in ('z','y','t')]; fit=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-10)[0]; aff=lambda a:np.c_[np.ones(len(a)),a]; rng=np.random.default_rng(12)
fs=[lambda x,h:x+x*(h[0]+h[1]*x*x)/(1+h[2]*x*x+h[3]*x**4),lambda x,h:x+h[0]*np.sin(h[1]*x)/(1+h[2]*x*x)+h[3]*x**3,lambda x,h:x+h[0]*np.tanh(h[1]*x)/(1+h[2]*x*x)+h[3]*x**3,lambda x,h:x+h[0]*np.arctan(h[1]*x)+h[3]*x*x*x/(1+h[2]*x*x),lambda x,h:x+h[0]*np.sin(h[1]*x)+h[3]*x/(1+h[2]*x*x),lambda x,h:x+h[0]*np.sin(h[1]*x)+h[3]*np.tanh(h[2]*x),lambda x,h:x+h[0]*np.tanh(h[1]*x)+h[3]*x/(1+h[2]*x*x),lambda x,h:x+h[0]*np.tanh(h[1]*x)+h[3]*np.arctan(h[2]*x),lambda x,h:x+h[0]*np.arctan(h[1]*x)/(1+h[2]*x*x)+h[3]*x**3,lambda x,h:x+h[0]*x*(1+h[3]*x*x)/(1+h[2]*x*x)**h[1]]
models=[]
for k,f in enumerate(fs):
 for mod in range(3):
  def m(x,h):
   q=f(x,h); return aff(q if mod==0 else q+h[4]*(x*x if mod==1 else np.roll(x,1,axis=1)*x))
  def r(h):
   A=m(u,h);return (A@fit(A,v)-v).ravel()
  O=min((least_squares(r,[a,b,c,d,0],bounds=([-8,0,0,-8,-5],[8,10,16,8,5]),max_nfev=180) for a,b,c,d in [(-.4,3.,1.,0.),(-.6,1.5,2.,0.),(-1.,2.,1.,.2),(-1.,1.,1.,.1),(.3,.5,2.,-.4),(.3,1.,1.,-1.)]),key=lambda o:o.cost)
  e=np.std(O.fun); h=O.x; print(k,mod,round(e,7),h.round(4)); models.append((e,k,mod,m(z,h),m(z,h)@fit(m(u,h),v)))
models.sort(key=lambda m:m[:3]); np.savez('/tmp/bfit08.npz',**{f'f{i}':m[3][:,1:] for i,m in enumerate(models[:12])},**{f'b{i}':m[4] for i,m in enumerate(models[:12])});
for name,X in [('B',d['B']),('z',z)]+[(str(m[1:3]),m[4]) for m in models[:5]]:
 Q=np.c_[aff(X),np.column_stack([X[:,i]*X[:,j] for i,j in cwr(range(4),2)])]; A=np.c_[Q,z]; Y=np.c_[y,t[:,0],t[:,-1]]; p=A@fit(A[:52],Y[:52]);print(name,'coarse',np.mean(abs(p[:52]-Y[:52]),0),np.mean(abs(p[52:]-Y[52:]),0));print('quadratic ranks y, last',*[np.linalg.eigvalsh(np.array([[fit(Q[:52],Y[:52])[5+list(cwr(range(4),2)).index(tuple(sorted((i,j)))),k]/(1 if i==j else 2) for j in range(4)]for i in range(4)])) for k in (0,2)])
