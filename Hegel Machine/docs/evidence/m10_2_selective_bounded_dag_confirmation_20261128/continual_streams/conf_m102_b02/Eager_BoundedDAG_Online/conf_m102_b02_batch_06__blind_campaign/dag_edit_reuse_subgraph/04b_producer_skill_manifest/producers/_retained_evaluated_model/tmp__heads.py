exec(open('/tmp/diagnose.py').read().split("for k in ['panel_a'")[0])
from scipy.special import erf
from scipy.linalg import solve
p=s['panel_c'];u=p['u']; v=p['v2'];
forms=[lambda x,q:x+q[0]*x*x/(1+q[1]*x*x),lambda x,q:x+q[0]*np.log1p(q[1]*x*x),lambda x,q:x+q[0]*np.tanh(q[1]*x*x),lambda x,q:x+q[0]*(np.sqrt(1+q[1]*x*x)-1),lambda x,q:x+q[0]*(1-np.exp(-q[1]*x*x))]
rr=[]
for fn in forms:
 def design(x,q):return D(fn(x,q))
 q=least_squares(lambda q:(design(u,q)@old.fit(design(u,q),v)-v).ravel(),[.55,.2],bounds=([0,.001],[10,5]),max_nfev=400); print('C2',q.x,np.std(q.fun));rr.append((q.cost,fn,q.x))
_,fn,q=min(rr,key=lambda r:r[0]); c= D(fn(z,q))@old.fit(D(fn(u,q)),v)
c1=D(z*np.roll(z,-1,axis=1))@old.fit(D(u*np.roll(u,-1,axis=1)),p['v1'])
print('y quantiles',np.quantile(T[:,0],np.linspace(0,1,11))); print('trajectory endpoint', np.quantile(t['traj'][:,-1],np.linspace(0,1,11)))
np.savez('/tmp/feat.npz',z=z,b=b,c=c,c1=c)
# Profile one nonlinear index plus a low-dimensional linear background. Optimize on training rows only.
inputs={'z':z,'b':b,'c':c,'c1':c1,'zb':np.c_[z,b],'bc':np.c_[b,c],'zc':np.c_[z,c],'zbc':np.c_[z,b,c],'zbc1':np.c_[z,b,c1],'bc1':np.c_[b,c1],'all':np.c_[z,b,c,c1]}
Y=np.c_[T[:,0],t['traj'][:,-1]]
acts={'tanh':np.tanh,'sin':np.sin,'erf':erf,'rational':lambda x:x/np.sqrt(1+x*x),'quadratic':lambda x:x*x,'cubic':lambda x:x**3,'radial':lambda x:np.exp(-x*x)}
res=[]
for name,X in inputs.items():
 X=D((X-X[:n].mean(0))/X[:n].std(0)); d=X.shape[1];
 for typ,g in acts.items():
  for back in ['0','z','self','bc']:
   B=np.ones((len(X),1)) if back=='0' else D(inputs[name if back=='self' else back]); scores=[]; rms=[]
   for i,j in sl[:2]:
    vals=[]; rng=np.random.default_rng(23)
    for h in range(2):
     w0=old.fit(X[i],Y[i,h]); w0[0]=0;w0/=max(.01,np.linalg.norm(w0));
     def f(p,ix):return np.c_[B[ix],g(X[ix]@p)]
     def fun(p):
      A=f(p,i); return A@old.fit(A,Y[i,h])-Y[i,h]
     fits=[least_squares(fun,a,max_nfev=65) for a in [w0,2*w0,rng.normal(0,.4,d)]]; fit=min(fits,key=lambda r:r.cost); vals.append(np.abs(f(fit.x,j)@old.fit(f(fit.x,i),Y[i,h])-Y[j,h]).mean()); rms.append(np.std(fit.fun))
    scores.append(vals)
   res.append((np.array(scores),np.array(rms).reshape(-1,2),name,typ,back))
for h in range(2):
 for r in sorted(res,key=lambda r:r[0][:,h].mean())[:16]: print(h,r)
