exec(open('/tmp/probe.py').read().split('for name,p')[0])
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
ls=a._ls; bias=a._bias
u,v=(s['panel_a'][k] for k in ('u','v')); rr=np.linalg.norm(v-u,axis=1); keep=rr<4*np.median(rr);u,v=u[keep],v[keep]
forms={'sin':np.sin,'cos':np.cos,'tanh':np.tanh,'atan':np.arctan,'soft':lambda w:w/(1+abs(w)),'exp':lambda w:np.expm1(-w*w),'rational':lambda w:w/(1+w*w)}
models=[]
for n1,g in forms.items():
 for n2,h in [(None,None),*forms.items()]:
  if n1==n2:continue
  def f(U,p,g=g,h=h):return bias(np.c_[U,g(p[0]*U)] if h is None else np.c_[U,g(p[0]*U),h(p[1]*U)])
  def err(p):
   F=f(u,p);return (F@ls(F,v)-v).ravel()
  r=least_squares(err,[1.5] if h is None else [1.5,1.5],bounds=(.1,7),max_nfev=120);models.append((np.mean(abs(r.fun)),n1,n2,r.x,f))
models.sort(key=lambda q:q[0]);print('A fits',[(m,n,l,p) for m,n,l,p,_ in models[:10]]); _,_,_,p,f=models[0]; pars=ls(f(u,p),v);print('A coeff',np.round(pars,5));
def coords(X):
 z=X.copy();e=np.eye(4)*1e-5
 for i in range(20):
  res=f(z,p)@pars-X
  if abs(res).max()<1e-10:break
  d=np.stack([(f(z+ej,p)@pars-f(z-ej,p)@pars)/(2e-5) for ej in e],axis=2)
  z-=np.clip(np.linalg.solve(d,res[...,None])[...,0],-.35,.35)
 print('inverse',abs(f(z,p)@pars-X).max(),z.min(),z.max());return z
z=coords(x); u=s['panel_c']['u']; W=[]
for key,F,p0 in [('v1',lambda U,p:bias(np.c_[np.tanh(p[0]*U),p[1]*U/(1+abs(p[1]*U))]),[1.53,1.5]),('v2',lambda U,p:bias(np.c_[U,np.expm1(-p[0]*U*U)]),[.29])]:
 v=s['panel_c'][key];r=least_squares(lambda p:(F(u,p)@ls(F(u,p),v)-v).ravel(),p0);B=ls(F(u,r.x),v);print(key,r.x,np.round(B,4));W.append(F(z,r.x)@B)
A,B=W;w=np.c_[A,B];t=s['eval_times'];
def stats(d):return np.round([d[:,0].mean(),d[:,1:].mean(),np.max(d[:,1:].mean(0))],6)
for name,F in [('basic',a._readout_features(w)),('allquad',bias(np.c_[w,*[w[:,i]*w[:,j] for i,j in combinations_with_replacement(range(8),2)]])),('AB',bias(np.c_[w,*[A[:,i]*B[:,j] for i in range(4) for j in range(4)]])),('ABdiag',bias(np.c_[w,A*B])),('full',bias(np.c_[z,w,A*B,*[A[:,i]*B[:,j] for i in range(4) for j in range(4) if i!=j]]))]:
 print('FEATURE',name,'trainres',stats(abs(F@ls(F,Y)-Y)))
 for hi in [np.arange(52,72),np.arange(20),np.arange(72)[::3],np.arange(72)[1::3],np.argsort(abs(z).max(1))[-14:]]:
  ii=np.setdiff1d(np.arange(72),hi); print(stats(abs(F[hi]@ls(F[ii],Y[ii])-Y[hi])),end=' ')
 print()
np.savez('/tmp/recovered.npz',z=z,A=A,B=B,Y=Y)
