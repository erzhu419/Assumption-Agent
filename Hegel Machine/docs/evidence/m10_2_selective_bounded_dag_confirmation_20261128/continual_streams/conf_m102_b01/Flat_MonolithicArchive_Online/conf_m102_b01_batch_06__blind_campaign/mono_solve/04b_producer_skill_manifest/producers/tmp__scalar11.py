import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from refine6 import *
import importlib.util
sp=importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py');m0=importlib.util.module_from_spec(sp);sp.loader.exec_module(m0)
N=len(Y); y=Y[:,0]; state,old=m0._designs(s,x); B0=A(np.c_[q,Z]); core=A(np.c_[w,q,Z]); h=state@L(state[:52],Y[:52,-1]); print('state coefficients',L(state[:52],Y[:52,-1]));print('y coeff',L(state[:52],y[:52]));
splits=[(np.arange(52),np.arange(52,N))]+[(p[:52],p[52:]) for p in [np.random.default_rng(i).permutation(N) for i in range(20)]]
# Add extrapolation slices by downstream response amplitude and proximity to panel support.
for v in [np.abs(h),np.max(abs(z),axis=1),np.max(abs(w),axis=1)]:
 p=np.argsort(v);splits.append((p[:52],p[52:]))
def metrics(X):
 es=[];ts=[]
 for tr,te in splits:
  c=L(X[tr],y[tr]);es.append(np.mean(abs(X[te]@c-y[te])));ts.append(np.mean(abs(X[tr]@c-y[tr])))
 return np.round([es[0],np.mean(es[1:21]),max(es[1:21]),*es[21:],ts[0]],5)
for n,d in [('state',state),('core',core),('old',old),('minimal',np.c_[B0,q[:,0]*w[:,2]**2]),('old-cubic',np.c_[old,q[:,0]*w[:,2]**2]),('core-rat',np.c_[core,q[:,0]/(1+.22*w[:,2]**2)]),('core-qcubic',np.c_[core,q[:,0]**3])]:print(n,metrics(d))
vals=[]
for bn,b in [('state',state),('core',core),('B0',B0),('old',old)]:
 for n,v in [('h',h),('z',z),('w',w),('q',q),('r',r),('R',R),('Z',Z)]:
  for j,a in enumerate(np.asarray(v).reshape(N,-1).T):
   for k in [.22,1.25]:
    for typ,f in [('tanh',np.tanh(k*a)),('sin',np.sin(k*a)),('sat',a/(1+k*a*a)),('quad',a*a),('cube',a**3),('den',1/(1+k*a*a)),('soft',np.logaddexp(k*a,-k*a))]:
     for nn,d in [('add',np.c_[b,f]),('q0',np.c_[b,q[:,0]*f]),('h',np.c_[b,h*f]),('full',b/(1+k*a[:,None]**2))] if typ=='den' else [('add',np.c_[b,f]),('q0',np.c_[b,q[:,0]*f]),('h',np.c_[b,h*f])]:
      e=metrics(d); vals.append((e[0]+e[1]+e[2],(bn,n,j,k,typ,nn),e))
for _,n,e in sorted(vals,key=lambda a:a[0])[:24]: print(n,e)
