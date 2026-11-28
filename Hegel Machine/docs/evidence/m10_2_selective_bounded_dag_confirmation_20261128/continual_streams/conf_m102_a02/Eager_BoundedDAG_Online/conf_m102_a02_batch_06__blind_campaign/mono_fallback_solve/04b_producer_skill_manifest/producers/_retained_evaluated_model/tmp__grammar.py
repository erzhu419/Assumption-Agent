exec(open('/tmp/readout.py').read().split("print('range'")[0])
from scipy.optimize import least_squares
q=(g*np.roll(g,2,axis=1))[:,:2]; D3=np.c_[D1,q]
def out(a): return cv(a)
Z={'z':z,'g':g,'v':v}; C={}
for i in range(4):
 for j in range(i+1,4):
  C['gg%d%d'%(i,j)]=(g[:,i]*g[:,j])[:,None]
for nm,a in Z.items():
 for mk,b in Z.items():
  for shift in range(4):
   C[f'{nm}{mk} rol{shift}']=a*np.roll(b,shift,axis=1)
 for mode,f in [('sq',lambda x:x*x),('cube',lambda x:x**3),('sin',np.sin),('tanh',np.tanh),('rat',lambda x:x/(1+x*x))]:
  for w in [.5,1.,2.]: C[f'{nm}-{mode}-{w}']=f(w*a)
for scale in [-1.,-.5,-.25,.25,.5,.75,1.,1.25,2.]:
 for base in ['gv','gz','zv']:
  a=Z[base[0]]+scale*Z[base[1]]
  for k,p in [('quadratic',poly(a)[:,5:]),('opposite',(a*np.roll(a,2,axis=1))[:,:2]),('sat',np.tanh(a)),('sine',np.sin(a))]: C[f'{base}{scale} {k}']=p
for m,b in [('trajectory',T[:,-1:]),('scalar1',D@fit(D,y[:,None])),('scalar2',D3@fit(D3,y[:,None]))]:
 for name,fn in [('sq',lambda u:u*u),('cube',lambda u:u**3),('sin',np.sin),('tanh',np.tanh)]:
  C[m+' '+name]=fn(b)
for base,A in [('D',D),('D1',D1),('D3',D3)]:
 ans=[]
 for k,b in C.items():
  if k.startswith(('trajectory','scalar')):continue
  e=out(np.c_[A,b]); ans.append((float(.6*e[1]+.4*e[3]),k,e))
 print(base, sorted(ans,key=lambda x:x[0])[:16])
# Compute nested sparse residual fits, selecting features with inner random folds.
for typ in ('g-full','gv-full','all-full'):
 if typ=='g-full': a=g
 if typ=='gv-full': a=np.c_[g,v]
 if typ=='all-full': a=np.c_[g,z,v]
 B=poly(a)[:,1+a.shape[1]:]; B=np.c_[B,g**3,z**3,v**3]; B=B/(np.linalg.norm(B,axis=0)+1e-12)
 def select(i,j,lam):
  ai,aj=D3[i],D3[j]; bi,bj=B[i],B[j]; coef=fit(ai,np.c_[y[i],bi]); ey=y[i]-ai@coef[:,0]; ebi=bi-ai@coef[:,1:]; ebj=bj-aj@coef[:,1:]; w=np.linalg.solve(ebi.T@ebi+lam*np.eye(B.shape[1]),ebi.T@ey)
  return aj@coef[:,0]+ebj@w
 for lam in [.0001,.001,.01,.05,.2,1]:
  es=np.array([abs(y[j]-select(i,j,lam)).mean() for i,j in splits]);print(typ,lam,np.round([es[0],es[1:-1].mean(),es[1:-1].max(),es[-1]],5))
# Optimize the mixing scale of a shared quadratic readout stage by variable projection.
for mode in ('opposite','full','shared'):
 def mat(idx,k):
  a=g[idx]+k*v[idx]; P=(a*np.roll(a,2,axis=1))[:,:2]
  if mode=='full': P=poly(a)[:,5:]
  if mode=='shared': P=P[:,1:]
  return np.c_[D1[idx],P]
 es=[]; params=[]
 for i,j in splits:
  def residual(k):
   A=mat(i,k[0]);return A@fit(A,y[i])-y[i]
  sol=min([least_squares(residual,[a],bounds=(-3.,3.),max_nfev=60) for a in [-1.,0.,1.]],key=lambda x:x.cost)
  k=sol.x[0]; params.append(k); es.append(abs(mat(j,k)@fit(mat(i,k),y[i])-y[j]).mean())
 print('mixed',mode,np.round([es[0],np.mean(es[1:-1]),max(es[1:-1]),es[-1]],5),'param',np.round(params,3))
