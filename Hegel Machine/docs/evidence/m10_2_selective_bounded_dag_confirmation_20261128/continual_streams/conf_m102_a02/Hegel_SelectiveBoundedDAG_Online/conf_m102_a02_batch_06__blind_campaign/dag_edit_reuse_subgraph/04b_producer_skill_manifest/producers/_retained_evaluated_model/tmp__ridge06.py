exec(open('/tmp/fit06.py').read().split("for name,h in [('cal',q)")[0])
from scipy.optimize import least_squares
from sklearn.preprocessing import StandardScaler
Z=B; rng=np.random.default_rng(829); n=len(y); splits=[(np.arange(52),np.arange(52,n)),(np.arange(20,n),np.arange(20))]; S,V=np.linalg.svd(tr,full_matrices=False)[1:]; print('shapes',len(y),y.shape,tr.shape)
for name,h in [('B',B),('q+B',np.c_[q,B])]:
 for d in (1,2,3):
  f=design(h,d); f=np.c_[np.ones(len(f)),StandardScaler().fit_transform(f[:,1:])]; print('ridge',name,d,[(a,*[round(np.mean(abs(f[j]@np.linalg.solve(f[i].T@f[i]+a*np.eye(f.shape[1]),f[i].T@v[i])-v[j])),4) for i,j in splits for v in (y,tr)]) for a in (.001,.03,.15,1.)]);
funcs={'tanh':np.tanh,'sin':np.sin,'softplus':lambda u:np.logaddexp(0,u),'exp':lambda u:np.exp(np.clip(u,-12,12)), 'rat':lambda u:1/(1+u*u),'cubic':lambda u:u**3}
for name,phi in funcs.items():
 for kind in ('ridge','gated','quad+ridge','twoquad'):
  A=design(B,1); Q=design(B,2); T=tr@V[0]; results=[]
  def f(p):
   gate=phi(A@p); gate=gate[:,None]
   return {'ridge':np.c_[A,gate],'gated':np.c_[A,A*gate],'quad+ridge':np.c_[Q,gate],'twoquad':np.c_[Q,Q*gate]}[kind]
  for target in (T,y):
   scores=[]
   for i,j in splits:
    def resid(p):
     a=f(p)[i]; return a@ls(a,target[i])-target[i]
    candidates=[least_squares(resid,ini,bounds=(-9,9),max_nfev=85) for ini in rng.normal(0,0.8,(9,5))]; o=min(candidates,key=lambda o:o.cost); a=f(o.x); pred=a[j]@ls(a[i],target[i]); scores.append((round(np.mean(abs(pred-target[j])),4),round(np.sqrt(np.mean(o.fun**2)),4),o.x.round(2).tolist()))
   results.append(scores)
  print('nonlinear',name,kind,results,flush=True)
