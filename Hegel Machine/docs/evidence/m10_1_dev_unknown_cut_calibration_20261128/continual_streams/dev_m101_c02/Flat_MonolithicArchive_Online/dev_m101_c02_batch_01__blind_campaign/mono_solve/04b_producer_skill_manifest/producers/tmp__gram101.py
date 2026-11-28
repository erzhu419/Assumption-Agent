exec(open('/tmp/graph101.py').read().split('for key,p,v in')[0])
from scipy.special import erf
u=s['panel_b']['u']; v=s['panel_b']['v']; w=(x-off)/sc
print('A calibration rows',np.sort(np.linalg.norm(a['v']-O(a['u'])@L(O(a['u'][g]),a['v'][g]),axis=1))); print('A matrix',L(O(a['u'][g]),a['v'][g]))
print('times, telemetry',t,s['telemetry_vec']); print('gate',open('/tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json').read())
fams={'tanh':lambda q,k:np.tanh(k*q), 'atan':lambda q,k:np.arctan(k*q), 'erf':lambda q,k:erf(k*q), 'exp':lambda q,k:q*(-np.expm1(-k*q*q)), 'rat2':lambda q,k:q**3/(1+k*q*q),'rat4':lambda q,k:q**3/(1+k*q**4),'rat3':lambda q,k:q**3/(1+k*abs(q)**3),'ratabs':lambda q,k:q**3/(1+k*abs(q)),'powers':lambda q,k:q**3+k*q**5,'sat':lambda q,k:np.tanh(k*q**3),'log':lambda q,k:q*np.log1p(k*q*q)}
P=lambda q:np.column_stack([q[:,i]*q[:,j] for i,j in combinations(range(4),2)])
models=[]
for n,f in fams.items():
 for m in ('shared','mixed','shared+q','mixed+q'):
  def des(q,r):
   h=f(q,r[0]); F=O(h+r[1]*q) if m.startswith('shared') else np.c_[O(q),h]; return np.c_[F,q*q] if m.endswith('+q') else F
  shared=m.startswith('shared'); lo=[.001,-8] if shared else [.001]; hi=[8,8] if shared else [8]
  def residual(r):
   F=des(u,r);return (F@L(F,v)-v).ravel()
  fit=min([least_squares(residual,[k,0] if shared else [k],bounds=(lo,hi),max_nfev=140) for k in [.15,1,2]],key=lambda a:a.cost)
  F=des(u,fit.x); C=L(F,v); z=des(w,fit.x)@C; errors=[]
  for i,j in folds:
   A=np.c_[O(z),z[:,0]*z[:,2]+z[:,1]*z[:,3]]; B=O(z); U,S,V=np.linalg.svd(Y[i,1:],full_matrices=False); T=V[:2]; e=abs(np.c_[A[j]@L(A[i],Y[i,0]),(B[j]@L(B[i],Y[i,1:]@T.T))@T]-Y[j]); errors.append([e[:,0].mean(),e[:,1:].mean()]);
  h=np.sum(np.linalg.svd(F,full_matrices=False)[0]**2,axis=1); loo=np.mean(abs(residual(fit.x).reshape(v.shape)/(1-h[:,None]))); models.append((n,m,fit.x,np.sqrt(np.mean(fit.fun**2)),loo,np.array(errors),C,z))
for key in [lambda k:k[4],lambda k:k[5].max(0).sum()]:
 print('ranking')
 for n,m,r,rm,loo,e,C,z in sorted(models,key=key)[:10]:print(n,m,r,'panel rmse/loo',rm,loo,'56/16',e[0],'worst',e.max(0))
n,m,r,rm,loo,e,C,z=min(models,key=lambda k:k[5].max(0).sum()); print('best C',C); print('scalar full quadratic',L(np.c_[O(z),P(z),z*z],Y[:,0])); ee=Y[:,1:]-O(z)@L(O(z),Y[:,1:]); print('spatial residual sv',np.linalg.svd(ee,compute_uv=False)); print('trajectory residual correlation with sensor polynomials',L(np.c_[O(z),P(z),z*z],ee))
