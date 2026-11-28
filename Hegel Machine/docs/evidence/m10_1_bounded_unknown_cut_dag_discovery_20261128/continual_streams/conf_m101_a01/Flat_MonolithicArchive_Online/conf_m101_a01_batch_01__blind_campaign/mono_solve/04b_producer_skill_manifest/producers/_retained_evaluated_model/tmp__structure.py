from features import *
from scipy.linalg import solve
np.set_printoptions(precision=4,suppress=True)
p=s['panel_c']; u=p['u']; v1=p['v1']; v2=p['v2'];
for out in (v1,v2):
 print('PANEL'); R=out-af(u)@ls(af(u),out); print('samples u,res',np.c_[u,R][:7]);
 F={'separable2':np.c_[af(u),u**2], 'separable3':np.c_[af(u),u**2,u**3], 'separable4':np.c_[af(u),u**2,u**3,u**4], 'pairwise':np.c_[af(u),*[u[:,i]*u[:,j] for i in range(4) for j in range(i+1,4)]]}
 for g in [.5,1,1.6,2,3,4]:
  for nm,fn in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('quadmod',lambda z:z*z/(1+z*z)),('root',lambda z:np.sqrt(1+z*z))]: F[nm+str(g)]=np.c_[af(u),fn(g*u)]; F['just_'+nm+str(g)]=af(fn(g*u))
 scores=[]
 for k,X in F.items():
  e=[]
  for j in range(len(u)):
   i=np.arange(len(u))!=j;e.append(X[j]@ls(X[i],out[i])-out[j]);
  scores.append((np.mean(np.abs(e)),k,np.mean(abs(out-X@ls(X,out)))))
 print(sorted(scores)[:12]);
 for k in [q[1] for q in sorted(scores)[:2]]:print(k,ls(F[k],out))
print('target additive cross coefficient');
for key in ['x','a','ai','xb','ab','aib','xc','ac','aic']:
 U=coords[key]; W=ls(pol(U,3),Y); print(key, 'quadratic',ls(pol(U,2),Y)[5:].T,'cubic_amp',W[15:,1]);
print('joint residual correlation',np.corrcoef((Y-pol(x,2)@ls(pol(x,2),Y)).T)); print('SVD traj residual proj',np.linalg.svd(tr-h[:,None]*v,full_matrices=False)[1]);
for key,U in [('x',x),('a',coords['a']),('ai',coords['ai']),('ab',coords['ab']),('aib',coords['aib'])]:
 for j in range(2):
  target=Y[:,j]; I=np.arange(52); Q=np.arange(52,72); F=pol(U,2); L=af(U);
  fs={'quad':lambda p:F@p,'sinh_quad':lambda p:p[0]+p[1]*np.sinh(np.clip(F@p[2:],-10,10)),'tanh_quad':lambda p:p[0]+p[1]*np.tanh(F@p[2:]), 'satlinear_quad':lambda p:F@p[:-5]+np.tanh(L@p[-5:]), 'sq_nonlin_quad':lambda p:F@p[:-5]+(np.tanh(L@p[-5:]))**2, 'sin_linear_quad':lambda p:F@p[:-5]+np.sin(L@p[-5:]), 'rat_quad':lambda p:(F@p[:-5])/(1+(L@p[-5:])**2), 'rat_resid_quad':lambda p:L@p[:5]+(F@p[5:-5])/(1+(L@p[-5:])**2)}
  q=ls(F[I],target[I]);
  for nm,fun in fs.items():
   if key not in ('x','ab') and nm not in ('rat_quad','satlinear_quad'):continue
   best=None
   for seed in range(5):
    r=np.random.default_rng(seed); start=q if nm=='quad' else np.r_[np.mean(target[I]),1.,q] if nm in ('sinh_quad','tanh_quad') else np.r_[np.zeros(5) if nm=='rat_resid_quad' else [],q,.5*r.normal(size=5)];
    fit=least_squares(lambda p:(fun(p)-target)[I],start,max_nfev=250); e=abs(fun(fit.x)-target); a=(np.mean(e[I]**2),np.mean(e[I]),np.mean(e[Q]),nm);
    if best is None or a<best:best=a
   print(key,j,*best[1:]);
