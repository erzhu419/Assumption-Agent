import itertools as it
import pickle, runpy
import numpy as np
from scipy.optimize import least_squares
m=runpy.run_path('/holdout_task/batch_solver.py'); s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; ls=m['_least']; aff=m['_affine']; solve=m['solve_batch']
p=s['panel_a']; u=p['u']; v=p['v']; w=m['_calibrate'](p); G=[m['_calibration_design'](u,j) for j in range(4)]; pred=np.column_stack([a@w[:,j] for j,a in enumerate(G)]); err=np.linalg.norm(pred-v,axis=1); clean=err<6*np.median(err)
for n in [3,4,5]:
 e=[]
 for j,a in enumerate(G):
  a=a[:,:n]
  for r in np.flatnonzero(clean):
   k=clean.copy(); k[r]=False; e.append(a[r]@ls(a[k],v[k,j])-v[r,j])
 print('reference LOO columns, rows, RMSE, max:',n,clean.sum(),np.std(e),max(abs(np.array(e))))
b=s['panel_b']; op=ls(aff(b['u']),b['v']); print('panel B affine RMS:',np.std(aff(b['u'])@op-b['v'])); print('study keys:',sorted(s),'train keys:',sorted(tr)); print('times:',s['eval_times'])
x=m['_inverse'](tr['x'],w); g=aff(x)@op; y=tr['y']; z=tr['traj']; t=s['eval_times']; pairs=list(it.combinations_with_replacement(range(4),2)); Q=np.column_stack([g[:,i]*g[:,j] for i,j in pairs]); L=aff(g)
def assess(i,j):
 d=dict(s,train={k:(v[i] if k in ('x','y','traj') else v) for k,v in tr.items()},test={'x':tr['x'][j]}); out=solve(d); base=np.mean(abs(out['y_hat']-y[j])); state=np.mean(abs(out['state_pred']-z[j])); p=ls(np.c_[L,Q][i],y[i]); selected=abs(p[5:])>.04; M=np.c_[L,Q[:,selected]]; coef=ls(M[i],y[i]); e=y[i]-M[i]@coef
 def candidate(i,j):
  # Fit a nested residual correction; its size and direction are chosen inside training.
  H=np.c_[L,Q]; pp=ls(H[i],y[i]); R=np.c_[L,Q[:,abs(pp[5:])>.04]]; best=(np.inf,None,None)
  for a,b in it.combinations(range(4),2):
   for c in (-2,-1,-.5,.5,1,2):
    q=g[:,a]+c*g[:,b]
    for lam in (.05,.2,1.):
     D=R/(1+lam*q[:,None]**2); B=np.c_[R,D]; fit=ls(B[i],y[i]); r=y[i]-B[i]@fit; leverage=np.sum(np.linalg.svd(B[i],full_matrices=False)[0]**2,axis=1); score=np.mean(r*r)/(1-B.shape[1]/len(i))**2
     if score<best[0]: best=(score,B,fit)
  return np.mean(abs(best[1][j]@best[2]-y[j]))
 rat=candidate(i,j)
 cubic=np.column_stack([g[:,a]*g[:,b]*g[:,c] for a,b,c in it.combinations_with_replacement(range(4),3)]); C=np.c_[M,cubic]; cc=ls(C[i],y[i]); cubic_mae=np.mean(abs(C[j]@cc-y[j])); simple=np.mean(abs(L[j]@ls(L[i],z[i])-z[j])); print('split [scalar, trajectory, rational, cubic, unconstrained time]',*(round(k,7) for k in (base,state,rat,cubic_mae,simple)),'active pairs',[p for p,k in zip(pairs,selected) if k]); return base,state
assess(np.arange(52),np.arange(52,72)); rng=np.random.default_rng(473); 
for _ in range(5):
 p=rng.permutation(72); assess(p[:52],p[52:])
A=aff(g); C=ls(A[:52],z[:52]);
def f(p): return -np.expm1(-t[:,None]*np.exp(p))
def res(p): return (f(p)@ls(f(p),C.T)-C.T).ravel()
p=least_squares(res,np.log([.4,1.8])).x; print('rates and projected trajectory RMS:',np.exp(p),np.std(res(p)),'trajectory observation residual RMS:',np.std(z[:52]-m['_trajectory'](g[:52],g[:52],t,z[:52])))
o=solve(s); assert np.array_equal(o['diagnostics_vec'],np.array([o['y_hat'].mean(),o['y_hat'].std(),o['state_pred'][:,-1].mean(),o['state_pred'][:,-1].std()])); assert all(np.isfinite(v).all() for v in o.values())
