from compose import *
from scipy.spatial.distance import cdist
from itertools import combinations
p=s['train']; y=p['y']; T=p['traj']; time=s['eval_times']; U,S,V=np.linalg.svd(T,full_matrices=False); Y=np.c_[y,U[:,0]*S[0]]; x=p['x']; z=inverse(afit(s)['diagsecond'],x); op=operators(s); B,C,D=[op[k] for k in 'BCD']; b=B(z); c=C(b); d=D(b)
folds=[np.arange(52,72)]+[np.random.default_rng(i).permutation(72)[:20] for i in range(12,18)]; inds=[(np.setdiff1d(np.arange(72),f),f) for f in folds]; scales=np.array([1,np.abs(V[0]).mean()]);
def cv(A,reg):
 out=[]
 for tr,te in inds:
  m=A[tr].mean(0); sd=A[tr].std(0).clip(1e-8); aa=(A-m)/sd; w=np.linalg.solve(aa[tr].T@aa[tr]+reg*np.eye(A.shape[1]),aa[tr].T@(Y[tr]-Y[tr].mean(0)));out.append(np.mean(abs(aa[te]@w+Y[tr].mean(0)-Y[te]),0)*scales)
 return np.array(out)
base={'b':b,'c':c,'d':d,'cd':np.c_[c,d],'bcd':np.c_[b,c,d],'DC':D(c),'CD':C(d),'z':z}; res=[]
for n,a in base.items():
 fs={'lin':a}; m=a.shape[1];i,j=np.triu_indices(m); fs['quadratic']=np.c_[a,a[:,i]*a[:,j]]
 if m==4:
  for shift in (1,2,3):
   r=np.roll(a,-shift,axis=1)
   for w in (.2,.5,1.,1.7,2.5,4.):
    for q,k in [(a+r,'sum'),(a-r,'dif'),(a*r,'prod'),(a+w*r,'mix'),(a-w*r,'sub')]:
     if k in ('mix','sub'):
      for fn,g in [('sin',np.sin),('tanh',np.tanh),('cos',np.cos)]:fs[f'{shift} {w} {k} {fn}']=np.c_[a,g(1.7*q),c]
     else:
      for fn,g in [('sin',np.sin),('tanh',np.tanh),('cos',np.cos)]:fs[f'{shift} {w} {k} {fn}']=np.c_[a,g(w*q),c]
    fs[f'axis {w}']=np.c_[a,c,np.sin(w*a),np.cos(w*a)]
 for k,v in fs.items():
  for l in (.025,.4,2):res.append((n,k,l,cv(v,l)))
for j in (0,1):
 for sort in ('mean','gate'):
  print(j,sort,[(n,k,l,e[0].round(4),e.mean(0).round(4),e.max(0).round(4)) for n,k,l,e in sorted(res,key=lambda x:x[-1].mean(0)[j] if sort=='mean' else x[-1][0,j])[:5]])
# Fit a single nonlinear latent projection, profiling the linear nuisance readout.
for target in range(2):
 results=[]
 for n,a0 in base.items():
  if a0.shape[1]!=4:continue
  for q,baseline in [('none',np.empty((72,0))),('cyc',c),('linear',np.c_[c,d]),('own',a0)]:
   for g,fun in [('tanh',np.tanh),('sin',np.sin),('softsign',lambda x:x/np.sqrt(1+x*x)),('square',np.square),('rat',lambda x:1/(1+x*x))]:
    errs=[]; fits=[]
    for tr,te in inds[:4]:
     a=(a0-a0[tr].mean(0))/a0[tr].std(0); X=aff(baseline); tgt=Y[:,target]; Xtr=X[tr]; F=np.linalg.pinv(Xtr); r=tgt[tr]-Xtr@(F@tgt[tr]); rng=np.random.default_rng(103); starts=[np.r_[0.,ls(aff(a[tr]),r)[1:]*3],*rng.normal(0,.5,(4,5))]; best=(np.inf,None)
     for init in starts:
      def resid(par):
       h=fun(aff(a[tr])@par); h-=Xtr@(F@h); amp=(h@r)/(h@h+1e-12); return np.r_[amp*h-r,0.009*par]
      fit=least_squares(resid,init,max_nfev=70); val=fit.cost
      if val<best[0]:best=(val,fit.x)
     h=fun(aff(a)@best[1]); XA=np.c_[X,h]; pred=XA[te]@ls(XA[tr],tgt[tr]); errs.append(np.mean(abs(pred-tgt[te]))*scales[target]); fits.append(best[0]);
    results.append((np.mean(errs),n,q,g,np.round(errs,4),np.mean(fits)))
 print('projection',target,*sorted(results)[:10],sep='\n')
