exec(open('/tmp/probe.py').read().split('for name in')[0])
from scipy.optimize import least_squares
from sklearn.linear_model import OrthogonalMatchingPursuit, MultiTaskLasso
x=np.load('/tmp/inverse_probe.npz')['latent']; y=s['train']['y']; z=s['train']['traj']; t=s['eval_times']; Y=np.column_stack((y,z[:,0],z[:,-1])); F=pol(x,3); names=['int']+[''.join(map(str,j)) for d in (1,2,3) for j in C(range(4),d)]; print('panel shapes',[(k,{a:b.shape for a,b in s[k].items()}) for k in ['panel_a','panel_b','panel_c']]); print('t',t); print('cubic',np.column_stack((names,np.linalg.lstsq(F,Y,rcond=None)[0].round(3)))); print('max', np.max(abs(F@np.linalg.lstsq(F,Y,rcond=None)[0]-Y),0));
rng=np.random.default_rng(23); folds=[(np.arange(52),np.arange(52,72))]+[(np.setdiff1d(np.arange(72),k),k) for k in np.array_split(rng.permutation(72),4)]; YY=np.column_stack((y,z));
def cv(f,ls):
 res=[]
 for i,j in folds:
  pred=f[j]@np.linalg.solve(f[i].T@f[i]+np.diag(np.broadcast_to(ls,f.shape[1])),f[i].T@YY[i]); res.append([np.abs(pred[:,0]-y[j]).mean(),np.abs(pred[:,1:]-z[j]).mean()]);
 return np.array(res)
D={'x':x}
for h in [.3,.6,1,1.5,2.]:
 D[f'sq{h}']=x/np.sqrt(1+h*x*x);D[f'tanh{h}']=np.tanh(h*x)/h;D[f'sin{h}']=np.sin(h*x)/h
for k in ['v1','v2']:
 p=s['panel_c']; u=p['u']; v=p[k]; print('panel',k,'lin sv',np.linalg.svd(np.linalg.lstsq(pol(u,1),v,rcond=None)[0][1:],compute_uv=False));
 for d in [1,2,3]:
  A=pol(u,d); B=pol(x,d)
  coeff=np.linalg.lstsq(A,v,rcond=None)[0]; q=B@coeff; D[f'{k}d{d}']=q
  if d==1:
   for h in [.09,.5,1]:D[f'{k}sq{h}']=q/np.sqrt(1+h*q*q)
  for lam in [.001,.02,.1]:
   proj=A@np.linalg.solve(A.T@A+lam*np.eye(A.shape[1]),A.T); err=(v-proj@v)/(1-np.diag(proj))[:,None]; print(d,lam,abs(err).mean(),abs(q).max())
results=[]
for name,q in D.items():
 for deg in (2,3,4):
  f=pol(q,deg); fs=[('only',f),('plus',np.column_stack((F,f)))];
  for kind,g in fs:
   for lam in [.0001,.003,.02,.1,.3]:
    a=cv(g,lam); results.append((a.max(),a.mean(),name,kind,deg,lam,a.round(4).tolist()))
print('best');
for r in sorted(results)[:24]:print(r)
for n in [7,10,13,17,23]:
 out=[]; supports=[]
 for i,j in folds:
  scale=np.sqrt(np.mean(F[i]**2,0)); m=OrthogonalMatchingPursuit(n_nonzero_coefs=n,fit_intercept=False).fit(F[i]/scale,YY[i]); p=m.predict(F[j]/scale);out.append([abs(p[:,0]-y[j]).mean(),abs(p[:,1:]-z[j]).mean()]);supports.append([names[l] for l in np.flatnonzero(m.coef_[0])]);
 print('omp',n,np.round(out,4),supports[:2]);
