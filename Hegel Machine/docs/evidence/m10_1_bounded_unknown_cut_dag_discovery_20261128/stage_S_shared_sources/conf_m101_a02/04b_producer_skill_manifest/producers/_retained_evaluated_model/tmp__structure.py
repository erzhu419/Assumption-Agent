import numpy as np,pickle
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; y=tr['y']; z=tr['traj']; t=s['eval_times']; np.set_printoptions(precision=6,suppress=True,linewidth=130)
L=lambda x: np.c_[np.ones(len(x)),x]
ix=list(combinations_with_replacement(range(4),2)); Q=lambda x: np.column_stack([x[:,i]*x[:,j] for i,j in ix]); A=lambda x: np.c_[L(x),Q(x)]
def fit(a,b): return np.linalg.lstsq(a,b,rcond=None)[0]
def hessian(c):
 H=np.zeros((4,4));
 for (i,j),v in zip(ix,c[5:]): H[i,j]=H[j,i]=v/(1+(i!=j))
 return H
for k in [52,72]:
 c=fit(A(x[:k]),y[:k]); H=hessian(c); print('y',k,'resid std',np.std(y[:k]-A(x[:k])@c),'linear',c[:5],'quad',H,'eigh',np.linalg.eigh(H),sep='\n')
 print('traj residual std, linear vs quadratic',np.std(z[:k]-L(x[:k])@fit(L(x[:k]),z[:k])),np.std(z[:k]-A(x[:k])@fit(A(x[:k]),z[:k])))
 C=fit(L(x[:k]),z[:k]); print('time basis',np.linalg.svd(C)[1:]); print('increments',np.diff(C,axis=1))
print('calibrated panel coordinates:')
p=s['panel_a']; u=p['u']; v=p['v']; good=np.max(abs(v-u),axis=1)<1; bias,scale=np.array([fit(np.c_[np.ones(good.sum()),a[good]],b[good]) for a,b in zip(u.T,v.T)]).T
b=s['panel_b']; B=fit(L(b['u']),b['v']); print('M B',B); print('quadratic in raw, A, invA, B, A->B, invA->B');
for az in (x,x*scale+bias,(x-bias)/scale):
 for zz in (az,L(az)@B):
  c=fit(A(zz),y); print('linear',c[:5], 'quadratic',c[5:]);
def reduced_y(sel, xnew,r):
 x0=x[sel]; y0=y[sel]; a=L(x0); H=hessian(fit(A(x0),y0)); d,U=np.linalg.eigh(H); inds=np.argsort(abs(d))[-r:]; signs=np.sign(d[inds]); F=U[:,inds]*np.sqrt(abs(d[inds]));
 def shift(p): return np.sum((x0@p.reshape(4,r))**2*signs,axis=1)
 def err(p):
  res=y0-shift(p); return res-a@fit(a,res)
 F=least_squares(err,F.ravel()).x.reshape(4,r); return L(xnew)@fit(a,y0-shift(F.ravel()))+np.sum((xnew@F)**2*signs,axis=1)
def traj(sel,xnew,kind):
 a=L(x[sel]); C=fit(a,z[sel]);
 if kind=='rank2':
  u,d,v=np.linalg.svd(a@C,full_matrices=False); C=(C@v[:2].T)@v[:2]
 elif kind!='linear':
  def basis(p):
   if kind=='2exp': return np.vstack([-np.expm1(-np.exp(v)*t) for v in p]);
   if kind=='cascade':
    k,j=np.exp(p); a=-np.expm1(-k*t); b=a-k*t*np.exp(-k*t) if abs(k-j)<1e-6 else 1-(j*np.exp(-k*t)-k*np.exp(-j*t))/(j-k); return np.vstack((a,b))
   if kind=='explin': return np.vstack((-np.expm1(-np.exp(p[0])*t),t))
   if kind=='exppulse': return np.vstack((-np.expm1(-np.exp(p[0])*t),t*np.exp(-np.exp(p[1])*t)))
  def err(p):
   g=basis(p); return (a@(C-C@np.linalg.pinv(g)@g)).ravel()
  opt=least_squares(err,[-.3] if kind=='explin' else [-.3,.5],bounds=(-5,4)); g=basis(opt.x); C=C@np.linalg.pinv(g)@g
  if len(sel)==52 and sel[0]==0 and sel[-1]==51: print(kind,np.exp(opt.x),np.std(err(opt.x)))
 return L(xnew)@C
models=['linear','rank2','2exp','cascade','explin','exppulse']; rng=np.random.default_rng(190)
splits=[(np.arange(52),np.arange(52,72))]; splits += [(np.setdiff1d(np.arange(72),va),va) for va in np.array_split(rng.permutation(72),4)]
for sel,va in splits:
 print('Y MAE',np.mean(abs(y[va]-A(x[va])@fit(A(x[sel]),y[sel]))),[(r,round(np.mean(abs(y[va]-reduced_y(sel,x[va],r))),6)) for r in (1,2,3)])
 print('T MAE (mean,max horizon)',[(k,np.round([np.mean(e),np.max(np.mean(e,axis=0))],6)) for k in models for e in [abs(z[va]-traj(sel,x[va],k))]])
