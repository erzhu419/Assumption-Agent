exec(open('/tmp/chamber_probe.py').read().split('for k in (1,2,3,4):')[0])
from scipy.optimize import least_squares
print('diagnostic outer keys',list(q))
for i,s in enumerate((a,b,c)):
 print('refs',i,[(int(l),round(float(np.median(s['ref_coupons'][s['lab_id']==l].mean(0))),3),round(float(np.median(s['ref_coupons'][s['lab_id']==l].std(0))),3)) for l in np.unique(s['lab_id'])])
Z=old._memory(taus,past,dt); labs=c['lab_id']; O=np.ones(len(Y));H=hold
D={name:theta[2]+xx@theta[3:] for name,xx in [('pooled',x[n+m:]),('diag_only',old._features(c)),('train_diag',old._features(a,c)[n:])]}

def score(X, fixed=False):
 def f(ix):
  def calc(v):
   T=taus if fixed else np.exp(v); B=basis(T); W=np.linalg.pinv(X[ix]) @ Y[ix] @ np.linalg.pinv(B).T
   return X@W@B.T-Y,T,W
  v=np.log(taus) if fixed else least_squares(lambda v:calc(v)[0][ix].ravel(),np.log(taus),bounds=(np.log(.08),np.log(18))).x
  return calc(v)
 R,T,W=f(np.ones(len(Y),bool)); C=np.empty_like(R)
 for l in np.unique(labs):
  mask=labs==l; C[mask]=f(~mask)[0][mask]
 return np.round([np.sqrt(np.mean(R**2)),max(np.sqrt(np.mean(C[labs==l]**2)) for l in np.unique(labs)),*np.sqrt(np.mean(C**2,axis=0))[[0,-2,-1]]],6),T,W
for name,r in D.items():
 print('FORCING',name,'delta',np.max(abs(r-drift)))
 for desc,F in [('affine',[]),('mixed',[H*r]),('drift_sq',[r*r]),('load_sq',[H*H]),('all_quad',[H*r,r*r,H*H]),('memory0',[Z[:,0]]),('memory1',[Z[:,1]]),('r*mem',[r*Z[:,0],r*Z[:,1]]),('load/mem',[H*Z[:,0],H*Z[:,1]])]:
  s,T,W=score(np.column_stack([O,H,r]+F)); print(desc,s,np.round(T,4));
  if s[0]<.015:print(' coef',W.round(5))
