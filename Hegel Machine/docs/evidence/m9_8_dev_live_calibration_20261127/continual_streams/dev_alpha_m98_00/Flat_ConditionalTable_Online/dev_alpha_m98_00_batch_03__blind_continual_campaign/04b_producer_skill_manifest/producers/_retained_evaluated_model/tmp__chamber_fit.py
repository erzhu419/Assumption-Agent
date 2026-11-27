import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 exec(open('/tmp/chamber_probe.py').read())
from scipy.optimize import least_squares

def diagnostic_drift(mode):
 if mode=='deployment':return drift
 def combine(*groups):return {k:np.concatenate([gg[k] for gg in groups]) for k in ('lab_id','stratum','ref_coupons','coupon_features_raw')}
 if mode=='diagnostic':
  mask=np.arange(len(hold))<len(hold)//2
  parts=[{k:q[k][j] for k in ('lab_id','stratum','ref_coupons','coupon_features_raw')} for j in (mask,~mask)]
  return m._aligned_features(*parts)@theta[2:]
 return m._aligned_features(combine(tr,te),q)[-len(hold):]@theta[2:]

def fit_model(dr, form, train, test):
 def predictors(par):
  tt=tau*np.exp(par) if form.endswith('time') else tau
  G=-np.expm1(-t[:,None]/tt)
  def branch(a,b):return np.column_stack((a[:,None]*G[:,b],a/tt[b]))
  def lin(a):return np.column_stack((a[:,None]*t,a))
  A=[hold*theta[b] for b in range(2)]
  H=[-z[:,b]*theta[b] for b in range(2)]
  base=branch(A[0],0)+branch(A[1],1)+lin(dr)
  cols=[branch(dr*tt[b],b)-lin(dr) for b in range(2)]
  if form.startswith('fixed'):base+=branch(H[0],0)+branch(H[1],1)
  elif form.startswith('shared'):cols += [branch(H[0],0)+branch(H[1],1)]
  else:cols += [branch(H[0],0),branch(H[1],1)]
  if 'free' in form: cols += [lin(dr)]
  if 'shift' in form:cols += [lin(np.ones_like(dr)),branch(np.ones_like(dr),0),branch(np.ones_like(dr),1)]
  return base,np.stack(cols,axis=2)
 obs=np.c_[y,v]; start=np.zeros(2) if form.endswith('time') else np.empty(0)
 base,P=predictors(start); k=P.shape[-1]; C=P[train].reshape(-1,k); target=(obs-base)[train].ravel(); beta=np.linalg.lstsq(C,target,rcond=None)[0];
 def predict(p):
  base,P=predictors(p[k:]); return base+P@p[:k]
 p=least_squares(lambda p:(predict(p)-obs)[train].ravel(),np.r_[beta,start],loss='soft_l1',f_scale=.02,max_nfev=400,x_scale='jac').x
 return p,(predict(p)-obs)[test]

allrows=np.arange(len(hold)); labs_q=q['lab_id']; results=[]
for mode in ('deployment','diagnostic','combined'):
 dr=diagnostic_drift(mode)
 print('drift',mode,np.round(dr,3))
 for form in ('fixed','shared','split','fixedfree','splitfree','splitshift','fixedtime','splittime','splitfreetime'):
  beta,err=fit_model(dr,form,allrows,allrows)
  cv=[fit_model(dr,form,allrows[labs_q!=lab],allrows[labs_q==lab])[1] for lab in np.unique(labs_q)]
  c=np.array([[np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2))] for e in cv]); worst=c.max(0); rms=np.sqrt(np.mean(err**2,axis=0)); score=worst.sum()
  results.append((score,mode,form,beta,np.sqrt(np.mean(err[:,:-1]**2)),rms[-1],worst))
for _,mode,form,beta,cr,rr,worst in sorted(results):print(mode,form,'params',np.round(beta,5),'in',np.round([cr,rr],5),'worst LOLO',np.round(worst,5))
