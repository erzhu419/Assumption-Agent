import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from chamber_structure import *
from itertools import combinations_with_replacement
from scipy.linalg import svd
allrows=np.ones(len(h),bool)
for label,dv in [('joint',dr),('deployment',dr_deploy),('diagnostic-only',theta[2]+old._features(g)@theta[3:])]:
 pred,_,ts=fit('linear-memory',allrows,dv); rr=y-pred
 print(label,'rmse',rm(rr[:,:-1]),rm(rr[:,-1]),'singular',np.round(svd(rr,compute_uv=False)[:6],5))
 dv0=np.column_stack((h,z,dv)); xbase=np.column_stack((np.ones(len(h)),dv0)); q=np.linalg.qr(xbase)[0]; aerr=amps(ts)-q@(q.T@amps(ts));
 names=['h','z0','z1','d','gap0','gap1','zfast','zslow',*[f'p{i}' for i in range(6)],*[f'x{i}' for i in range(4)],*[f'hold{i}' for i in range(4)]]
 f=np.column_stack((dv0,h[:,None]-z,memory(p,dt,ts[0]),memory(p,dt,ts[1]),p,xd,g['hold_channel_raw']))
 signals={k:v for k,v in zip(names,f.T)}
 for i,j in combinations_with_replacement(range(10),2):
  signals[names[i]+'*'+names[j]]=f[:,i]*f[:,j]
 for k in range(8):
  v=f[:,k]; vv=(v-v.mean())/v.std()
  for c in [.25,1.,4.]:
   signals[f'{names[k]}/(1+{c}*{names[k]}^2)']=vv/(1+c*vv*vv)
   signals[f'd/(1+{c}*{names[k]}^2)']=dv/(1+c*vv*vv)
 def rank(e):
  scored=[]
  for k,v in signals.items():
   w=v-q@(q.T@v)
   if np.linalg.norm(w)>1e-7: scored.append((abs(w@e)/(np.linalg.norm(e)*np.linalg.norm(w)),k))
  return [(round(s,3),k) for s,k in sorted(scored,reverse=True)[:9]]
 for j in range(3): print('amplitude',j,rank(aerr[:,j]))
 print('reference offset-scale',[(k,np.round(np.r_[[(np.median(s['ref_coupons'][s['lab_id']==k].mean(0))) for s in (tr,te,g)],[(np.median(s['ref_coupons'][s['lab_id']==k].std(0))) for s in (tr,te,g)]],4)) for k in np.unique(labs)])
print('Chamber unconstrained amplitude residual singular values',np.round(svd(r,compute_uv=False),5));print('benchtop residual quantiles',np.quantile(np.abs(old._design(tau,coupon(tr,old._features(tr,te)[:n]),dt,t)[0]@theta-obs(tr)[0]),[.5,.9,.95,.99,1]))
print('diagnostic past range',np.min(p,axis=0),np.max(p,axis=0)); pp=te['past_channel_raw'].sum(axis=-1); print('deployment past range',np.min(pp,axis=0),np.max(pp,axis=0))
