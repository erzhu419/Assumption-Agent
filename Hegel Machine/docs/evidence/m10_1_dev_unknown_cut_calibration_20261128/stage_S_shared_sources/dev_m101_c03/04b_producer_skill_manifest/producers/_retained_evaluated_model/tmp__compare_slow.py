exec(open('/tmp/final_eval.py').read().split('errs=[]')[0])
compact=np.c_[L,M[:,1]]; full=np.c_[L,M]; diag=np.c_[L,M[:,[0,1,5,7,10,14,15]]]
for rate in (False, True):
 errors=[]
 for va in folds:
  tr=~va; ft=F
  if rate:
   def resid(k):
    f=-np.expm1(-t[:,None]*k); return (f@np.linalg.lstsq(f,q[tr].T,rcond=None)[0]-q[tr].T).ravel()
   ft=-np.expm1(-t[:,None]*least_squares(resid,[1.8,.4],bounds=(.01,10)).x)
  amps=np.linalg.lstsq(ft,q[tr].T,rcond=None)[0].T
  fast=D[va]@m._ridge(D[tr],amps[:,0],.00015)
  tails=[w[va]@m._ridge(w[tr],amps[:,1],.0003) for w in (compact,full,diag)]
  candidates=[tails[0],.75*tails[0]+.25*tails[1],.5*(tails[0]+tails[1]),.5*(tails[0]+tails[2]),tails[1]]
  errors.append([abs(np.c_[fast,a]@ft.T-q[va]).mean(0) for a in candidates])
 e=np.array(errors)
 print('fitted rates:',rate,'rows: compact, mostly compact, compact/full, compact/diagonal, full')
 print(np.round(np.c_[e[:6].mean((0,2)),e[:6].max((0,2)),e[6:].max(2).T],5))
