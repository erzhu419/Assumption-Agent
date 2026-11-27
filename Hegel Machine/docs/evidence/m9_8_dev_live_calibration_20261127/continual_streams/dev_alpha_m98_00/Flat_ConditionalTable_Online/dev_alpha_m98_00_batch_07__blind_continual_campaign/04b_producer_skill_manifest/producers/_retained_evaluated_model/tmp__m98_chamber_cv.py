exec(open('/tmp/m98_probe.py').read().split('for label,z in')[0])
from scipy.optimize import least_squares
for name,z in [('frozen',frozen(g)),('jointTD',m._aligned_features(tr,g)[-len(cy):]),('diagOnly',m._aligned_features(g,g)[:len(cy)])]:
 P,H,_=m._coupon(g,z,c); A,R=m._design(taus,dt,t,(P,H,z)); C=A@theta; r=R@theta; dr=z@theta[2:]; U=G*taus-t[:,None]
 Z=dr[:,None,None]*U; mem=(A[:,:,:2]-H[:,None,None]*G)*theta[:2]; mr=(R[:,:2]-H[:,None]/taus)*theta[:2]; lab=g['lab_id']
 def pred(q):
  if len(q)==2: return C+Z@q,r
  return C+Z@q[:2]+mem@q[2:],r+mr@q[2:]
 def fit(keep,n):
  def res(q):
   p,s=pred(q);return np.r_[((p-cy)[keep]/.006).ravel(),(s-ry)[keep]/.008]
  return least_squares(res,[.62686,.60591]+[0]*(n-2),loss='soft_l1').x
 for mode,n in [('base',0),('archive',0),('fit',2),('memfit',4)]:
  ee=[]; rr=[]; pars=[]
  for l in np.unique(lab):
   mask=lab==l; q=fit(~mask,n) if n else ([.62686,.60591] if mode=='archive' else [0,0]); p,s=pred(q); ee.append((p-cy)[mask]);rr.append((s-ry)[mask]);pars.append(np.round(q,4).tolist())
  rmse=lambda a:np.sqrt(np.mean(np.concatenate(a)**2)); print(name,mode,'pooled',np.round([rmse(ee),rmse(rr)],6),'worst lab',np.round([max(rmse([i]) for i in ee),max(rmse([i]) for i in rr)],6),'fold params',pars,'full',fit(np.ones(len(lab),bool),n).round(6) if n else None)
