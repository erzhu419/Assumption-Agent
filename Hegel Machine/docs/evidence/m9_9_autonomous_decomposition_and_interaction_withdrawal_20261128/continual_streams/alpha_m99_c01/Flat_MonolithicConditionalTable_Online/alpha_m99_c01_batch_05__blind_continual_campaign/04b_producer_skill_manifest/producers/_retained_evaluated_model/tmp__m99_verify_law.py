exec(open('/tmp/m99_chamber.py').read().split('for name,A,V in')[0])
keys=('coupon_features_raw','ref_coupons','lab_id','stratum')
empty={k:g[k][:0] for k in keys}; xgd,_=m._features(g,empty); dg=beta[2]+xgd@beta[3:]; complete=np.isfinite(g['coupon_features_raw']).all(1)
base=np.c_[hg[:,None]*beta[:2],np.zeros(len(hg))]@F.T; Q=Y-base
# Onset-preserving response: (tau*(1-exp(-t/tau))-t)/dt has zero onset slope.
G=np.vstack((f*tau-t[:,None],np.zeros(2))); steady=np.r_[t,1.]
def fit(mask,conserve):
 D=dg[mask,None,None]*(G[None,:,:] if conserve else F[None,:,:]); target=Q[mask]-(dg[mask,None]*steady if conserve else 0); return np.linalg.lstsq(D.reshape(-1,D.shape[-1]),target.ravel(),rcond=None)[0]
def pred(ind,w,conserve): return dg[ind,None]*(steady+G@w if conserve else F@w)+base[ind]
for cons in (False,True):
 for mask in (complete,np.ones(len(hg),bool)):
  w=fit(mask,cons); print('conserve',cons,'nfit',sum(mask),'coef',w,'full err per point by group')
  for sl in [complete,~complete]:
   E=[]
   for i in np.flatnonzero(sl):
    keep=mask.copy(); keep[i]=False; E.append(pred(i,fit(keep,cons),cons)-Y[i]);
   E=np.array(E); print(' n',sum(sl),'rmse/max',np.sqrt(np.mean(E[:,:-1]**2)),np.sqrt(np.mean(E[:,-1]**2)),np.max(abs(E[:,:-1])),np.max(abs(E[:,-1])))
  print('    worst held-lab complete slice',[(l,np.max(abs(E[:,:-1])),np.max(abs(E[:,-1]))) for l in np.unique(g['lab_id']) if np.any(sl:=complete&(g['lab_id']==l)) for E in [pred(sl,fit(mask&~(g['lab_id']==l),cons),cons)-Y[sl]]])
print('diagnostic imputation checks (lab,stratum,imputed chamber drift, measured onset residual)',np.c_[g['lab_id'],g['stratum'],dg,Q[:,-1]][~complete]); print('diagnostic-only x impute center', np.unique(xgd[np.isnan(g['coupon_features_raw'])])); print('train-test x impute center',np.unique(np.vstack([x,xt])[np.isnan(np.vstack([tr['coupon_features_raw'],te['coupon_features_raw']]))])); print('test missing by column',np.isnan(te['coupon_features_raw']).sum(0))
