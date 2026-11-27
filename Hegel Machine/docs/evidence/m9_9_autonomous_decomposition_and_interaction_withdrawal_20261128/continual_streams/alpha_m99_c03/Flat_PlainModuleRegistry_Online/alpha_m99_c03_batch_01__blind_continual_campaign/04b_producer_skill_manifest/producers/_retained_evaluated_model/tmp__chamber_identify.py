exec(open('/tmp/chamber_memory.py').read().split("print('zero-drift")[0])
from itertools import combinations
missing=np.isnan(dg['coupon_features_raw']).any(1)
print('missing coupons',np.flatnonzero(missing)); print('test telemetry shape',np.shape(te['aux_chamber_telemetry']))
aux=np.asarray(te['aux_chamber_telemetry']); print('test telemetry mean/std',np.round(aux.mean(0),4),np.round(aux.std(0),4))
T=np.r_[t,1.]; after=np.cumsum(dt[::-1])[::-1]-dt
z=p@(-np.expm1(-dt[:,None]/taus)*np.exp(-after[:,None]/taus))
def features(mode, mask):
 if mode=='train_diag': return m._features(tr,dg)[n:]
 if mode=='train_test_diag':
  pool={k:np.concatenate([te[k],dg[k]]) for k in ('coupon_features_raw','ref_coupons','lab_id','stratum')}; return m._features(tr,pool)[-N:]
 pool=tr if mode=='train' else dg
 x=dg['coupon_features_raw'].copy()
 for l in np.unique(lab):
  panel=pool['ref_coupons'][pool['lab_id']==l]; x[lab==l]=(x[lab==l]-np.median(panel.mean(0)))/np.median(panel.std(0))
 return np.where(np.isfinite(x),x,d[:,1:])
def rms(e): return np.sqrt([np.mean(e[:,:-1]**2),np.mean(e[:,-1]**2)])
for panel in ('train_diag','train_test_diag','train','diag'):
 x=features(panel,None); r=th[2]+x@th[3:];
 print('PANEL',panel,'drift differ max',max(abs(r-d@th[2:])))
 for name,terms,fit_q in [('route',('r',),False),('route_drift',('r',),False),('route_drift_gap',('r','h','own'),False),('route_drift_hist',('r','h','z0','z1'),False),('route_drift_fitq',('r','h'),True),('route_drift_gap_fitq',('r','h','own'),True),('route_drift_offset',('r','1'),False)]:
  def fit(idx):
   def pred(q,ret=False):
    B=basis(q); vals={'r':r,'h':h,'z0':z[:,0],'z1':z[:,1],'1':np.ones(N)}
    F=np.stack([B[None,:,j]*(z[:,j] if f=='own' else vals[f])[:,None] for j in range(2) for f in terms]+([r[:,None]*T] if 'drift' in name else []),axis=-1)
    base=(h[:,None]*th[:2])@B.T; v=np.linalg.lstsq(F[idx].reshape(len(idx)*len(T),-1),(y-base)[idx].ravel(),rcond=None)[0]; e=base+F@v-y
    return (e,v,q) if ret else e[idx].ravel()
   q=least_squares(pred,taus,bounds=(.05,25)).x if fit_q else taus
   return pred(q,True)
  use=ix[~missing]; e,c,q=fit(use); folds=[]
  for l in np.unique(lab):
   held=use[lab[use]==l]; good=use[lab[use]!=l]; folds.append(rms(fit(good)[0][held]))
  print(name,'complete',rms(e[use]).round(5),'missing',rms(e[missing]).round(5),'worst LOLO complete',np.max(folds,0).round(5),'q',q.round(5),'coef',c.round(5))
