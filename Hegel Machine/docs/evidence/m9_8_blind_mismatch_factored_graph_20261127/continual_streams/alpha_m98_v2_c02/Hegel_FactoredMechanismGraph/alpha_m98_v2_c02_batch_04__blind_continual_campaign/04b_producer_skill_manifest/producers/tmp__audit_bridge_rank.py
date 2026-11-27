import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
from upstream_transducer import _fit,_invert,_design,_POWERS
compact=(_POWERS.sum(axis=1)<=2)|(np.count_nonzero(_POWERS,axis=1)==1)
for item in pickle.load(open('/holdout_task/public_batch.pkl','rb')):
 c=item['dataset']['calibration_load_cells']; x,y=c['true_channel_loads'],c['bridge_raw']; a=_design(x,_POWERS); _,clean=_fit(x,y)
 load=np.linalg.norm(x,axis=1); high=load>=np.quantile(load[clean],.75); mixed=np.count_nonzero(x,axis=1)>1
 for name,held in [('full',np.zeros(len(x),bool)),('upper',high&mixed),('lower',~high&mixed)]:
  at,yt=a[~held],y[~held]; b0=np.linalg.lstsq(at[:,:5],yt,rcond=None)[0]; r=np.linalg.norm(yt-at[:,:5]@b0,axis=1); good=r<=6*np.median(r)+1e-12
  print(name,'train',len(at),'initial accepted',int(good.sum()),'full/compact ranks',np.linalg.matrix_rank(at[good]),np.linalg.matrix_rank(at[good][:,compact]),flush=True)
  b,old=_fit(x[~held],yt); print('original final accepted',int(old.sum()),'full rank',np.linalg.matrix_rank(at[old]),'support',np.count_nonzero(b,axis=0),'forward RMSE on held clean',np.sqrt(np.mean((a[held&clean]@b-y[held&clean])**2)) if held.any() else None,flush=True)
  for _ in range(6):
   ac=at[good][:,compact]; bc=np.linalg.lstsq(ac,yt[good],rcond=None)[0]; r=np.linalg.norm(yt-at[:,compact]@bc,axis=1); update=r<=6*np.median(r)+1e-12
   if np.array_equal(good,update): break
   good=update
  ac=at[good][:,compact]; bc=np.linalg.lstsq(ac,yt[good],rcond=None)[0]
  sigma=np.linalg.norm(yt[good]-ac@bc,axis=0)/np.sqrt(max(1,good.sum()-ac.shape[1])); keep=abs(bc)>6*np.linalg.norm(np.linalg.pinv(ac),axis=1)[:,None]*sigma+1e-12; keep[0]=True
  beta=np.zeros_like(b); inds=np.flatnonzero(compact)
  for k in range(4):
   cols=inds[keep[:,k]]; beta[cols,k]=np.linalg.lstsq(at[good][:,cols],yt[good,k],rcond=None)[0]
  test=held&clean if held.any() else clean
  print('compact final accepted',int(good.sum()),'support',np.count_nonzero(beta,axis=0),'forward RMSE',np.sqrt(np.mean((a[test]@beta-y[test])**2)),'inverse max',np.max(abs(_invert(y[test],beta)-x[test])),flush=True)
