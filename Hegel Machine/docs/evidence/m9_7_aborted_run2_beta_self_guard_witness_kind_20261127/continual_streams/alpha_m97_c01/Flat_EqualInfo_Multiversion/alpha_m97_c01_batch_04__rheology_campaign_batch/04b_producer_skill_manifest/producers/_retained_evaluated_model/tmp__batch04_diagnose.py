exec(open('/tmp/batch04_cv.py').read().split("print('damage counts")[0])
from copy import deepcopy
raw=np.concatenate([d[s]['coupon_features_raw'] for s in ('train','test')]); missing=~np.isfinite(raw).all(1)
all_lab=np.concatenate([d[s]['lab_id'] for s in ('train','test')]); all_str=np.concatenate([d[s]['stratum'] for s in ('train','test')]); xlocal=x.copy()
for lab, st in np.unique(np.c_[all_lab,all_str],axis=0):
 mask=(all_lab==lab)&(all_str==st)
 for j in range(x.shape[1]):
  observed=mask & np.isfinite(raw[:,j]); unobserved=mask & ~np.isfinite(raw[:,j]);
  if observed.any(): xlocal[unobserved,j]=x[observed,j].mean()
for name in ('specified','all_log','all_linear','cell_local'):
 q=m._Model(d,past,hold,xlocal if name=='cell_local' else x)
 if name in ('all_log','all_linear'):
  q.flow[:,:-1]=(np.log1p(q.t) if name=='all_log' else q.t)
  q.drift=q.flow[:,:,None]*np.c_[np.ones(len(x)),x][:,None,:]
 summaries=[]
 for groups in (labs,strata):
  pred=np.empty_like(y)
  for c in np.unique(groups):
   mask=groups==c; lt,w=m._fit(q,y,ids[~mask]); pred[mask]=q.design(lt,ids[mask])@w
  r=pred-y
  scores=[]
  for c in np.unique(groups):
   for a in (False,True):
    use=(groups==c)&((q.act[:n]>0)==a)
    if use.any(): scores.append([np.median(abs(r[use,:-1])),np.median(abs(r[use,-1])),np.sqrt(np.mean(r[use,:-1]**2)),np.sqrt(np.mean(r[use,-1]**2))])
  summaries.append(np.max(scores,axis=0).round(5).tolist())
  if name=='specified' and groups is labs:
   for a in (False,True):
    use=(q.act[:n]>0)==a; print('signed median by horizon active=',a,np.median(r[use],axis=0).round(4)); print('tail fraction abs>0.75',np.mean(abs(r[use,:-1])>.75,axis=0).round(3))
   print('largest residual rows',[(int(i),int(labs[i]),int(strata[i]),bool(missing[i]),r[i].round(2).tolist()) for i in np.argsort(np.max(abs(r[:,:-1]),axis=1))[-5:]])
  if groups is labs: print(name,'missing/complete curve MAE',*[round(np.median(abs(r[mask,:-1])),5) for mask in (missing[:n],~missing[:n])])
 print(name,'worst slice median abs curve/rate then RMSE curve/rate, lab then stratum:',summaries)
