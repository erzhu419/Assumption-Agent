exec(open('/tmp/batch04_cv.py').read().split("print('damage counts")[0])
print('dataset keys',list(d),'train keys',list(tr),'test keys',list(d['test']))
print('times',model.t,'critical stress',d['crit_stress'])
raw=np.concatenate([d[s]['coupon_features_raw'] for s in ('train','test')]); complete=np.isfinite(raw).all(1)
model.flow[:,:-1]=np.log1p(model.t)
model.drift=model.flow[:,:,None]*np.c_[np.ones(len(x)),x][:,None,:]
for name, eligible in [('all rows',np.ones(n,dtype=bool)),('complete coupon features',complete[:n])]:
 lt,w=m._fit(model,y,ids[eligible]); print(name,'full tau',np.exp(lt).round(6),'weights',w.round(6))
 for gname,groups in [('lab',labs),('stratum',strata)]:
  pred=np.empty_like(y)
  for c in np.unique(groups):
   mask=groups==c; lt,w=m._fit(model,y,ids[eligible & ~mask]); pred[mask]=model.design(lt,ids[mask])@w
  r=pred-y; scores=[]
  for c in np.unique(groups):
   for active in (False,True):
    use=(groups==c)&((model.act[:n]>0)==active)
    if use.any(): scores.append([np.median(abs(r[use,:-1])),np.median(abs(r[use,-1])),np.sqrt(np.mean(r[use,:-1]**2)),np.sqrt(np.mean(r[use,-1]**2))])
  print(name,gname,'worst slice',np.max(scores,axis=0).round(6))
  for desc,use in [('missing',~complete[:n]),('complete inactive',complete[:n]&(model.act[:n]==0)),('complete active',complete[:n]&(model.act[:n]>0))]:
   print(desc,'n',use.sum(),'median abs',np.median(abs(r[use]),axis=0).round(5),'signed median',np.median(r[use],axis=0).round(5))
