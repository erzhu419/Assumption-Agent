import importlib.util, pickle, numpy as np
p='/holdout_task/alpha_m97_c01_batch_04_solver.py'
s=importlib.util.spec_from_file_location('solver',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_alpha_m97_c01_batch_04.pkl','rb'))[0]['dataset']
past,hold,x,e0=m._inputs(d); model=m._Model(d,past,hold,x); tr=d['train']; labs=tr['lab_id']; strata=tr['stratum']; n=len(labs); ids=np.arange(n)
y=np.column_stack((tr['strain_curves_obs']-e0[:n,None],tr['instant_rate_obs']))
print('damage counts by lab',[(int(c),int(np.sum(model.act[:n][labs==c]>0)),int(np.sum(labs==c))) for c in np.unique(labs)])
lt,w=m._fit(model,y,ids); r=model.design(lt,ids)@w-y
print('full tau',np.exp(lt),'weights',w,'residual quantiles',np.quantile(abs(r),[.5,.9,.99,1],axis=0).round(6))
for name, groups in [('lab',labs),('stratum',strata)]:
 pred=np.empty_like(y)
 for c in np.unique(groups):
  mask=groups==c; lt,w=m._fit(model,y,ids[~mask]); pred[mask]=model.design(lt,ids[mask])@w
  print(name,c,'tau',np.exp(lt).round(5),'regime weights',w[-2:].round(6))
 r=pred-y
 for c in np.unique(groups):
  for a in (0,1):
   use=(groups==c)&((model.act[:n]>0)==a)
   if use.any():
    print(name,c,'active',a,'n',use.sum(),'curve/rate RMSE',np.sqrt(np.mean(r[use,:-1]**2)),np.sqrt(np.mean(r[use,-1]**2)),'curve/rate median abs',np.median(abs(r[use,:-1])),np.median(abs(r[use,-1])))
