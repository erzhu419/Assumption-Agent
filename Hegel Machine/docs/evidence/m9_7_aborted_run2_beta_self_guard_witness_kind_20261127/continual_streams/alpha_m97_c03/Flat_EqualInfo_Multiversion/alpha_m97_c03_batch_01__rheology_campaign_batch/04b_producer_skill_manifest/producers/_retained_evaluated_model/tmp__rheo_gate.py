import importlib.util,pickle,numpy as np
from pathlib import Path
p=Path('/holdout_task/alpha_m97_c03_batch_01_solver.py'); s=importlib.util.module_from_spec(importlib.util.spec_from_file_location('solver',p)); exec(compile(p.read_text(),str(p),'exec'),s.__dict__)
i=pickle.load(open('/holdout_task/public_alpha_m97_c03_batch_01.pkl','rb'))[0];d=i['dataset'];gt=i['gt'];tr,te=d['train'],d['test'];g,b=s._calibrate(d['calibration_load_cells']);dt,t=d['past_dt'],d['eval_times'];x,xt=s._align(tr,te); args=(g,b,dt,t,d['crit_stress'],d['shear_banding_index']);lt=np.log(gt['taus']);w=np.r_[gt['theta'],gt['vp_theta']]; labs=tr['lab_id'];y=tr['strain_curves_obs']-tr['strain_0'][:,None];v=tr['instant_rate_obs'];mtr=np.isnan(tr['coupon_features_raw']).any(axis=1);mte=np.isnan(te['coupon_features_raw']).any(axis=1)
raw=p.read_text(); alt={};
for mode in ('linear','log'):
 if mode=='linear': alt[mode]=s._design
 else:
  env={};exec(compile(raw.replace('np.where(arrest[:, None], np.log1p(t), t)','np.log1p(t) if banding <= .50 else t').replace('terminal = np.log1p(t) if banding <= .50 else t','terminal = np.broadcast_to(np.log1p(t) if banding <= .50 else t, (len(x), len(t)))'),str(p),'exec'),env);alt[mode]=env['_design']
for mode, design in alt.items():
 for complete in (False,True):
  yy=np.where((complete&mtr)[:,None],np.nan,y);vv=np.where(complete&mtr,np.nan,v);cv=[]; build,ar=design(tr,x,*args)
  for lab in range(4):
   train=labs!=lab; lt1,w1=s._fit(build,np.where(train[:,None],yy,np.nan),np.where(train,vv,np.nan),dt,t);c,r=build(lt1);e=np.abs(c@w1-y); f=np.abs(r@w1-v)
   for a in (False,True):
    for miss in (False,True):
     m=~train&(ar==a)&(mtr==miss)
     if m.any():cv.append((lab,a,miss,m.sum(),np.median(e[m]),np.quantile(e[m],.95),np.sqrt(np.mean(f[m]**2))))
  lt1,w1=s._fit(build,yy,vv,dt,t);build,ar=design(te,xt,*args);c,r=build(lt1);e=c@w1+te['strain_0'][:,None]-gt['te_clean'];f=r@w1-gt['te_true_rate0'];rmse=lambda v:np.sqrt(np.mean(v*v));print(mode,'fit complete only',complete,'worst complete inactive CV median/p95/rate',np.round(np.max([z[-3:] for z in cv if not z[1] and not z[2]],axis=0),5),'worst CV p95/rate',np.round(np.max([z[-2:] for z in cv],axis=0),5),'public RMSE tau/hered/regime/strain/rate',np.round([rmse(np.exp(lt1)-gt['taus']),rmse(w1[:7]-w[:7]),rmse(w1[7:]-w[7:]),rmse(e),rmse(f)],6),'fit taus',np.exp(lt1));print('public RMSE by active / missing:',[(a,m,round(rmse(e[(ar==a)&(mte==m)]),5),round(rmse(f[(ar==a)&(mte==m)]),5)) for a in (False,True) for m in (False,True)])
 if mode=='log':
  c,r=build(lt); er=gt['te_clean']-te['strain_0'][:,None]-c@w; rr=gt['te_true_rate0']-r@w; dr=np.median(er/c[:,:,2],axis=1);print('reference log complete inactive shape max:',np.max(abs(er/c[:,:,2]-dr[:,None])[~ar&~mte]),'missing rows inferred drift / cell-aligned drift',np.round(np.c_[rr[mte]+xt[mte]@w[3:7],xt[mte]@w[3:7]],4)); print('missing patterns',np.isnan(te['coupon_features_raw'][mte]).astype(int).tolist())
