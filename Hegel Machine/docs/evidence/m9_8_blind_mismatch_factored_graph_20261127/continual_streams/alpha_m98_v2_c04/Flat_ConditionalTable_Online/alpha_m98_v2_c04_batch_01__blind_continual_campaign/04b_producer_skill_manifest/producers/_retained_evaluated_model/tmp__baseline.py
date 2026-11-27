import importlib.util,pickle,numpy as np
s=importlib.util.spec_from_file_location('base','/tmp/hegel_continual_archive/alpha_m98_v2_c04_s00_solver.py'); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0];d=p['dataset'];tr,te,di=d['train'],d['test'],p['campaign_diagnostics'];dt,t=d['past_dt'],d['eval_times'];n=len(tr['lab_id']);x=m._features(tr,te);br=m._bridge(d['calibration_load_cells']);cal=d['calibration_load_cells'];raw=cal['bridge_raw']; err=np.c_[np.ones(len(raw)),raw]@br-cal['true_channel_loads'];print('bridge',br,'rms',np.sqrt(np.mean(err**2,axis=0)),'max',abs(err).max(axis=0));out=m.solve_batch(d);tau,th=out['estimated_taus'],out['hereditary_weights']; print('material',tau,th)
def diag_features():
 labs=np.r_[tr['lab_id'],te['lab_id']]; refs=np.vstack((tr['ref_coupons'],te['ref_coupons'])); xd=di['coupon_features_raw'].copy(); cells=np.c_[labs,np.r_[tr['stratum'],te['stratum']]]; _,ii,counts=np.unique(cells,axis=0,return_inverse=True,return_counts=True); w=1/counts[ii]; xx=np.vstack((tr['coupon_features_raw'],te['coupon_features_raw']));
 for lab in np.unique(labs):
  q=refs[labs==lab]; a=np.median(q.mean(0)); b=np.median(q.std(0)); xd[di['lab_id']==lab]=(xd[di['lab_id']==lab]-a)/b; xx[labs==lab]=(xx[labs==lab]-a)/b
 return np.where(np.isfinite(xd),xd,np.nansum(w[:,None]*xx,axis=0)/np.sum(w[:,None]*np.isfinite(xx),axis=0))
xd=diag_features();pd,hd=m._loads(di,br); yd=di['strain_curves_obs']-di['strain_0'][:,None]; rd=di['instant_rate_obs'];
for name,b,xx in [('train',tr,x[:n]),('chamber',di,xd)]:
 ps,hs=m._loads(b,br); a,r=m._design(xx,ps,hs,dt,t,tau); ec=(a@th).reshape(-1,len(t))+b['strain_0'][:,None]-b['strain_curves_obs']; er=r@th-b['instant_rate_obs']; print(name,'curve rmse by t',np.sqrt(np.mean(ec**2,0)).round(5),'bias',ec.mean(0).round(5),'rate rmse',np.sqrt(np.mean(er**2))); print('rates by lab',[(int(l),er[b['lab_id']==l].round(5)) for l in np.unique(b['lab_id'])]);
 if name=='chamber': print('diagnostic residual endpoints and rate',np.c_[ec[:,0],ec[:,-1],er].round(5))
