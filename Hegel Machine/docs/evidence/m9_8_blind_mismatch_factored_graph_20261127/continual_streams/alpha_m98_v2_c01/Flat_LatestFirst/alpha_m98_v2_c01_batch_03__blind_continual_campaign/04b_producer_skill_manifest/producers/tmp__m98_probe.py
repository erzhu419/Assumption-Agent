import pickle,sys,numpy as np
from importlib.util import spec_from_file_location,module_from_spec
s=spec_from_file_location('arch','/tmp/hegel_continual_archive/alpha_m98_v2_c01_s00_solver.py'); m=module_from_spec(s);s.loader.exec_module(m)
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=item['dataset']; q=item['campaign_diagnostics']; tr,te=d['train'],d['test']; dt,t=d['past_dt'],d['eval_times']; b=m._axial_load_map(d['calibration_load_cells']); x,y=m._features(tr,te); a=m._specimen(tr,x,b); tau,theta=m._fit(a,dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']);
def panel(group):
 z=group['coupon_features_raw'].copy()
 for lab in np.unique(group['lab_id']):
  i=group['lab_id']==lab; p=group['ref_coupons'][i];z[i]=(z[i]-np.median(p.mean(0)))/np.median(p.std(0))
 return z
w=panel(q); w=np.where(np.isfinite(w),w,np.nanmean(np.r_[x,y],axis=0)); c=m._specimen(q,w,b)
def obs(group): return np.c_[group['strain_curves_obs']-group['strain_0'][:,None],group['instant_rate_obs']]
def law(spec, h=dt):
 C,R=m._branches(spec,h,t,tau); f=spec[2]@theta[2:]; return np.c_[C@theta[:2]+f[:,None]*t,R@theta[:2]+f]
if __name__=='__main__':
 print('tau theta bridge',tau,theta,b); B=d['calibration_load_cells']; L=B['true_channel_loads']; raw=B['bridge_raw']; print('bridge residual',np.quantile((raw-b[0])@b[1]-L.sum(1),[0,.1,.5,.9,1])); sw=B['single_channel_sweeps'];
 for lab in np.unique(sw['lab_id']):
  i=sw['lab_id']==lab; print('sweep',lab,np.round(np.c_[sw['cmd_stress'][i],sw['true_stress'][i]],4).tolist())
 for name,g,sp in [('train',tr,a),('diag',q,c)]:
  O=obs(g); E=O-law(sp); print(name,'groups',np.unique(np.c_[g['lab_id'],g['stratum']],axis=0,return_counts=True),'missing',np.isnan(g['coupon_features_raw']).sum(0)); print(name,'residual quantiles by horizon',np.quantile(E,[0,.1,.5,.9,1],axis=0).round(5)); print('stress median ranges',np.quantile(np.c_[sp[0],sp[1]],[0,.5,1],axis=0).round(4));
  for lab in np.unique(g['lab_id']):
   i=g['lab_id']==lab; print(name,lab,'ref',np.median(g['ref_coupons'][i].mean(0)),np.median(g['ref_coupons'][i].std(0)),'mae curve rate',np.abs(E[i,:-1]).mean(),np.abs(E[i,-1]).mean());
  if name=='diag': print('diagnostic rows lab stratum hold residual at first/last/rate',np.c_[g['lab_id'],g['stratum'],sp[1],E[:,[0,-2,-1]]].round(5));
 print('test panel',[(lab,np.median(te['ref_coupons'][te['lab_id']==lab].mean(0)),np.median(te['ref_coupons'][te['lab_id']==lab].std(0))) for lab in np.unique(te['lab_id'])]); print('aux by group',[(k,np.quantile(d[k]['aux_chamber_telemetry'],[0,.5,1],axis=0).round(3)) for k in ('train','test')]);
