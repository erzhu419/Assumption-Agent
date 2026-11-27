import importlib.util,pickle,numpy as np
p='/tmp/hegel_continual_archive/dev_alpha_m98_00_batch_02_solver.py';s=importlib.util.spec_from_file_location('arc',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0];d=i['dataset'];a=i['campaign_diagnostics'];tr,te=d['train'],d['test'];dt,t=d['past_dt'],d['eval_times'];ct,ce=m._prepare(d);tau,b=m._fit(dt,t,ct,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']);print('intrinsic',tau,b)
lab=np.r_[tr['lab_id'],te['lab_id']];ys=np.vstack([tr['ref_coupons'],te['ref_coupons']]);x=a['coupon_features_raw'].copy();xjoint=m._aligned_features(tr,te)[:,1:];raw=np.vstack([tr['coupon_features_raw'],te['coupon_features_raw']]);_,ix,cs=np.unique(np.c_[lab,np.r_[tr['stratum'],te['stratum']]],axis=0,return_inverse=True,return_counts=True);w=1/cs[ix,None];fill=(w*xjoint).sum(axis=0)/w.sum();
for j in np.unique(lab):
 ref=ys[lab==j];sel=a['lab_id']==j;x[sel]=(x[sel]-np.median(ref.mean(axis=0)))/np.median(ref.std(axis=0));
x=np.c_[np.ones(len(x)),np.where(np.isnan(x),fill,x)];cal=m._stress_calibration(d['calibration_load_cells']);ps=a['past_channel_raw']@cal[:-1]+cal[-1];h=a['hold_channel_raw']@cal[:-1]+cal[-1];co=ps,h,x;v=x@b[2:];y=np.c_[a['strain_curves_obs']-a['strain_0'][:,None],a['instant_rate_obs']];labs=a['lab_id'];F=np.c_[np.vstack([-np.expm1(-t[:,None]/tau),1/tau]),np.r_[t,1.]];amps=np.linalg.lstsq(F,y.T,rcond=None)[0].T

def err(name,pred):
 e=pred-y;print(name,'pooled',np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2)),'worst lab',[(round(np.sqrt(np.mean(e[labs==j,:-1]**2)),6),round(np.sqrt(np.mean(e[labs==j,-1]**2)),6)) for j in np.unique(labs)])
err('intrinsic-time free amp',amps@F.T);D,R=m._design(tau,dt,t,co);err('baseline',np.c_[D@b,R@b]);P,r=m._deployment(tau,b,dt,t,co);err('batch02',np.c_[P,r]);ad=np.c_[h*b[0]+.34880024*v,h*b[1]+1.63895287*v,v*(1-.34880024/tau[0]-1.63895287/tau[1])];err('batch01',ad@F.T)
def states(r,times=dt):
 z=np.zeros((len(h),2));
 for step,p in zip(times,r.T):z+=-np.expm1(-step/tau)*(p[:,None]-z)
 return z
z=states(ps);zr=states(ps[:,::-1]);print('amp ranges',np.percentile(amps,[0,50,100],axis=0),'drift range',v.min(),v.max())
for name,X in [('hold+drift',np.c_[h,v]),('hold+drift+z',np.c_[h,v,z]),('hold+drift+z+reverse',np.c_[h,v,z,zr]),('hold+features+past',np.c_[h,x,ps])]:
 p=0*y
 for j in np.unique(labs):
  mask=labs!=j;B=np.linalg.lstsq(X[mask],amps[mask],rcond=None)[0];p[~mask]=X[~mask]@B@F.T
 print(name,'fit',np.round(np.linalg.lstsq(X,amps,rcond=None)[0],5));err(name,p)
print('first 8: hold, drift, z, free amp');print(np.round(np.c_[h,v,z,amps][:8],5))
