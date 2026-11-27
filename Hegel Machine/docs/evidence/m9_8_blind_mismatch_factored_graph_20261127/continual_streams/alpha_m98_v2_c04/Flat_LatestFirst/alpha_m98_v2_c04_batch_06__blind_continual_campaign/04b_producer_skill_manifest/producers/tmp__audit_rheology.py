import importlib.util, pickle, numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=p['dataset']; c=d['calibration_load_cells']; x=c['true_channel_loads']; y=c['bridge_raw']; b=m._fit_bridge(c); A=m._bridge_basis(x)
print('calibration support',x.shape,np.round(np.min(x,axis=0),4),np.round(np.max(x,axis=0),4)); print('forward residual percentiles',np.percentile(np.linalg.norm(A@b-y,axis=1),[50,80,90,95,99,100])); print('sweep fields',list(c['single_channel_sweeps']) if isinstance(c['single_channel_sweeps'],dict) else np.shape(c['single_channel_sweeps']))
D=np.column_stack([x[:,i]+s*x[:,j] for i,j in m._PAIRS for s in (1,-1)])
P=np.column_stack([x[:,i]*x[:,j]*x[:,k] for i,j,k in [(0,1,2),(0,1,3),(0,2,3),(1,2,3)]])
E=np.column_stack((x*x,D**3,P)); Q=np.column_stack((x,D)); Q=Q/np.maximum(np.std(Q,axis=0),1e-12)
blocks=np.arange(len(x))//max(1,len(x)//5); blocks=np.minimum(blocks,4)
for fold in np.unique(blocks):
 tr=blocks!=fold; va=~tr; base=m._fit_bridge({'true_channel_loads':x[tr],'bridge_raw':y[tr]}); et=np.linalg.norm(A[tr]@base-y[tr],axis=1); keep=tr & (np.linalg.norm(A@base-y,axis=1)<8*max(np.median(et),1e-8)); cutoff=8*max(np.median(et),1e-8); normal=va & (np.linalg.norm(A@base-y,axis=1)<cutoff)
 mats=[('baseline',A),('expanded polynomial',np.c_[A,E])]; candidates=[]
 for j in range(Q.shape[1]):
  for a in (.05,.2,1.):
   B=A/(1+a*Q[:,j,None]**2); C=np.c_[A,B[:,1:5]]; co=np.linalg.lstsq(C[keep],y[keep],rcond=None)[0]; candidates.append((np.median(np.linalg.norm(C[keep]@co-y[keep],axis=1)),j,a,C))
 _,j,a,C=min(candidates,key=lambda z:z[0]); mats.append((f'profiled rational {j}/{a}',C))
 print('sweep block',fold,'inliers',normal.sum(),'of',va.sum())
 for label,C in mats:
  co=np.linalg.lstsq(C[keep],y[keep],rcond=None)[0]; e=np.linalg.norm(C@co-y,axis=1); print(label,'median all',np.median(e[va]),'p90 ordinary',np.percentile(e[normal],90) if normal.any() else np.nan)
tr,te=d['train'],d['test']; f=m._features(tr,te); n=len(tr['lab_id']); out=m.solve_batch(d); t=d['eval_times']; dt=d['past_dt']; tau=out['estimated_taus']; th=out['hereditary_weights']; a,r=m._design(f[:n],*m._loads(tr,b),dt,t,tau); ec=(a@th).reshape(n,-1)-(tr['strain_curves_obs']-tr['strain_0'][:,None]); er=r@th-tr['instant_rate_obs']; mask=np.isnan(tr['coupon_features_raw']).any(axis=1)
for lab in np.unique(tr['lab_id']):
 for miss in (False,True):
  ix=(tr['lab_id']==lab)&(mask==miss)
  if ix.any(): print('benchtop lab/missing/n',lab,miss,ix.sum(),'signed curve/rate median',np.round(np.median(ec[ix]),5),np.round(np.median(er[ix]),5),'rate median absolute',np.round(np.median(abs(er[ix])),5))
