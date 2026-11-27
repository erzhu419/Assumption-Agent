import pickle,runpy,numpy as np
m=runpy.run_path('/holdout_task/batch_solver.py'); item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; a=s['sensor_calibration_sweeps']; x=a['ref_rosette']; y=a['raw_rosette']; b=a['single_bank_sweeps']; bid=b['bank_id']; p=b['ref_signal']; r=b['raw_readout']; S,_=m['_forward'](s); Xp=np.stack([(S@q['w_pulse']).reshape(-1,4) for q in item['campaign_diagnostics']]); Yp=np.stack([q['y_pulse_raw'].reshape(-1,4) for q in item['campaign_diagnostics']]);
def fit(kind, tr, ts):
 if kind=='full': return np.linalg.lstsq(np.c_[np.ones(tr.sum()),x[tr]],y[tr],rcond=None)[0]
 c=np.zeros((5,4))
 for j in range(4):
  u,v=x[tr,j],y[tr,j]; mask=ts&(bid==j)
  if kind=='joint': u,v=np.r_[u,p[mask]],np.r_[v,r[mask]]
  if kind=='single': u,v=p[mask],r[mask]
  c[[0,j+1],j]=np.linalg.lstsq(np.c_[np.ones(len(u)),u],v,rcond=None)[0]
 return c
for k in ('full','diag','joint','single'):
 e=np.empty_like(y); es=np.empty_like(r)
 for i in range(6):
  h=np.arange(len(x))%6==i; hs=np.arange(len(r))%6==i
  c=fit(k,~h,~hs); e[h]=c[0]+x[h]@c[1:]-y[h]; es[hs]=c[0,bid[hs]]+p[hs]*np.diag(c[1:])[bid[hs]]-r[hs]
 c=fit(k,np.ones(len(x),bool),np.ones(len(r),bool)); E=np.linalg.solve(c[1:].T,(Yp-c[0]).reshape(-1,4).T).T.reshape(Xp.shape)-Xp; E=E.reshape(len(E),len(s['mu_shifts']),-1,4)
 print(k,'worst rosette CV',np.sqrt(np.mean(e*e,axis=0)).max(),'single-bank CV',max(np.sqrt(np.mean(es[bid==j]**2)) for j in range(4)),'pulse-shift',np.sqrt(np.mean(E*E,axis=(2,3))).max(),'bank-pulse-shift',np.sqrt(np.mean(E*E,axis=2)).max())
 print('mean residual per channel',E.mean(axis=(0,1,2)))
