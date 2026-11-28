exec(open('/tmp/wall_probe.py').read().split('folds=')[0])
import json
log=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))
for r in log:
 a=np.asarray(r['facility_telemetry_trace']); print(r['completed_campaign_short_name'], 'mean and sd of channels 0,5',a[:,[0,5]].mean(0),a[:,[0,5]].std(0))
for k in ('aux_manifold_telemetry','row_bank_id','Z_nuisance'):
 a=np.asarray(s.get(k)); print(k,a.shape,a[:5] if a.ndim else a)
idx=np.arange(len(y)); groups=np.arange(len(y))%4; bank=np.eye(4)[groups][:,:3]-np.eye(4)[groups][:,3:]; z0=z.copy()
for extra in (False,True):
 z=np.column_stack((z0,bank)) if extra else z0
 for modes in [('sealed','sealed'),('periodic','sealed'),('periodic','periodic'),('robin','robin')]:
  S,U=response._responses(s,axes(modes)); w,g=fit(S,idx,ridge); print('bank offset diagnostic',extra,modes,'w',w.round(5),'field range',U.min(),U.max())
  for folds in ([idx[idx//ns==k] for k in range(len(s['mu_shifts']))],[idx[i:i+4].tolist()+idx[i+ns:i+ns+4].tolist()+idx[i+2*ns:i+2*ns+4].tolist() for i in range(0,ns,4)]):
   err=np.zeros_like(y)
   for te in folds:
    tr=np.setdiff1d(idx,te); w,g=fit(S,tr,ridge); err[te]=y[te]-S[te]@w-z[te]@g
   print(' pooled/worst',np.sqrt(np.mean(err**2)),max(np.sqrt(np.mean(err[te]**2)) for te in folds))
