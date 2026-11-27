import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 exec(open('/tmp/pilot_diag.py').read())
lab=pi['lab_id']; ref=pi['ref_coupons']; x=pi['coupon_features_raw'].copy()
for l in np.unique(lab):
 q=ref[lab==l]; x[lab==l]=(x[lab==l]-np.median(q.mean(0)))/np.median(q.std(0))
_,i,n=np.unique(np.c_[lab,pi['stratum']],axis=0,return_inverse=True,return_counts=True); w=1/n[i,None]; mu=np.nansum(w*x,axis=0)/np.sum(w*np.isfinite(x),axis=0)
for name,v in [('pilot balanced',mu),('pilot unweighted',np.nanmean(x,axis=0))]:
 inp=m._inputs(pi,np.where(np.isfinite(x),x,v),c); y,r=m._predict(inp,dt,t,tau,theta); e=pi['strain_curves_obs']-pi['strain_0'][:,None]-y; q=pi['instant_rate_obs']-r
 print(name,'curve abs p50/p95/max',np.quantile(abs(e),[.5,.95,1]),'rate RMSE/max',np.sqrt(np.mean(q*q)),np.max(abs(q)),'worst lab',max(np.sqrt(np.mean(q[lab==l]**2)) for l in np.unique(lab))); print('missing rows/counts rate and late residual:',np.c_[np.where(np.isnan(x))[0],q[np.isnan(x).any(1)],e[np.isnan(x).any(1),-1]])
c0=p['calibration_load_cells']; raw=c0['bridge_raw']; z=c0['true_channel_loads']; print('nonzero counts',np.unique(np.count_nonzero(z,axis=1),return_counts=True)); print('first and last intended loads',z[:8],z[-8:]); print('sweep', {k: np.asarray(v)[:12] for k,v in c0['single_channel_sweeps'].items()});print('telemetry correlations with train observables',np.corrcoef(np.column_stack([tr['aux_chamber_telemetry'],a[0],a[1],a[2][:,1:]]).T)[:4,4:].round(2));print('telemetry first rows',tr['aux_chamber_telemetry'][:3].round(3),te['aux_chamber_telemetry'][:3].round(3))
