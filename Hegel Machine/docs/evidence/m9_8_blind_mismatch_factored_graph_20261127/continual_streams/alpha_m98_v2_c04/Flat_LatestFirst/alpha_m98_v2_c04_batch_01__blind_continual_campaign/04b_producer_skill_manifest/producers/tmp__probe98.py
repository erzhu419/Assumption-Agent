import importlib.util,pickle,numpy as np
p='/tmp/hegel_continual_archive/alpha_m98_v2_c04_s00_solver.py'
s=importlib.util.spec_from_file_location('prev',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; a,b,c=d['train'],d['test'],i['campaign_diagnostics']; dt,t=d['past_dt'],d['eval_times']; cal=d['calibration_load_cells']; raw=cal['bridge_raw']; y=cal['true_channel_loads']; n=len(a['lab_id'])
print('sweeps',cal['single_channel_sweeps']); print('cal range',raw.min(0),raw.max(0))
for degree in [1,2,3]:
 from itertools import combinations_with_replacement
 A=np.column_stack([np.ones(len(raw))]+[np.prod(raw[:,idx],axis=1) for k in range(1,degree+1) for idx in combinations_with_replacement(range(4),k)])
 inv=np.linalg.pinv(A); e=(y-A@(inv@y))/(1-(A*inv.T).sum(1))[:,None]; print('bridge LOO degree',degree,'RMS',np.sqrt(np.mean(e**2,0)),'total RMS',np.sqrt(np.mean(e.sum(1)**2)))
B=m._bridge(cal); x=m._features(a,b)[:n]; tau,th=m._fit(x,*m._loads(a,B),dt,t,a['strain_curves_obs']-a['strain_0'][:,None],a['instant_rate_obs']); print('tau theta',tau,th)
for block,X in [(a,x),(c,m._features(a,c)[n:])]:
 P,H=m._loads(block,B); Y=block['strain_curves_obs']-block['strain_0'][:,None]; R=block['instant_rate_obs']; A,D=m._design(X,P,H,dt,t,tau); E=Y-(A@th).reshape(Y.shape); F=R-D@th; print('residual rms time',np.sqrt(np.mean(E**2,0)),'rate',np.sqrt(np.mean(F**2)),'median abs',np.median(abs(E)),np.median(abs(F))); print('by lab',[(int(l),np.sqrt(np.mean(E[block['lab_id']==l]**2)),np.sqrt(np.mean(F[block['lab_id']==l]**2))) for l in np.unique(block['lab_id'])]); print('row residual start/mid/end/rate',np.round(np.column_stack([E[:,[0,5,-1]],F])[:18],6))
 drift=th[2]+X@th[3:]; basis=np.column_stack([-np.expm1(-t[:,None]/tau),t]); coeff=np.linalg.lstsq(basis,Y.T,rcond=None)[0].T; print('temporal fit RMS',np.sqrt(np.mean((Y-coeff@basis.T)**2)), 'rate discrepancy',np.sqrt(np.mean((R-coeff@np.r_[1/tau,1])**2))); print('baseline load ranges',P.min(),P.max(),H.min(),H.max()); print('drift range',drift.min(),drift.max())
