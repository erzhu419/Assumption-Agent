import importlib.util, pickle
import numpy as np
p='/holdout_task/batch_solver.py'
spec=importlib.util.spec_from_file_location('candidate',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
for item in pickle.load(open('/holdout_task/public_batch.pkl','rb')):
 s=item['study']; sw=s['sensor_calibration_sweeps']; x=sw['ref_rosette']; y=sw['raw_rosette']; A=m._features(x)
 names=('full','compact','rational_005','rational_020'); errors={k:np.zeros_like(y) for k in names}; keep=np.zeros(len(x),dtype=bool); supports=[]
 def rational(q,a):
  F=m._features(q)
  for j in range(4): F[:,5+m._TERMS.index((j,j,j))] /= 1+a*q[:,j]**2
  return F
 for te in np.array_split(np.random.default_rng(13).permutation(len(x)),10):
  tr=np.setdiff1d(np.arange(len(x)),te); c=m._fit_calibration(x[tr],y[tr]); fiterr=np.linalg.norm(y[tr]-A[tr]@c,axis=1); clean=tr[fiterr<.002]; keep[te]=np.linalg.norm(y[te]-A[te]@c,axis=1)<.002; masks=abs(c)>.001; supports.append(int(masks.sum())); errors['full'][te]=A[te]@c-y[te]
  for name,F in [('compact',A),('rational_005',rational(x,.05)),('rational_020',rational(x,.20))]:
   for j,mask in enumerate(masks.T):
    coef=np.linalg.lstsq(F[clean][:,mask],y[clean,j],rcond=None)[0]; errors[name][te,j]=F[te][:,mask]@coef-y[te,j]
 print('paired calibration:',int((~keep).sum()),'OOF gross outliers; compact support sizes',sorted(set(supports)))
 for name,e in errors.items(): print(name,'clean OOF RMSE per channel',np.sqrt(np.mean(e[keep]**2,axis=0)),'worst absolute',np.max(abs(e[keep])))
 b=sw['single_bank_sweeps']; banks=b['bank_id']; bx=np.eye(4)[banks]*b['ref_signal'][:,None]; bankerr=(m._features(bx)@m._fit_calibration(x,y))[np.arange(len(bx)),banks]-b['raw_readout']; print('independent single-bank max after excluding gross outliers',np.max(abs(bankerr[abs(bankerr)<.002])))
 yy=m._calibrate(s); c=m._fit_calibration(x,y); print('observed rosette inverse max raw error',np.max(abs(m._features(yy.reshape(-1,4))@c-s['y_obs_raw'].reshape(-1,4))))
 S,_=m._forward(s); Z=s['Z_nuisance']; sr=s['source_ridge']; rr=s['nuisance_ridge']; cap=s['u_box']; B=s['total_budget']; ids=np.arange(len(yy)); byshift=ids.reshape(len(s['mu_shifts']),len(s['sensor_ij'])); bank=s['row_bank_id'].ravel();
 for label,groups in [('shift',byshift),('sensor',byshift.T),('bank',[ids[bank==k] for k in np.unique(bank)])]:
  for profile in (False,True):
   rm=[]
   for te in groups:
    tr=np.setdiff1d(ids,te); D=S[tr]; t=yy[tr]; z=Z[tr]; P=np.linalg.solve(z.T@z+rr*np.eye(z.shape[1]),z.T) if profile else np.zeros((z.shape[1],len(tr))); d=D-z@(P@D); r=t-z@(P@t); w,_=m._box_budget(d.T@d+sr*np.eye(d.shape[1]),-d.T@r,cap,B); res=yy[te]-S[te]@w-Z[te]@(P@(t-D@w)); rm.append(float(np.sqrt(np.mean(res**2))))
   print(label,'profiled' if profile else 'unprofiled','worst RMSE',max(rm))
 out=m.solve_batch(s); kc=out['kkt_certificate']; print('isolated solve: budget residual',abs(out['weights'].sum()-B),'KKT gap',abs(kc['primal_objective']-kc['dual_lower_bound']))
