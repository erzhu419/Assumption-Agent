import importlib.util,pickle,numpy as np
np.set_printoptions(precision=6,suppress=True)
p='/tmp/hegel_continual_archive/beta_m99_c03_s00_solver.py'
sp=importlib.util.spec_from_file_location('prior',p); prior=importlib.util.module_from_spec(sp);sp.loader.exec_module(prior)
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; pulses=item['campaign_diagnostics']; cal=s['sensor_calibration_sweeps']; ref=cal['ref_rosette']; raw=cal['raw_rosette']; C=np.c_[np.ones(len(ref)),ref]; coef=np.linalg.lstsq(C,raw,rcond=None)[0]; dr=raw-C@coef
print('affine coef',coef,'cal rms',np.sqrt(np.mean(dr**2,axis=0)), 'max',np.max(abs(dr),axis=0))
features=[ref[:,i]*ref[:,j] for i in range(4) for j in range(i,4)]; C2=np.column_stack([C]+features); print('quadratic coefficients',np.linalg.lstsq(C2,raw,rcond=None)[0][5:].T); print('LOO rms linear/quadratic',*[np.sqrt(np.mean(((raw-X@np.linalg.lstsq(X,raw,rcond=None)[0])/(1-np.sum(np.linalg.qr(X)[0]**2,axis=1))[:,None])**2,axis=0)) for X in [C,C2]])
bank=cal['single_bank_sweeps']; print('single-bank keys',list(bank));
for i in range(4):
 ix=bank['bank_id']==i; print('bank',i,np.polynomial.polynomial.polyfit(bank['ref_signal'][ix],bank['raw_readout'][ix],2))
print('row ordering',s['row_bank_id'][:32], 'sensor_ij',s['sensor_ij']); S,U=prior._forward(s); W=np.column_stack([p['w_pulse'] for p in pulses]); R=np.column_stack([p['y_pulse_raw'] for p in pulses]); Y=np.column_stack([prior._calibrated_observations({**s,'y_obs_raw':r}) for r in R.T]); E=Y-S@W
print('pulses',W.T); print('pulse RMS',np.sqrt(np.mean(E**2,axis=0)),'raw calibrated ranges',Y.min(),Y.max(),'forward ranges', (S@W).min(),(S@W).max());print('error by row',E[:32]); print('pulse affinity with nuisance',np.sqrt(np.mean((E-s['Z_nuisance']@np.linalg.lstsq(s['Z_nuisance'],E,rcond=None)[0])**2,axis=0)))
