import importlib.util,pickle
import numpy as np
sp=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=i['study']; sw=s['sensor_calibration_sweeps']; X,Y=sw['ref_rosette'],sw['raw_rosette']; S,U=m._forward(s)
print('study keys:', sorted(s)); print('nuisance column norms:',np.linalg.norm(s['Z_nuisance'],axis=0))
for name in ('diagonal','affine','quadratic'):
    q=np.triu_indices(4); pairs=lambda x:x[:,q[0]]*x[:,q[1]]
    def basis(x): return np.c_[np.ones(len(x)),x,pairs(x)] if name=='quadratic' else np.c_[np.ones(len(x)),x]
    def fit(idx):
        B=basis(X[idx]); coef=np.linalg.lstsq(B,Y[idx],rcond=None)[0]
        if name=='diagonal':
            coef[:]=0
            for j in range(4): coef[[0,j+1],j]=np.linalg.lstsq(B[:,[0,j+1]],Y[idx,j],rcond=None)[0]
        return coef
    errors=[]; pulses=[]
    for ids in np.array_split(np.arange(len(X)),5):
        c=fit(np.setdiff1d(np.arange(len(X)),ids)); errors.append(np.sqrt(np.mean((basis(X[ids])@c-Y[ids])**2,axis=0)))
        for p in i['campaign_diagnostics']:
            x=(S@p['w_pulse']).reshape(-1,4); e=basis(x)@c-p['y_pulse_raw'].reshape(-1,4); pulses.append(np.sqrt(np.mean(e.reshape(len(s['mu_shifts']),-1)**2,axis=1)))
    print(name,'held sweep max channel RMS',np.max(errors),'independent tracer max shift RMS in raw units',np.max(pulses))
print('current calibrated pulse worst shift RMS', max(np.max(np.sqrt(np.mean((S@p['w_pulse']-m._calibrate(dict(s,y_obs_raw=p['y_pulse_raw']))).reshape(len(s['mu_shifts']),-1)**2,axis=1))) for p in i['campaign_diagnostics']))
