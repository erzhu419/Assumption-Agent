import pickle, runpy
import numpy as np
from scipy.optimize import least_squares
m = runpy.run_path('/holdout_task/batch_solver.py')
with open('/holdout_task/public_batch.pkl','rb') as f:
    item = pickle.load(f)[0]
s = item['study']; sw = s['sensor_calibration_sweeps']; x,y = sw['ref_rosette'],sw['raw_rosette']
for axis in ('x','y'):
    k=s['kappa_'+axis]; h=s['L'+axis]/(len(k)-1)
    A=m['_axis'](k,s['vel_'+axis],h)
    err=np.max(np.abs(A@np.ones(len(k))))
    print(axis,'constant-state stencil residual',err)
    assert err < 1e-12
# Compare the saved model's calibration family to a predeclared saturation
# model, without using tracer responses or held-out sweep rows to fit either.
E=np.empty((2,*y.shape))
for i in range(5):
    test=np.arange(len(x))%5==i
    for j in range(4):
        z=x[:,j]; X=np.c_[np.ones(len(z)),z]
        b,g=np.linalg.lstsq(X[~test],y[~test,j],rcond=None)[0]
        E[0,test,j]=b+g*z[test]-y[test,j]
        def model(p,z):
            return p[0]+p[1]*z/(1+p[2]*z*z)
        p=least_squares(lambda p: model(p,z[~test])-y[~test,j],(b,g,0),bounds=([-np.inf,-np.inf,0],np.inf),gtol=1e-13).x
        E[1,test,j]=model(p,z[test])-y[test,j]
print('held-out sweep RMSE by channel (affine, saturating):',np.sqrt(np.mean(E**2,axis=1)).tolist())
S,_=m['_forward'](s); Z=s['Z_nuisance']; P=np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T)
E=np.array([m['_calibrate'](dict(s,y_obs_raw=p['y_pulse_raw']))-S@p['w_pulse'] for p in item['campaign_diagnostics']]); banks=s['row_bank_id']; block=np.repeat(np.arange(len(s['mu_shifts'])),len(s['sensor_ij']))
for name,V in [('direct',E),('projected',E-(E@P.T)@Z.T)]:
    worst=max(np.sqrt(np.mean(V[:,(banks==b)&(block==k)]**2,axis=1)).max() for b in np.unique(banks) for k in np.unique(block) if np.any((banks==b)&(block==k)))
    print(name,'worst tracer bank/pulse/shift RMSE',worst)
    assert worst<5e-6
