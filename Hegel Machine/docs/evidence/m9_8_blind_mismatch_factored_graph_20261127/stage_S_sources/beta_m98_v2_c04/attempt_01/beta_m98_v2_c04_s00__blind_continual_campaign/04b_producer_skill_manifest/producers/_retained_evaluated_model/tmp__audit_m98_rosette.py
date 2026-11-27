import pickle
import runpy
import numpy as np

m = runpy.run_path('/holdout_task/batch_solver.py')
s = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['study']
c = s['sensor_calibration_sweeps']
x, y = c['ref_rosette'], c['raw_rosette']
aff = np.c_[np.ones(len(x)), x]
pairs = [(i,j) for i in range(4) for j in range(i+1,4)]
quad = np.column_stack([aff, x*x] + [x[:,i]*x[:,j] for i,j in pairs])
cubic = np.column_stack([quad, x*x*x])
rational = np.column_stack([aff, x/(1+x*x)] + [(x[:,i]+sign*x[:,j])/(1+(x[:,i]-sign*x[:,j])**2) for i,j in pairs for sign in (-1,1)])
for name, d in [('independent affine',aff),('coupled affine',aff),('quadratic',quad),('cubic',cubic),('rational',rational)]:
    e = np.empty_like(y)
    for k in range(5):
        test = np.arange(len(x)) % 5 == k
        for j in range(4):
            dj = d[:,[0,j+1]] if name == 'independent affine' else d
            e[test,j] = dj[test] @ np.linalg.lstsq(dj[~test],y[~test,j],rcond=None)[0] - y[test,j]
    worst = max(np.sqrt(np.mean(e[np.arange(len(x)) % 5 == k]**2, axis=0)).max() for k in range(5))
    print(name, 'OOF RMSE', np.sqrt(np.mean(e*e)), 'worst fold/channel', worst)
for a in 'xy':
    n=s['n'+a]; h=s['L'+a]/(n-1)
    op=m['_axis'](s['kappa_'+a],s['vel_'+a],s['L'+a],s['robin_'+a])
    expected=np.zeros(n); expected[[0,-1]]=s['robin_'+a]/h
    err=np.max(abs(op@np.ones(n)-expected))
    print(a,'row balance max error',err)
    assert err < 1e-11
S,_=m['_forward'](s); y=m['_calibrate'](s); Z=s['Z_nuisance']; sensor=np.asarray(s['sensor_ij']); shifts=s['mu_shifts']; wall=np.any((sensor==0)|(sensor==np.array([s['nx']-1,s['ny']-1])),axis=1)
wall=np.tile(wall,len(shifts)); shift_id=np.repeat(np.arange(len(shifts)),len(sensor))
for name,test in [(f'shift {mu}',shift_id==k) for k,mu in enumerate(shifts)]+[('wall',wall),('interior',~wall)]:
    A,z,target=S[~test],Z[~test],y[~test]
    p=np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T)
    X=A-z@(p@A); b=target-z@(p@target)
    w,_=m['_box_budget'](X.T@X+s['source_ridge']*np.eye(S.shape[1]),-X.T@b,s['u_box'],s['total_budget'])
    gamma=p@(target-A@w)
    e=S[test]@w+Z[test]@gamma-y[test]
    print(name,'held-out RMSE',np.sqrt(np.mean(e*e)))
