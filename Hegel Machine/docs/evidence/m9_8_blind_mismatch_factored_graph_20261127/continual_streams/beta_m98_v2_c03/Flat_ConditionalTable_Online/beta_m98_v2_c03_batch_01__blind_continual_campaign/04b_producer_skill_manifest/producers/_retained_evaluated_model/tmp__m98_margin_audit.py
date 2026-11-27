import pickle, runpy
import numpy as np
m = runpy.run_path('/holdout_task/batch_solver.py')
for item in pickle.load(open('/holdout_task/public_batch.pkl','rb')):
    s = item['study']; c = s['sensor_calibration_sweeps']
    x, y = c['ref_rosette'], c['raw_rosette']; n = len(x)
    designs = {'diagonal': [np.c_[np.ones(n), x[:, j]] for j in range(4)],
               'affine': [np.c_[np.ones(n), x]] * 4,
               'quadratic': [np.c_[np.ones(n), x, *[x[:, i]*x[:, j] for i in range(4) for j in range(i,4)]]] * 4}
    for name, ds in designs.items():
        rms = []
        for test in np.array_split(np.arange(n), 5):
            tr = np.ones(n, dtype=bool); tr[test] = False
            rms.append(max(np.sqrt(np.mean((D[test]@np.linalg.lstsq(D[tr],y[tr,j],rcond=None)[0]-y[test,j])**2)) for j,D in enumerate(ds)))
        print(name, 'worst held-out sweep block/channel RMSE', max(rms))
    S,_ = m['_forward'](s)
    nx,ny = s['nx'],s['ny']; ix,iy = s['sensor_ij'].T
    wall = (ix==0)|(ix==nx-1)|(iy==0)|(iy==ny-1)
    Z = s['Z_nuisance']; Q = np.linalg.qr(Z)[0]; worst=0
    for pulse in item.get('campaign_diagnostics') or []:
        obs=m['_calibrate'](dict(s,y_obs_raw=pulse['y_pulse_raw']))
        e = S@pulse['w_pulse']-obs
        projected=e-Q@(Q.T@e)
        slice_rmse=[float(np.sqrt(np.mean(r[mask]**2))) for rr in (e,projected) for r in rr.reshape(len(s['mu_shifts']),-1) for mask in (wall,~wall) if mask.any()]
        worst=max(worst,max(slice_rmse)); print('pulse wall/interior RMSE, raw residual then projected', slice_rmse)
    assert worst < 1e-5, worst
    out=m['solve_batch'](s); w=out['weights']; u=out['terminal_field']; mu=s['mu_shifts'][-1]
    def axis(a):
        k=s['kappa_'+a]; v=s['vel_'+a]; N=len(k); h=s['L'+a]/(N-1)
        A=np.zeros((N,N))
        for i in range(N):
            for j in (i-1,i+1):
                if 0<=j<N:
                    A[i,j]=-k[i]/h**2 if i in (0,N-1) else -(k[i]+k[j])/(2*h*h)-max((i-j)*v[i],0)/h
            A[i,i]=-sum(A[i])
        return A
    rhs=np.tensordot(w,s['source_modes'],axes=1)
    r=axis('x')@u+u@axis('y').T+(s['V_reaction_2d']+mu)*u-rhs
    relative=np.linalg.norm(r)/np.linalg.norm(rhs)
    print('independent terminal PDE relative residual',relative)
    assert relative<1e-11
