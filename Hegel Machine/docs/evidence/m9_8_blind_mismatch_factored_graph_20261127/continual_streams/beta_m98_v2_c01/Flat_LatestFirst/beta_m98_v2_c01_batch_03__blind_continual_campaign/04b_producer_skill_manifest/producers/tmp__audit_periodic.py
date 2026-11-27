import importlib.util
import pickle
import numpy as np

path = '/holdout_task/batch_solver.py'
spec = importlib.util.spec_from_file_location('candidate', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def axis(k, v, h):
    n = len(k)
    a = np.zeros((n,n))
    for i in range(n):
        for step in (-1,1):
            j = (i+step)%n
            c = (k[i]+k[j])/(2*h*h) + max(-step*v[i],0)/h
            a[i,j] -= c
            a[i,i] += c
    return a

for item in pickle.load(open('/holdout_task/public_batch.pkl','rb')):
    s = item['study']; nx, ny = s['nx'],s['ny']
    axes = [axis(s['kappa_'+a], s['vel_'+a], s['L'+a]/(s['n'+a]-1)) for a in 'xy']
    for a in axes:
        np.testing.assert_allclose(a@np.ones(len(a)),0,atol=1e-12)
        assert a[0,-1]<0 and a[-1,0]<0
    L = np.kron(np.eye(ny),axes[0])+np.kron(axes[1],np.eye(nx))+np.diag(s['V_reaction_2d'].ravel(order='F'))
    ij = s['sensor_ij']; ii = ij[:,0]+nx*ij[:,1]; modes = s['source_modes']; rhs = np.column_stack([u.ravel(order='F') for u in modes]); I = np.eye(nx*ny)
    fields = [np.linalg.solve(L+mu*I,rhs) for mu in s['mu_shifts']]
    S = np.vstack([u[ii] for u in fields]); S0,_ = m._forward(s)
    print('independent forward max discrepancy:',np.max(abs(S0-S)))
    np.testing.assert_allclose(S0,S,rtol=0,atol=1e-12)
    edge = (ij[:,0]==0)|(ij[:,0]==nx-1)|(ij[:,1]==0)|(ij[:,1]==ny-1)
    sweeps = s['sensor_calibration_sweeps']; ref,raw = sweeps['ref_rosette'],sweeps['raw_rosette']; design = np.c_[np.ones(len(ref)),ref]
    z = s['Z_nuisance']; p = np.eye(len(z))-z@np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T); A = p@S
    out = m.solve_batch(s); w = out['weights']; cap = s['u_box']; ridge = s['source_ridge']; wt = fields[-1]@w
    np.testing.assert_allclose(out['terminal_field'],wt.reshape((nx,ny),order='F'),rtol=0,atol=1e-12)
    def calibrate(raw_vector,coef):
        return np.linalg.solve(coef[1:].T,(raw_vector.reshape(-1,4)-coef[0]).T).T.ravel()
    cal_err=[]; pulse_errors=[]; gaps=[]
    for fold in range(5):
        test = np.arange(len(ref))%5==fold
        coef = np.linalg.lstsq(design[~test],raw[~test],rcond=None)[0]
        cal_err.extend(np.sqrt(np.mean((design[test]@coef-raw[test])**2,axis=0)))
        for pulse in item.get('campaign_diagnostics') or []:
            residual = (S@pulse['w_pulse']-calibrate(pulse['y_pulse_raw'],coef)).reshape(-1,len(ii))
            pulse_errors.extend(float(e) for mask in (edge,~edge) if mask.any() for e in np.sqrt(np.mean(residual[:,mask]**2,axis=1)))
        y = calibrate(s['y_obs_raw'],coef); g = A.T@(A@w-p@y)+ridge*w
        linear = np.zeros(len(w)); remaining = s['total_budget']
        for i in np.argsort(g):
            linear[i] = min(remaining,cap); remaining -= linear[i]
        gaps.append(g@(w-linear))
    print('worst held-out calibration channel RMSE:',max(cal_err))
    print('worst pulse/shift/spatial-slice RMSE:',max(pulse_errors),'edge sensors:',int(edge.sum()))
    print('worst calibration-fold objective-gap bound:',max(gaps))
    print('nuisance attenuation operator norm:',np.linalg.norm(p@np.linalg.qr(z)[0],2))
    assert max(cal_err)<3e-5 and max(pulse_errors)<2e-5 and max(gaps)<2e-5
