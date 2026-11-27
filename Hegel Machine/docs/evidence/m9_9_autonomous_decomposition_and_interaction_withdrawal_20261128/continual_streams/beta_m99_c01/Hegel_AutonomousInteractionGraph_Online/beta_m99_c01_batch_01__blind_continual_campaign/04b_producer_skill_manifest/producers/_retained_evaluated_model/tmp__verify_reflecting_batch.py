import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root + 'batch_solver.py')
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)

def dense_axis(k, v, h):
    n = len(k)
    a = np.zeros((n,n))
    for i in range(n):
        if i in (0,n-1):
            j = 1 if i == 0 else n-2
            a[i,j] = -k[i]/h**2
            a[i,i] = -a[i,j]
        else:
            a[i,i-1] = -(k[i]+k[i-1])/(2*h**2)-max(v[i],0)/h
            a[i,i+1] = -(k[i]+k[i+1])/(2*h**2)-max(-v[i],0)/h
            a[i,i] = (k[i-1]+2*k[i]+k[i+1])/(2*h**2)+abs(v[i])/h
    return a

for item in pickle.load(open(root+'public_batch.pkl','rb')):
    d = item['study']
    nx,ny = d['nx'],d['ny']
    ax,ay = [dense_axis(d['kappa_'+a],d['vel_'+a],d['L'+a]/(d['n'+a]-1)) for a in 'xy']
    L = np.kron(np.eye(ny),ax)+np.kron(ay,np.eye(nx))+np.diag(d['V_reaction_2d'].flatten('F'))
    rhs = np.column_stack([u.flatten('F') for u in d['source_modes']])
    fields = np.array([np.linalg.solve(L+mu*np.eye(nx*ny),rhs) for mu in d['mu_shifts']])
    ii,jj = d['sensor_ij'].T
    S = fields[:,ii+nx*jj,:].reshape(-1,rhs.shape[1])
    S_actual,U_actual = solver._forward(d)
    np.testing.assert_allclose(S_actual,S,atol=2e-13,rtol=0)
    cal = d['sensor_calibration_sweeps']
    x,y = cal['ref_rosette'],cal['raw_rosette']
    def calibrate(ref,raw,obs):
        return np.column_stack([(obs[:,b]-c[1])/c[0] for b in range(4) for c in [np.polyfit(ref[:,b],raw[:,b],1)]])
    cv = []
    for f in range(5):
        train = np.arange(len(x))%5 != f
        e = calibrate(x[train],y[train],y[~train])-x[~train]
        cv.append(np.sqrt(np.mean(e**2,axis=0)).max())
    pulse = []
    for p in item.get('campaign_diagnostics') or []:
        obs = solver._calibrate(dict(d,y_obs_raw=p['y_pulse_raw']))
        e = (S@p['w_pulse']-obs).reshape(len(fields),-1)
        pulse.extend(np.sqrt(np.mean(e**2,axis=1)))
    print('worst calibration fold/channel',max(cv),'worst pulse/shift',max(pulse,default=0))
    assert max(cv)<3e-5 and max(pulse,default=0)<1e-5
    z = d['Z_nuisance']; P = np.eye(len(z))-z@np.linalg.solve(z.T@z+d['nuisance_ridge']*np.eye(z.shape[1]),z.T)
    y_cal = calibrate(x,y,d['y_obs_raw'].reshape(-1,4)).ravel()
    np.testing.assert_allclose(solver._calibrate(d),y_cal,atol=2e-14,rtol=0)
    A,y_proj = P@S,P@y_cal
    cap = d['u_box']; n = rhs.shape[1]; ridge = d['source_ridge']; worst = np.zeros(4)
    touched = set()
    for budget in sorted({0., cap/4, cap, 2*cap, d['total_budget'], (n-1)*cap, n*cap-cap/4, n*cap}):
        out = solver.solve_batch(dict(d,total_budget=budget)); w = out['weights']; c = out['kkt_certificate']
        np.testing.assert_array_equal(w,c['weights'])
        lo,hi = np.array(c['lam_lower']),np.array(c['lam_upper'])
        g = A.T@(A@w-y_proj)+ridge*w
        J = ((A@w-y_proj)@(A@w-y_proj)+ridge*(w@w))/2
        assert np.all(lo>=0) and np.all(hi>=0)
        assert abs(c['primal_objective']-J)<1e-12 and J-c['dual_lower_bound']>=-1e-12
        np.testing.assert_allclose(out['terminal_field'],(fields[-1]@w).reshape(nx,ny,order='F'),atol=2e-13,rtol=0)
        errors = [max(abs(w.sum()-budget),-w.min(),w.max()-cap), np.max(abs(g-c['nu_sum']-lo+hi)), max(np.max(abs(lo*w)),np.max(abs(hi*(cap-w)))), abs(J-c['dual_lower_bound'])]
        worst = np.maximum(worst,errors)
        if 0<budget<n*cap:
            touched.update(np.where(w<1e-10,'lower',np.where(w>cap-1e-10,'upper','free')))
    print('worst feasibility/stationarity/complementarity/dual gap',worst,'interior budget active states',sorted(touched))
    assert worst.max()<1e-11 and touched == {'lower','upper','free'}
