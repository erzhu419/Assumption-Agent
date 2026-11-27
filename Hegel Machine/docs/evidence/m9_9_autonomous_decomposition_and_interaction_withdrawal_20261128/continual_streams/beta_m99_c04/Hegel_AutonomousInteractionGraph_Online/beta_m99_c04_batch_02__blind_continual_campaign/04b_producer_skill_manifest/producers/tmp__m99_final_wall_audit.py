import pickle
import sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from batch_solver import solve_batch
from downstream_solver import _design
from _verified_upstream import calibrate_upstream
from scipy.sparse import diags, eye, kron, lil_matrix
from scipy.sparse.linalg import splu

with open('/holdout_task/public_batch.pkl', 'rb') as f:
    items = pickle.load(f)
for item in items:
    s = item['study']
    nx, ny = s['nx'], s['ny']
    ix, iy = s['sensor_ij'].T
    shifts = s['mu_shifts']
    pulses = item['campaign_diagnostics']
    W = np.column_stack([p['w_pulse'] for p in pulses])
    def cal_y(raw, study=s):
        return calibrate_upstream(dict(study, y_obs_raw=raw))['y_obs_raw']
    Y = np.column_stack([cal_y(p['y_pulse_raw']) for p in pulses])
    def axis(name, n, periodic, robin):
        k, v = s['kappa_'+name], s['vel_'+name]
        h = s['L'+name]/(n-1)
        mat = lil_matrix((n,n))
        for i in range(n):
            if not periodic and i in (0,n-1):
                j = 1 if i == 0 else n-2
                rate = k[i]/h**2
                mat[i,j] = -rate
                mat[i,i] = rate + (s['robin_'+name]/h if robin else 0)
            else:
                for step in (-1,1):
                    j = (i+step)%n
                    rate = (k[i]+k[j])/(2*h**2) + max(-step*v[i],0)/h
                    mat[i,j] -= rate
                    mat[i,i] += rate
        if not robin:
            np.testing.assert_allclose(mat @ np.ones(n), 0, atol=1e-12)
        return mat.tocsc()
    identity = eye(nx*ny,format='csc')
    rhs = np.column_stack([m.ravel(order='F') for m in s['source_modes']])
    def build(x,y,robin=False):
        a = kron(eye(ny),axis('x',nx,x,robin),format='csc') + kron(axis('y',ny,y,robin),eye(nx),format='csc') + diags(s['V_reaction_2d'].ravel(order='F'))
        return np.vstack([splu(a+mu*identity).solve(rhs)[ix+nx*iy] for mu in shifts])
    def scores(res):
        z = s['Z_nuisance']
        projected = res - z @ np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T@res)
        def worst(r):
            return np.sqrt(np.mean(r.reshape(len(shifts),-1,4,W.shape[1])**2,axis=1)).max()
        return np.array([worst(res),worst(projected)])
    S = build(True,True)
    np.testing.assert_allclose(S,_design(s)[0],rtol=1e-12,atol=1e-12)
    best = scores(S@W-Y)
    print('periodic both axes, worst pulse x shift x channel RMSE (raw calibrated, nuisance-profiled):',list(best))
    for x,y,robin in [(True,False,False),(False,True,False),(False,False,False),(False,False,True)]:
        alternative = scores(build(x,y,robin)@W-Y)
        print('wall ablation x periodic / y periodic / Robin:',x,y,robin,'worst slice RMSE:',list(alternative))
        assert np.all(alternative > 10*best)
    sw = s['sensor_calibration_sweeps']
    scores_cv = []
    for fold in range(5):
        keep = np.arange(len(sw['ref_rosette']))%5 != fold
        partial = dict(s, sensor_calibration_sweeps=dict(sw, ref_rosette=sw['ref_rosette'][keep], raw_rosette=sw['raw_rosette'][keep]))
        held_pulse_y = np.column_stack([cal_y(p['y_pulse_raw'],partial) for p in pulses])
        scores_cv.append(scores(S@W-held_pulse_y)[0])
    print('Worst tracer slice across five calibration training folds:',max(scores_cv))
    assert max(scores_cv) < 1e-5
    c = solve_batch(s)['kkt_certificate']
    print('Clean-process public primal-dual gap:',c['primal_objective']-c['dual_lower_bound'])
    assert abs(c['primal_objective']-c['dual_lower_bound']) < 1e-8
