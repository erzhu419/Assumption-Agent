import sys, pickle, json, itertools
sys.path.insert(0, '/holdout_task')
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from _verified_upstream import calibrate_upstream
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
with open('/holdout_task/_upstream_param_payload.json') as f:
    payload = json.load(f)
s = item['study']
pulses = item['campaign_diagnostics']
W = np.stack([p['w_pulse'] for p in pulses], axis=1)
Y = np.stack([calibrate_upstream({**s, 'y_obs_raw': p['y_pulse_raw']}, payload)['y_obs_raw'] for p in pulses], axis=1)

def axis(k, v, h, robin, wall):
    n = len(k)
    face = (k[:-1] + k[1:]) / (2*h*h)
    lo, hi = -face.copy(), -face.copy()
    lo[:-1] -= np.maximum(v[1:-1], 0)/h
    hi[1:] -= np.maximum(-v[1:-1], 0)/h
    loss = np.zeros(2)
    wrap = np.zeros(2)
    if wall == 'periodic':
        hi[0] -= max(-v[0], 0)/h
        lo[-1] -= max(v[-1], 0)/h
        wrap[:] = (k[0]+k[-1])/(2*h*h) + np.maximum([v[0], -v[-1]], 0)/h
    else:
        hi[0], lo[-1] = -k[0]/(h*h), -k[-1]/(h*h)
        if wall == 'robin':
            loss[:] = robin/h
    diag = np.r_[-hi[0]+wrap[0]+loss[0], -lo[:-1]-hi[1:], -lo[-1]+wrap[1]+loss[1]]
    a = sparse.diags((lo,diag,hi), (-1,0,1), format='csc')
    return a + sparse.csc_matrix((-wrap, ([0,n-1],[n-1,0])), shape=(n,n)) if wall == 'periodic' else a

nx, ny = s['nx'], s['ny']
reaction = s['V_reaction_2d'].ravel(order='F')
rows = s['sensor_ij'] @ [1,nx]
rhs = np.stack([a.ravel(order='F') for a in s['source_modes']], axis=1)
z = s['Z_nuisance']

def predict(ax, ay):
    A = sparse.kron(sparse.eye(ny),ax,format='csc') + sparse.kron(ay,sparse.eye(nx),format='csc')
    return np.vstack([splu(A+sparse.diags(reaction+mu,format='csc')).solve(rhs @ W)[rows] for mu in s['mu_shifts']])

def axes(x, y):
    return [axis(s['kappa_'+a], s['vel_'+a],s['L'+a]/(s['n'+a]-1), s['robin_'+a], wall) for a,wall in zip('xy',(x,y))]

if __name__ == '__main__':
    print('Observed pulse RMS', np.sqrt(np.mean(Y**2,axis=0)))
    for x,y in itertools.product(('robin','reflect','periodic'),repeat=2):
        e = predict(*axes(x,y))-Y
        ez = e-z@np.linalg.lstsq(z,e,rcond=None)[0]
        print(x,y,'pulse RMSE',np.round(np.sqrt(np.mean(e**2,axis=0)),8),'profiled',np.round(np.sqrt(np.mean(ez**2,axis=0)),8),'worst shift-pulse RMS',np.sqrt(np.max(np.mean(e.reshape(3,-1,3)**2,axis=1))))
    log = json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))
    for row in log:
        t = np.array(row['facility_telemetry_trace'])
        print(row['completed_campaign_short_name'],'mean', np.round(t.mean(0),3),'std',np.round(t.std(0),3))
