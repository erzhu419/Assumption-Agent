import json, pickle, sys
import numpy as np
from scipy.sparse import lil_matrix, eye
from scipy.sparse.linalg import spsolve
sys.path.insert(0, '/holdout_task')
from batch_solver import solve_batch
from downstream_solver import solve_downstream
from _verified_upstream import calibrate_upstream

item = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]
s = item['study']
pay = json.load(open('/holdout_task/_upstream_param_payload.json'))

def reference(st):
    nx, ny = st['nx'], st['ny']
    A = lil_matrix((nx*ny, nx*ny))
    for j in range(ny):
        for i in range(nx):
            row = i + nx*j
            A[row, row] = st['V_reaction_2d'][i,j]
            for axis, coord, n, stride in [('x',i,nx,1),('y',j,ny,nx)]:
                h = st['L'+axis]/(n-1)
                k, v = st['kappa_'+axis], st['vel_'+axis][coord]
                for direction in (-1,1):
                    neighbor = (coord+direction)%n
                    rate = (k[coord]+k[neighbor])/(2*h*h) + max(-direction*v,0)/h
                    A[row,row+(neighbor-coord)*stride] -= rate
                    A[row,row] += rate
    modes = st['source_modes']
    rhs = np.stack([m.flatten(order='F') for m in modes],axis=1)
    mats = [spsolve(A.tocsc() + shift*eye(nx*ny,format='csc'),rhs) for shift in st['mu_shifts']]
    ij = st['sensor_ij']
    S = np.vstack([u[ij[:,0]+nx*ij[:,1],:] for u in mats])
    U = np.stack([u.reshape(nx,ny,order='F') for u in mats[-1].T])
    return S,U

def verify(st,S,U,y,out):
    Z = st['Z_nuisance']
    whiten = lambda b: b - Z @ np.linalg.solve(Z.T @ Z + st['nuisance_ridge']*np.eye(Z.shape[1]), Z.T @ b)
    S,y = whiten(S),whiten(y)
    w, cert = out['weights'],out['kkt_certificate']
    cap, budget = st['u_box'],st['total_budget']
    nu = cert['nu_sum']
    ll, ul = np.asarray(cert['lam_lower']),np.asarray(cert['lam_upper'])
    r = S @ w - y
    g = S.T @ r + st['source_ridge']*w
    obj = (r@r + st['source_ridge']*(w@w))/2
    np.testing.assert_allclose(out['terminal_field'], np.tensordot(w,U,axes=1), rtol=1e-11, atol=1e-12)
    np.testing.assert_array_equal(cert['weights'],w)
    assert np.all(ll >= 0) and np.all(ul >= 0)
    assert min(w) >= -1e-12 and max(w) <= cap+1e-12
    assert abs(sum(w)-budget)<1e-10
    dual = obj - g@w + nu*budget + cap*np.minimum(g-nu,0).sum()
    assert abs(obj-cert['primal_objective'])<1e-10 and abs(dual-cert['dual_lower_bound'])<1e-10
    errs = (abs(sum(w)-budget),np.max(abs(g-nu-ll+ul)),np.max(abs(np.r_[ll*w,ul*(cap-w)])),abs(obj-dual))
    assert max(errs)<1e-9, errs
    return errs

S,U = reference(s)
cal = calibrate_upstream(s,pay)
y = cal['y_obs_raw']
errs = [verify(s,S,U,y,solve_batch(s))]
res = []
for p in item['campaign_diagnostics']:
    yp = calibrate_upstream(dict(s,y_obs_raw=p['y_pulse_raw']),pay)['y_obs_raw']
    res.append(np.sqrt(np.mean((S@p['w_pulse']-yp).reshape(len(s['mu_shifts']),-1)**2,axis=1)))
print('independent periodic tracer RMSE by pulse and shift:', np.array(res).tolist())
assert np.max(res) < 1e-5

# Perturb the study on the same supported grid: opposite velocity signs,
# different reaction, shifts, and a full affine rosette in downstream coordinates.
t = dict(cal, vel_x=-cal['vel_x'], vel_y=0.4-cal['vel_y'], V_reaction_2d=cal['V_reaction_2d']*1.4, mu_shifts=np.asarray([0.23,0.72,1.34]))
S,U = reference(t)
ref = t['sensor_calibration_sweeps']['ref_rosette']
B = np.array([[1.2,.04,0,0],[-.08,1.02,.05,0],[0,.06,.83,.04],[.1,0,-.04,1.05]])
b = np.array([.11,-.06,.01,.2])
t['sensor_calibration_sweeps'] = {'ref_rosette':ref,'raw_rosette':ref@B+b}
Z=t['Z_nuisance']; P=np.eye(len(S))-Z@np.linalg.solve(Z.T@Z+t['nuisance_ridge']*np.eye(Z.shape[1]),Z.T)
T=P@S
n = S.shape[1]; cap = t['u_box']; rng=np.random.default_rng(912)
for kind in ('interior','mixed','vertex','zero','full'):
    if kind=='interior': w=cap*rng.uniform(.15,.85,n)
    if kind=='mixed': w=cap*np.resize([0,1,.4,.7],n)
    if kind=='vertex': w=cap*(np.arange(n)%2)
    if kind=='zero': w=np.zeros(n)
    if kind=='full': w=np.full(n,cap)
    grad=.025+np.where(w==0,.01,np.where(w==cap,-.02,0))
    y=S@w+np.linalg.lstsq(T.T@P,t['source_ridge']*w-grad,rcond=None)[0]
    st=dict(t,total_budget=w.sum(),y_obs_raw=(y.reshape(-1,4)@B+b).ravel())
    out=solve_downstream(st)
    np.testing.assert_allclose(out['weights'],w,rtol=0,atol=1e-8)
    errs.append(verify(st,S,U,y,out))
print('worst budget, stationarity, complementarity, dual gap:',np.max(errs,axis=0).tolist())
