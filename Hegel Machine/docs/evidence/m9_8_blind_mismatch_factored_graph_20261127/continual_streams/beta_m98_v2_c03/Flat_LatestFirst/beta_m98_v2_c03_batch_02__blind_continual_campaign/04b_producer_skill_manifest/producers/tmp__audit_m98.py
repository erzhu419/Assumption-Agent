import importlib.util
import itertools
import pickle
import numpy as np

path = '/holdout_task/'
spec = importlib.util.spec_from_file_location('candidate', path+'batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
s = pickle.load(open(path+'public_batch.pkl', 'rb'))[0]['study']
y = m._calibrate(s)
Z = s['Z_nuisance']
K, N = len(s['mu_shifts']), len(s['sensor_ij'])
assert N % 4 == 0 and len(y) == K*N
wall_sensor = ((s['sensor_ij'] == 0) | (s['sensor_ij'] == [s['nx']-1, s['ny']-1])).any(axis=1)
rosette_wall = wall_sensor.reshape(-1,4).any(axis=1)
axis_rows = []
for axis in 'xy':
    k = s['kappa_'+axis]
    h = s['L'+axis]/(len(k)-1)
    robin = s['robin_'+axis]
    expected = np.zeros(len(k)); expected[[0,-1]] = robin/h
    for r, target in [(robin, expected), (0., np.zeros_like(expected))]:
        A = m._axis(k, s['vel_'+axis], h, r)
        axis_rows.append(np.max(abs(A @ np.ones(len(k))-target)))
assert max(axis_rows) < 1e-12
print('worst axis constant-field balance:', max(axis_rows))

def fit_predict(S, test, nuisance):
    train = ~test
    a, b, z = S[train], y[train], Z[train]
    if nuisance:
        q = np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]), z.T)
        a, b = a-z@(q@a), b-z@(q@b)
    w, _ = m._box_budget(a.T@a+s['source_ridge']*np.eye(a.shape[1]), -a.T@b, s['u_box'], s['total_budget'])
    pred = S[test]@w
    if nuisance:
        pred += Z[test]@(q@(y[train]-S[train]@w))
    return float(np.sqrt(np.mean((pred-y[test])**2)))

rosette = np.tile(np.arange(N)//4, K)
shift = np.repeat(np.arange(K), N)
for rx, ry in itertools.product([0,1], repeat=2):
    S, _ = m._forward(dict(s, robin_x=rx*s['robin_x'], robin_y=ry*s['robin_y']))
    for nuisance in ([True,False] if (rx,ry)==(1,1) else [True]):
        er = np.array([fit_predict(S,rosette==i,nuisance) for i in range(N//4)])
        es = np.array([fit_predict(S,shift==i,nuisance) for i in range(K)])
        print('Robin x/y',rx,ry,'profile nuisance',nuisance,
              'rosette pooled/worst',np.linalg.norm(er)/np.sqrt(len(er)),max(er),
              'worst wall-touching',max(er[rosette_wall]),
              'shift pooled/worst',np.linalg.norm(es)/np.sqrt(len(es)),max(es))

# Compare a nested family of shared denominators to the affine calibration.
# Pair sums/differences in an affine numerator are already spanned by its columns.
t=s['sensor_calibration_sweeps']; X=t['ref_rosette']; R=t['raw_rosette']
P=np.column_stack((np.ones(len(X)),X))
q=[X[:,i] for i in range(4)] + [(X[:,i]+sign*X[:,j])/np.sqrt(2) for i,j in itertools.combinations(range(4),2) for sign in (-1,1)]
designs=[P]+[P/(1+a*v[:,None]**2) for v in q for a in (0.001,0.01,0.1,1.)]
cv_affine=[]; cv_nested=[]; choices=[]
for i in range(len(X)):
    train=np.arange(len(X))!=i
    scores=[]
    for D in designs:
        U=np.linalg.qr(D[train],mode='reduced')[0]
        residual=R[train]-U@(U.T@R[train]); leverage=np.sum(U*U,axis=1)
        scores.append(np.mean((residual/(1-leverage[:,None]))**2))
    j=np.argmin(scores); choices.append(j)
    for out, D in [(cv_affine,P),(cv_nested,designs[j])]:
        out.append(D[i]@np.linalg.lstsq(D[train],R[train],rcond=None)[0]-R[i])
for name, e in [('affine',cv_affine),('nested rational family',cv_nested)]:
    print(name,'worst bank RMSE',np.sqrt(np.mean(np.square(e),axis=0)).max())
print('outer calibration folds choosing rational:',np.count_nonzero(choices),'/',len(choices))
