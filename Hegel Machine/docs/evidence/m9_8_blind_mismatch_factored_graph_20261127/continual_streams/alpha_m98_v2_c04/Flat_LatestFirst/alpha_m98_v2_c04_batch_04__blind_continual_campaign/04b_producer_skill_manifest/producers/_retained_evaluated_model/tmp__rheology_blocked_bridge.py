import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
d = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['dataset']
c = d['calibration_load_cells']
x, y = c['true_channel_loads'], c['bridge_raw']
A = m._bridge_basis(x)
axis = np.count_nonzero(x, axis=1) == 1
folds = [(f'axis {j}', axis & (x[:,j] != 0)) for j in range(4)]
radius = np.linalg.norm(x, axis=1)
folds.extend([('upper mixed-load band', (~axis) & (radius >= np.quantile(radius[~axis], .8)))])
for name, test in folds:
    tr = ~test
    b = m._fit_bridge({'true_channel_loads':x[tr], 'bridge_raw':y[tr]})
    e = np.linalg.norm(A@b-y, axis=1)
    ordinary = e < .01
    # Infer the sparse response independently in each training fold. The
    # 0.002 cutoff separates the observed order-0.1 terms from sensor noise.
    bs = b.copy()
    for j in range(4):
        use = abs(b[:,j]) > .002
        bs[:,j] = 0
        bs[use,j] = np.linalg.lstsq(A[tr & ordinary][:,use], y[tr & ordinary,j], rcond=None)[0]
    v = test & ordinary
    print(name, 'ordinary/total', v.sum(), test.sum(), 'min rejected forward residual', min(e[test & ~ordinary], default=float('nan')))
    for label, coef in [('current', b), ('sparse', bs)]:
        delta = m._decode_bridge(y[v], coef)-x[v]
        z = delta.sum(1)
        print(label, 'axial RMSE/max', np.sqrt(np.mean(z*z)), max(abs(z)), 'max channel error', np.max(abs(delta)))
