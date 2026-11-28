import pickle
import runpy
import numpy as np

m = runpy.run_path('/holdout_task/batch_solver.py')
s = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['study']
y, z = m['calibrate'](s), s['Z_nuisance']
ns = len(s['sensor_ij'])
row = np.arange(len(y))
base = np.repeat([s['robin_x'], s['robin_y']], 2)
patterns = [(0, 0)] + [(f, j) for f in (0.001, 0.003, 0.01, 0.03) for j in range(5)]
models = [m['chamber_response'](s, base*f*(np.eye(4)[j] if j < 4 else np.ones(4)))[0] for f,j in patterns]
for label, groups in [('shift', row//ns), ('rosette', row%ns//4)]:
    scores = []
    for group in np.unique(groups):
        tr = groups != group
        zz = z[tr]
        pz = np.linalg.solve(zz.T@zz+s['nuisance_ridge']*np.eye(z.shape[1]), zz.T)
        st = {**s, 'Z_nuisance': zz}
        fit = []
        for S in models:
            w, cert = m['_inverse'](st, S[tr], y[tr])
            gamma = pz@(y[tr]-S[tr]@w)
            e = y[~tr]-S[~tr]@w-z[~tr]@gamma
            fit.append((cert['primal_objective'], np.mean(e**2)**0.5))
        k = min(range(len(fit)), key=lambda j: fit[j][0])
        scores.append((fit[0][1], fit[k][1]))
        print(label, group, 'training-selected fraction/wall', patterns[k], 'held-out RMSE sealed/selected', scores[-1])
    print(label, 'worst sealed/selected', np.max(scores,axis=0), 'mean sealed/selected', np.mean(scores,axis=0))
