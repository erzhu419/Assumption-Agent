import importlib.util
import itertools
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
item = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]
s = item['study']
sweeps = s['sensor_calibration_sweeps']
x, y = map(np.asarray, (sweeps['ref_rosette'], sweeps['raw_rosette']))
F = m._features(x)
# Pairwise linear combinations up to cubic order are already in this basis.
for i,j in itertools.combinations(range(4),2):
    for sign in (-1, 1):
        for power in (1,2,3):
            target = (x[:,i] + sign*x[:,j])**power
            np.testing.assert_allclose(F@np.linalg.lstsq(F,target,rcond=None)[0],target,atol=1e-11)
# Test denominator perturbations along measured banks and bank sums/differences.
Q = np.column_stack([*x.T, *(x[:,i]+sign*x[:,j] for i,j in itertools.combinations(range(4),2) for sign in (-1,1))])
Q2 = Q**2
ix = np.arange(len(x))
# Separate grossly corrupted rows by instrument magnitude, independent of fold fits.
gross = np.max(np.abs(y),axis=1)>8

def fit(train):
    return m._fit_calibration(x[train], y[train])

def extras(train,C):
    active = np.linalg.norm(F[train]@C-y[train],axis=1)<.002
    columns = np.flatnonzero(train)[active]
    return np.linalg.lstsq(F[columns], (y[columns,:,None]*Q2[columns,None,:]).reshape(len(columns),-1),rcond=None)[0].reshape(F.shape[1],4,-1)

# Candidate 0 is the actual solver. Other candidates are fixed denominator
# perturbations with a full cubic numerator correction fitted on training rows.
strength = np.array([1e-6,1e-5,1e-4,1e-3,.01,.1])
def predict(test,C,E):
    base = F[test]@C
    delta = np.einsum('np,pcq->ncq',F[test],E)
    alt=(base[:,:,None,None]+delta[:,:,:,None]*strength)/(1+Q2[test,None,:,None]*strength)
    return np.concatenate([base[:,:,None],alt.reshape(len(base),4,-1)],axis=2)

err = []; gains=[]; selections=[]
for fold in range(5):
    test = ix%5==fold
    scores = 0
    for inner in range(4):
        itest = ~test & ((ix//5)%4==inner)
        train = ~(test | itest)
        C=fit(train)
        pred=predict(itest & ~gross,C,extras(train,C))
        scores += ((pred-y[itest & ~gross,:,None])**2).sum(axis=(0,1))
    best = int(np.argmin(scores)); selections.append(best)
    C=fit(~test)
    mask=test & ~gross
    pred=predict(mask,C,extras(~test,C))
    e=np.linalg.norm(pred[:,:, [0,best]]-y[mask,:,None],axis=1)
    err.append(e); gains.append(np.sqrt(np.mean(e**2,axis=0)).tolist())
print('gross sweep rows:', int(gross.sum()), 'nested choices (0=cubic):', selections)
print('outer sweep fold RMSE [cubic, selected]:', gains)
print('worst clean outer row [cubic, selected]:', np.vstack(err).max(axis=0).tolist())

S,_=m._forward(s); Z=np.asarray(s['Z_nuisance']); ij=np.asarray(s['sensor_ij']); nshifts=len(s['mu_shifts']); nr=len(ij)
# Fit nuisance only on the other sensor sites; evaluate each held site by shift.
worst=0.
for p in item['campaign_diagnostics']:
    residual=m._calibrate(dict(s,y_obs_raw=p['y_pulse_raw']))-S@p['w_pulse']
    sites=np.tile(np.arange(nr),nshifts)
    for j in range(nr):
        test=sites==j; train=~test
        g=np.linalg.lstsq(Z[train],residual[train],rcond=None)[0]
        err_sensor=residual[test]-Z[test]@g
        worst=max(worst,float(np.max(abs(err_sensor))))
print('worst held-out sensor/pulse/shift absolute error:',worst)
assert worst < 2e-5
print('study keys:', sorted(s))
