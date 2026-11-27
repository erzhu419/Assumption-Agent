import numpy as np
from pathlib import Path

# Reuse the exploratory forward model, without running its parameter fits.
p = Path('/tmp/probe_wall.py')
exec(compile(p.read_text().split('choices=', 1)[0], str(p), 'exec'))
choices = [(1,1,0), (0,1,0), (0,2,0), (0,1,1), (0,.5,.5), (1,0,0)]
errors = np.array([err(np.array([a]*2+[b]*2)) for a in choices for b in choices])
reflect = 1*len(choices)+1
worst = 0.
for held in range(Y.shape[1]):
    other = np.arange(Y.shape[1]) != held
    rmse = np.sqrt(np.mean(errors[:,:,other]**2, axis=(1,2)))
    best, runner = np.argsort(rmse)[:2]
    held_rmse = np.sqrt(np.mean(errors[best,:,held].reshape(len(shifts),-1)**2,axis=1))
    print('held pulse',held,'selected reflecting',best==reflect,'training runner-up RMSE',rmse[runner],'held RMSE by shift',held_rmse)
    assert best==reflect and held_rmse.max()<3e-6
    worst = max(worst,held_rmse.max())
print('Worst held-out pulse-by-shift RMSE',worst)

# Select any rational correction using only inner held-out sweep rows.
c = s['sensor_calibration_sweeps']; x,y = c['ref_rosette'],c['raw_rosette']
base = np.c_[x,np.ones(len(x))]
qs = [*x.T, *[(x[:,i]+sign*x[:,j])/np.sqrt(2) for i in range(4) for j in range(i+1,4) for sign in (-1,1)]]
designs = [base]
for q in qs:
    z = q/np.std(q)
    for a in (.01,.1,1.):
        designs.append(np.c_[base, base*(z*z/(1+a*z*z))[:,None]])
def residual(X, train, test):
    return X[test]@np.linalg.lstsq(X[train],y[train],rcond=None)[0]-y[test]
order=np.argsort(np.max(np.abs(x),axis=1)); worst=np.zeros(2)
for outer in np.array_split(order,5):
    tr=order[~np.isin(order,outer)]
    inner=np.array_split(tr,4)
    scores=[]
    for X in designs[1:]:
        r=np.concatenate([residual(X,tr[~np.isin(tr,fold)],fold) for fold in inner])
        scores.append(np.sqrt(np.mean(r*r,axis=0)).max())
    for j,X in enumerate((base,designs[1+int(np.argmin(scores))])):
        r=residual(X,tr,outer)
        worst[j]=max(worst[j],np.sqrt(np.mean(r*r,axis=0)).max())
print('Worst outer amplitude block x channel RMSE [affine, nested rational]',worst)
