import pickle, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from upstream_transducer import _fit, _forward, _inverse
s = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['study']
c = s['sensor_calibration_sweeps']; x, y = c['ref_rosette'], c['raw_rosette']
clean = np.linalg.norm(y-x, axis=1) < 8
labels = ('affine', 'affine+cubic', 'affine+product', 'full')
worst = dict.fromkeys(labels, 0.0)
worst_inverse = 0.0
for coordinate in (np.linalg.norm(x,axis=1), np.arange(len(x)), *x.T):
    for h in np.array_split(np.argsort(coordinate), 5):
        t = np.setdiff1d(np.arange(len(x)), h)
        h = h[clean[h]]
        if not len(h): continue
        coef = _fit(x[t], y[t])
        r = np.linalg.norm(y[t]-_forward(x[t],coef),axis=1)
        keep = t[r <= max(1e-6, 8*np.median(r))]
        for name, tail in zip(labels, ([], [6], [5], [5,6])):
            cols = [0,1,2,3,4]+tail
            pred = []
            for j in range(4):
                a = np.column_stack((np.ones(len(x)), x, x[:,j]*x[:,(j+1)%4], x[:,j]**3))[:,cols]
                pred.append(a[h] @ np.linalg.lstsq(a[keep],y[keep,j],rcond=None)[0])
            worst[name] = max(worst[name], np.sqrt(np.mean((np.array(pred).T-y[h])**2,axis=0)).max())
        worst_inverse = max(worst_inverse,np.sqrt(np.mean((_inverse(y[h],coef)-x[h])**2,axis=0)).max())
print('Blocked worst-channel forward RMSE by nested basis:', worst)
print('Exact artifact blocked worst-channel inverse RMSE:', worst_inverse)
assert worst['full'] < 1e-4 and worst_inverse < 1e-4
assert worst['affine+cubic'] > 10*worst['full'] and worst['affine+product'] > 10*worst['full']
