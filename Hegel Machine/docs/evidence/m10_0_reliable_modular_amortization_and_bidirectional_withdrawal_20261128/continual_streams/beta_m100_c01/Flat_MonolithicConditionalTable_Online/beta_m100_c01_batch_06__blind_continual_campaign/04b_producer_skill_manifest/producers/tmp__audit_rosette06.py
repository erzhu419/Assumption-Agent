import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root+'batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
s = pickle.load(open(root+'public_batch.pkl','rb'))[0]['study']
c = s['sensor_calibration_sweeps']; b = c['single_bank_sweeps']
x,y = c['ref_rosette'],c['raw_rosette']
clean = np.max(abs(y),axis=1)<5

def features(x):
    other = np.roll(x,2,axis=1)
    d,t = x-other,x+other
    return np.stack([q/(1+r*r) for q,r in ((d,d),(t,t),(x*x,other),(other*other,x),(x*other,d))],axis=-1)

records=[]
for k, mask in enumerate(np.array_split(np.arange(len(x)), 4)):
    held = np.isin(np.arange(len(x)),mask)
    train,test = ~held & clean,held & clean
    banks = {name:v[b['bank_id']!=k] for name,v in b.items()}
    p = m._fit_calibration(x[~held],y[~held],banks)
    e = y[test]-m._response(x[test],p)
    train_e = y[train]-m._response(x[train],p)
    F,T = features(x[train]),features(x[test])
    coef = np.stack([np.linalg.lstsq(F[:,j],train_e[:,j],rcond=None)[0] for j in range(4)])
    alt_e = e - np.einsum('ijk,jk->ij',T,coef)
    train_c = {'ref_rosette':x[~held],'raw_rosette':y[~held],'single_bank_sweeps':banks}
    yy = m._calibrate({'sensor_calibration_sweeps':train_c,'y_obs_raw':s['y_obs_raw']})
    q = np.max(abs(yy-m._calibrate(s)))
    rms = lambda z: np.sqrt(np.mean(z*z))
    records.append((rms(e),rms(alt_e),q))
print('blocked sweep RMSE (base, rational residual correction), max observed-input inversion change:', np.asarray(records))
print('worst across blocked folds:',np.max(records,axis=0))
