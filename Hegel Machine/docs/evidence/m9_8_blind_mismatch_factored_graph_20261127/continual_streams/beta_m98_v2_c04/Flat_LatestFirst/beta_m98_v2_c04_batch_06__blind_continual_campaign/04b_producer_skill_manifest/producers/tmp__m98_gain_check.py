import pickle, runpy
import numpy as np
from scipy.optimize import least_squares
m = runpy.run_path('/holdout_task/batch_solver.py')
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
st = item['study']; sw = st['sensor_calibration_sweeps']
ref, raw = sw['ref_rosette'], sw['raw_rosette']
def designs(v):
    return np.stack([m['_rosette_design'](v,j) for j in range(4)], axis=1)
X = designs(ref)
def predict(A, c):
    return np.einsum('njk,jk->nj', A,c)
def fit(train):
    peak = np.max(abs(raw),axis=1); med=np.median(peak[train]); limit=med+10*np.median(abs(peak[train]-med))
    clean=peak <= limit; take=train & clean
    A, B=X[take],raw[take]
    base=np.array([np.linalg.lstsq(A[:,j],B[:,j],rcond=None)[0] for j in range(4)])
    def coefs(p):
        gain=p[4:8]; out=np.empty((4,6)); out[:,0]=p[:4]; out[:,2]=gain
        out[:,[1,3,4,5]]=gain[:,None]*p[8:]; return out
    init=np.r_[base[:,0],base[:,2],np.mean(base[:,[1,3,4,5]]/base[:,2,None],axis=0)]
    p=least_squares(lambda p:(predict(A,coefs(p))-B).ravel(),init,gtol=1e-13,ftol=1e-13,xtol=1e-13).x
    return base, coefs(p), clean
errors=[[],[]]
for fold in range(5):
    train=np.arange(len(X))%5!=fold; a,b,clean=fit(train); valid=(~train)&clean
    for ls,c in zip(errors,(a,b)):
        ls.extend(predict(X[valid],c)-raw[valid])
S,_=m['_forward'](st); a,b,_=fit(np.ones(len(X),bool))
for name,e,c in zip(('independent','shared_gain'),errors,(a,b)):
    e=np.asarray(e); slices=[]; peak=[]
    for pulse in item['campaign_diagnostics']:
        y=pulse['y_pulse_raw'].reshape(len(st['mu_shifts']),-1,4)
        err=predict(designs((S@pulse['w_pulse']).reshape(-1,4)),c).reshape(y.shape)-y
        slices.extend((np.linalg.norm(err,axis=1)/np.linalg.norm(y,axis=1)).ravel())
        peak.append(np.linalg.norm(err)/np.linalg.norm(y))
    print(name,'held-out worst-bank RMS',np.sqrt(np.mean(e**2,axis=0)).max(),'held-out abs max',abs(e).max(),'worst pulse/shift/bank',max(slices),'worst whole pulse',max(peak))
print('normalized coefficients by bank:\n',a[:,[1,3,4,5]]/a[:,2,None])
