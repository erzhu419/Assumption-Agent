import pickle, runpy, itertools
import numpy as np
m=runpy.run_path('/holdout_task/batch_solver.py')
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
c=s['sensor_calibration_sweeps']; ref,raw=c['ref_rosette'],c['raw_rosette']
keep=np.max(np.abs(raw),axis=1)<9
print('retained raw maximum / rejected raw minimum',np.max(np.abs(raw[keep])),np.min(np.max(np.abs(raw[~keep]),axis=1)))
x,y=ref[keep],raw[keep]; idx=np.arange(len(x)); fold=np.random.default_rng(89).permutation(len(x))%5
settings=[('pair',0.)]+[(name,a) for name in ('own','difference','sum') for a in (0.001,0.01,0.05,0.2,0.5,1.)]
errors=np.empty((3,*y.shape)); choices=[]
for j in range(4):
    u,v=x[:,j],x[:,(j+1)%4]
    base=m['_calibration_features'](x,j)
    extra=np.column_stack((base,u*u,v*v,u*u*v,u*v*v))
    mats=[]
    for name,a in settings:
        q={'pair':u,'own':u,'difference':u-v,'sum':u+v}[name]
        mats.append(extra if name=='pair' else np.column_stack((base[:,0],base[:,1:]/(1+a*q[:,None]**2))))
    def predict(D,tr,te):
        return D[te]@np.linalg.lstsq(D[tr],y[tr,j],rcond=None)[0]
    for k in range(5):
        te=fold==k; tr=~te
        scores=[np.mean(np.concatenate([predict(D,tr&(fold!=l),fold==l)-y[fold==l,j] for l in range(5) if l!=k])**2) for D in mats]
        best=np.argmin(scores); choices.append(settings[best]);
        for row,D in enumerate((base,extra,mats[best])):
            errors[row,te,j]=predict(D,tr,te)-y[te,j]
rm=lambda e:np.sqrt(np.mean(e**2))
near=np.max(abs(x),axis=1)<=0.85
for name,e in zip(('compact','pair-augmented','inner-selected challenger'),errors):
    print(name,'pooled / worst-channel / near-zero / wide RMSE',rm(e),max(rm(v) for v in e.T),rm(e[near]),rm(e[~near]))
print('inner-fold choices',dict((str(k),choices.count(k)) for k in set(choices)))
b=c['single_bank_sweeps']; a=m['_fit_calibration'](x,y)
bank=np.asarray(b['bank_id']); v=np.asarray(b['ref_signal']); r=np.asarray(b['raw_readout'])
if bank.shape==v.shape==r.shape:
    predicted=a[bank,0]+a[bank,2]*v+a[bank,5]*v**3
    print('single-bank sweep RMSE by bank',[rm((predicted-r)[bank==j]) for j in range(4)])
else:
    print('single-bank sweep shapes',bank.shape,v.shape,r.shape)
d,_=m['_forward'](s); w=m['solve_batch'](s)['weights']; yc=m['_calibrate'](s); z=s['Z_nuisance']; g=z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]); e=d@w-yc; centered=e-z@np.linalg.solve(g,z.T@e)
for group,lab in ((np.repeat(s['mu_shifts'],len(s['sensor_ij'])),'shift'),(s['row_bank_id'],'bank')):
    print(lab,'slice RMSE (before, after nuisance projection)',[(float(k),rm(e[group==k]),rm(centered[group==k]),float(np.mean(centered[group==k]))) for k in np.unique(group)])
print('numerical certificate and field check already completed on the unchanged artifact in isolated Python on turn 9')
