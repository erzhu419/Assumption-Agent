import importlib.util
import itertools
import pickle
import sys
import numpy as np

spec=importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
spec=importlib.util.spec_from_file_location('wall_probe','/tmp/wall_probe.py')
probe=importlib.util.module_from_spec(spec); spec.loader.exec_module(probe)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
c=s['sensor_calibration_sweeps']; x,raw=c['ref_rosette'],c['raw_rosette']
y=m._calibrate(s); z=s['Z_nuisance']; ns=len(s['sensor_ij'])
shift=np.repeat(np.arange(len(s['mu_shifts'])),ns); ros=np.tile(np.arange(ns)//4,len(s['mu_shifts']))
walls=list(itertools.product(('robin','reflect','periodic'),repeat=2))
mat=[probe._transport(s,p)[0] for p in walls]
assert np.allclose(mat[0],m._transport(s)[0],rtol=0,atol=1e-12)

def predict(a,tr,te):
    zz=z[tr]; aa=a[tr]; yy=y[tr]
    P=np.linalg.solve(zz.T@zz+s['nuisance_ridge']*np.eye(zz.shape[1]),zz.T)
    A=aa-zz@(P@aa); r=yy-zz@(P@yy)
    w,_=m._box_simplex(A.T@A+s['source_ridge']*np.eye(A.shape[1]),-A.T@r,s['u_box'],s['total_budget'])
    return a[te]@w+z[te]@(P@(yy-aa@w))

for labels in (ros,shift):
    scores=[]; selected=[]; nuisance=[]
    for group in np.unique(labels):
        test=labels==group; train=~test
        inner=[train&(ros==r) for r in np.unique(ros[train])]
        losses=[sum(np.sum((predict(a,train&~q,q)-y[q])**2) for q in inner) for a in mat]
        best=int(np.argmin(losses)); selected.append(walls[best])
        scores.append(np.sqrt(np.mean((predict(mat[best],train,test)-y[test])**2)))
        zz=z[train]; gamma=np.linalg.solve(zz.T@zz+s['nuisance_ridge']*np.eye(zz.shape[1]),zz.T@y[train])
        nuisance.append(np.sqrt(np.mean((z[test]@gamma-y[test])**2)))
    print('outer '+('rosette' if labels is ros else 'shift'), 'selected',selected,'transport+profile RMSE',np.round(scores,6),'nuisance-only RMSE',np.round(nuisance,6))

# Challenge the local calibration with omitted polynomial and rational-denominator directions.
# For f=P/(1+a Q^2), the first omitted direction at a=0 is -P Q^2.
# Keep contamination selection and normalization entirely inside each training fold.
powers=[q for d in range(1,4) for q in itertools.combinations_with_replacement(range(4),d)]
poly=np.column_stack([np.ones(len(x))]+[x[:,q].prod(axis=1) for q in powers])
baseerr=[]; polyerr=[]; raterr=[]; amp=[]; rejected=[]
for i in range(len(x)):
    tr=np.arange(len(x))!=i
    p=m._fit_calibration(x[tr],raw[tr]); pred=m._forward(x,p)
    err=np.max(abs(raw-pred),axis=1); floor=10*max(np.median(err[tr]),np.finfo(float).eps)
    clean=tr&(err<=floor)
    if err[i]>floor:
        rejected.append(i); continue
    baseerr.append(pred[i]-raw[i]); amp.append(np.max(abs(x[i])))
    polyerr.append(poly[i]@np.linalg.lstsq(poly[clean],raw[clean],rcond=None)[0]-raw[i])
    residual=[]
    for j in range(4):
        u,v=x[:,j],x[:,(j+1)%4]
        q=np.column_stack((u,v,u+v,u-v))
        basis=np.column_stack((m._design(x,j),-(pred[:,j]-p[j,0])[:,None]*q**2))
        residual.append(basis[i]@np.linalg.lstsq(basis[clean],raw[clean,j],rcond=None)[0]-raw[i,j])
    raterr.append(residual)
amp=np.array(amp); print('sweep rows flagged out of fold',rejected,'observation max calibrated amplitude',np.max(abs(y)))
for name,e in [('cyclic',baseerr),('full polynomial cubic',polyerr),('rational tangent',raterr)]:
    e=np.asarray(e); slices=[('all',np.ones(len(e),bool)),('low amplitude',amp<=np.quantile(amp,.25)),('high amplitude',amp>=np.quantile(amp,.75))]
    print(name,[(label,float(np.max(np.sqrt(np.mean(e[mask]**2,axis=0))))) for label,mask in slices])
