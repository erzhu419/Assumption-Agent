import importlib.util
import itertools
import pickle
import numpy as np

P='/holdout_task/batch_solver.py'
def load(p):
    spec=importlib.util.spec_from_file_location('solver',p)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
m=load(P)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']
rng=np.random.default_rng(27)
folds=np.array_split(rng.permutation(len(x)),5)
aff=np.c_[np.ones(len(x)),x]
terms=[np.prod(x[:,ix],axis=1) for degree in (2,3) for ix in itertools.combinations_with_replacement(range(4),degree)]
poly=np.column_stack([aff,*terms])
# Q includes each channel and pairwise sums and differences. Pick any denominator only on inner folds.
qs=[*x.T]+[x[:,i]+sign*x[:,j] for i,j in itertools.combinations(range(4),2) for sign in (-1,1)]
rats=[aff]+[aff/(1+a*q[:,None]**2) for q in qs for a in (.0001,.001,.01,.1)]
def predict(A,tr,te):
    return A[te]@np.linalg.lstsq(A[tr],y[tr],rcond=None)[0]
errors={k:np.empty_like(y) for k in ('affine','cubic','rational')}; chosen=[]
for k,te in enumerate(folds):
    inner=[f for j,f in enumerate(folds) if j!=k]; tr=np.concatenate(inner)
    scores=np.array([sum(np.sum((predict(A,np.concatenate([j for h,j in enumerate(inner) if h!=i]),f)-y[f])**2,axis=0) for i,f in enumerate(inner)) for A in rats]); ids=scores.argmin(axis=0); chosen.extend(ids.tolist())
    for name,A in [('affine',aff),('cubic',poly)]: errors[name][te]=predict(A,tr,te)-y[te]
    for j,i in enumerate(ids): errors['rational'][te,j]=predict(rats[i],tr,te)[:,j]-y[te,j]
for name,e in errors.items(): print(name,'outer bank RMSE',np.sqrt(np.mean(e*e,axis=0)))
print('inner rational choices excluding affine',np.count_nonzero(chosen))
q=c['single_bank_sweeps']; b=q['bank_id']; xx=np.zeros((len(b),4)); xx[np.arange(len(b)),b]=q['ref_signal']; C=np.linalg.lstsq(aff,y,rcond=None)[0]; e=(np.c_[np.ones(len(b)),xx]@C)[np.arange(len(b)),b]-q['raw_readout']; print('independent single-bank max',[float(np.max(abs(e[b==j]))) for j in range(4)])

# Leave all shifts of one rosette position out; profile nuisance using training rows only.
y=m._calibrate(s); Z=s['Z_nuisance']; shifts=len(s['mu_shifts']); n_sensor=len(s['sensor_ij']); n_ros=n_sensor//4
for name,model in [('Robin',m),('sealed',load('/tmp/hegel_continual_archive/beta_m98_v2_c02_batch_01_solver.py')),('periodic',load('/tmp/hegel_continual_archive/beta_m98_v2_c02_batch_03_solver.py'))]:
    S,_=model._forward(s); err=np.empty_like(y); raw=np.empty_like(y)
    for pos in range(n_ros):
        test=(np.arange(len(y)) % n_sensor)//4==pos; tr=~test; z=Z[tr]; p=np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T)
        A=S[tr]-z@(p@S[tr]); u=y[tr]-z@(p@y[tr]); w,_=m._box_budget_qp(A.T@A+s['source_ridge']*np.eye(A.shape[1]),A.T@u,s['u_box'],s['total_budget']); raw[test]=S[test]@w-y[test]; err[test]=raw[test]+Z[test]@(p@(y[tr]-S[tr]@w))
    E=err.reshape(shifts,n_ros,4); print(name,'profiled CV pooled, worst position, worst shift, worst bank',*[float(v) for v in (np.sqrt(np.mean(E**2)),np.sqrt(np.mean(E**2,axis=(0,2))).max(),np.sqrt(np.mean(E**2,axis=(1,2))).max(),np.sqrt(np.mean(E**2,axis=(0,1))).max())], 'pre-nuisance same-weight RMSE',float(np.sqrt(np.mean(raw**2))))
