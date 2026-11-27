import glob, importlib.util, pickle
import numpy as np
from scipy.optimize import root
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']
def features(x):
    return np.c_[np.ones(len(x)),x,x**3,x*np.roll(x,-1,axis=1)]
A=features(x)
p0=np.linalg.lstsq(A,y,rcond=None)[0]; r=np.max(np.abs(A@p0-y),axis=1); print('initial residual median and largest',np.median(r),sorted(zip(r,range(len(r))))[-8:])
keep=r<5*np.median(r)
p=np.linalg.lstsq(A[keep],y[keep],rcond=None)[0]; r=np.max(np.abs(A@p-y),axis=1); print('excluded',np.flatnonzero(~keep),'inlier max',r[keep].max(),'excluded residuals',r[~keep]); print('coef',p.round(6))
loo = []
for i in np.flatnonzero(keep):
    tr=keep.copy(); tr[i]=False
    loo.append(A[i]@np.linalg.lstsq(A[tr],y[tr],rcond=None)[0]-y[i])
print("sweep LOO max by channel", np.max(np.abs(loo), axis=0)); assert np.max(np.abs(loo)) < 1e-4
b=c['single_bank_sweeps']; bank=b['bank_id']; xb=np.eye(4)[bank]*b['ref_signal'][:,None]; print('single-bank residuals',((features(xb)@p)[np.arange(len(bank)),bank]-b['raw_readout']).reshape(4,-1))
ys=[]
for o in s['y_obs_raw'].reshape(-1,4):
    q=root(lambda q: features(q[None,:])[0]@p-o,np.linalg.solve(p[1:5].T,o-p[0]),tol=1e-11); assert np.linalg.norm(q.fun)<1e-10; ys.append(q.x)
y=np.array(ys).ravel(); print('calibrated range',y.min(),y.max()); Z=s['Z_nuisance']; ridge=s['source_ridge']; cap=s['u_box']; total=s['total_budget']; nr=s['nuisance_ridge']; n=len(y)
for path in sorted(glob.glob('/tmp/hegel_continual_archive/*_solver.py')):
    spec=importlib.util.spec_from_file_location('m',path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    S,last=m._forward(s)
    def fit(train):
        X=Z[train]; M=np.c_[S[train],y[train]]; P=np.linalg.solve(X.T@X+nr*np.eye(X.shape[1]),X.T); D=M-X@(P@M); A,b=D[:,:-1],D[:,-1]; w,nu=m._box_budget_qp(A.T@A+ridge*np.eye(A.shape[1]),A.T@b,cap,total); gamma=P@(y[train]-S[train]@w); return w,gamma,0.5*(np.linalg.norm(A@w-b)**2+ridge*(w@w))
    w,g,J=fit(np.ones(n,bool)); e=y-S@w-Z@g
    reps=len(s['sensor_ij'])//4; fold=np.tile(np.repeat(np.arange(reps),4),len(s['mu_shifts'])); ee=np.empty(n)
    for i in range(reps):
        tr=fold!=i; wt,gt,_=fit(tr); ee[~tr]=(y-S@wt-Z@gt)[~tr]
    print(path.split('/')[-1], 'J',J,'RMSE',np.sqrt(np.mean(e**2)), 'CV shift RMSE',np.sqrt(np.mean(ee.reshape(3,-1)**2,axis=1)), 'weights',w.round(5),'largest residuals',sorted(zip(abs(e),range(n)))[-5:])
