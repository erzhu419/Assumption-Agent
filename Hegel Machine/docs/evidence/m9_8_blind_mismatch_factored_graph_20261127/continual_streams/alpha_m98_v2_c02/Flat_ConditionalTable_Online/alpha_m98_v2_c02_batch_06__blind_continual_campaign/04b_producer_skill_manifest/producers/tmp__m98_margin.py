import importlib.util, pickle
from itertools import combinations_with_replacement
import numpy as np
spec=importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl','rb') as f: d=pickle.load(f)[0]['dataset']
c=d['calibration_load_cells']; x=c['true_channel_loads']; y=c['bridge_raw']; n=len(x)
pairs=list(combinations_with_replacement(range(4),2))
B=np.column_stack([np.ones(n),x,*[x[:,j]*x[:,k] for j,k in pairs],x**3])
q=np.column_stack([x,x.sum(axis=1)/2,*[(x[:,j]+sign*x[:,k])/np.sqrt(2) for j in range(4) for k in range(j+1,4) for sign in [-1,1]]])
D=np.column_stack([np.ones(n),*[1+a*q[:,j]**2 for a in [.01,.05,.2,1.] for j in range(q.shape[1])]])
fold=np.arange(n)%5

def fit(tr,i):
    tr=tr & (np.max(abs(y),axis=1)<4*np.median(np.max(abs(y[tr]),axis=1)))
    p=B/D[:,i,None]
    return p@np.linalg.lstsq(p[tr],y[tr],rcond=None)[0]

def rmse(e): return np.sqrt(np.mean(e**2))
base=[]; alt=[]
for outer in range(5):
    eligible=fold!=outer
    cutoff=4*np.median(np.max(abs(y[eligible]),axis=1))
    good=np.max(abs(y),axis=1)<cutoff
    te=~eligible&good
    scores=[]
    for i in range(1,D.shape[1]):
        errors=[(fit(eligible&(fold!=inner),i)-y)[(fold==inner)&good] for inner in range(5) if inner!=outer]
        scores.append(rmse(np.vstack(errors)))
    choice=1+np.argmin(scores)
    e0=(fit(eligible,0)-y)[te]; e1=(fit(eligible,choice)-y)[te]
    base.append(rmse(e0)); alt.append(rmse(e1))
print('Bridge outer-fold raw-channel RMSE, polynomial:',base,'nested rational-only:',alt)
# Signed errors by lab: compare between-lab offsets to within-lab rate scatter.
a=d['train']; b=d['test']; out=m.solve_batch(d); xtr,_=m._standardize(a,b)
stress=m._stress_calibration(c); ss,rr=m._design(a,xtr,stress,d['past_dt'],d['eval_times'])(out['estimated_taus']); th=out['hereditary_weights']
rate=a['instant_rate_obs']-rr@th; late=(a['strain_curves_obs']-a['strain_0'][:,None]-ss@th)[:,-3:]
for lab in np.unique(a['lab_id']):
    j=a['lab_id']==lab
    print('lab',lab,'rate median/rmse',np.median(rate[j]),rmse(rate[j]),'late-strain median/medianAE',np.median(late[j]),np.median(abs(late[j])))
