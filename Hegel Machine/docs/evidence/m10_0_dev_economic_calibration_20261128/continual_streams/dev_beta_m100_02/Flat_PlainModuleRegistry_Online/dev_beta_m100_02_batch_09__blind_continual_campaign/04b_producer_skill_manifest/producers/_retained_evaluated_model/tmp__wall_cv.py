import sys,pickle,itertools
import numpy as np
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
from downstream_solver import _responses,_box_simplex
s=calibrate_upstream(pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'])
y,Z=s['y_obs_raw'],s['Z_nuisance']; r=s['source_ridge']; d=s['nuisance_ridge']; n=len(s['sensor_ij']); ys=np.arange(len(y)); groups=ys%n; sh=ys//n

def test(S,train,test):
    Xt=np.column_stack((S[train],y[train])); z=Z[train]
    B=np.linalg.solve(z.T@z+d*np.eye(Z.shape[1]),z.T@Xt)
    X=Xt-z@B; A,b=X[:,:-1],X[:,-1]
    w,_=_box_simplex(A.T@A+r*np.eye(A.shape[1]),-A.T@b,s['u_box'],s['total_budget'])
    gamma=B[:,-1]-B[:,:-1]@w
    return np.sqrt(np.mean((S[test]@w+Z[test]@gamma-y[test])**2)),w

for walls in itertools.product(('open','reflect','periodic'),repeat=2):
    S,_=_responses(s,walls); fit,w=test(S,ys,ys)
    lo=[test(S,groups!=i,groups==i)[0] for i in range(n)]
    mu=[test(S,sh!=i,sh==i)[0] for i in range(len(s['mu_shifts']))]
    ros=[test(S,ys//4!=i,ys//4==i)[0] for i in np.unique(ys//4)]
    print(walls,'fit',round(fit,7),'LOS rms/worst',np.round([np.sqrt(np.mean(np.square(lo))),max(lo)],7),'rosette worst',round(max(ros),7),'shift',np.round(mu,7),'w',np.round(w,4))
