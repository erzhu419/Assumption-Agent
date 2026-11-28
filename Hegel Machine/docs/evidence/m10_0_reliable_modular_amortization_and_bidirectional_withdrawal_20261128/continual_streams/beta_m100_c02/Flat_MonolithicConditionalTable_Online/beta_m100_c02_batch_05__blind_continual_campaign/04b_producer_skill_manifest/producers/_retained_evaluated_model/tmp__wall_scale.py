import sys
sys.path.insert(0,'/holdout_task')
import batch_solver as b
import compare_walls as c
import numpy as np
from scipy.optimize import least_squares
c.y=b.calibrate(c.s); tr=np.ones(len(c.y),dtype=bool)
def pred(p,tr,ridge):
    return c.fit(c.forward(c.axis('x',tuple(p[:2])),c.axis('y',tuple(p[2:])))[0],tr,ridge)[1]
folds=[np.tile(np.arange(c.ns)//4==i,len(c.s['mu_shifts'])) for i in range(c.ns//4)]
for ridge in (1e-7,.04):
    baseline=[]; free=[]; params=[]
    for te in folds:
        fit=least_squares(lambda p:pred(p,~te,ridge)[~te],np.ones(4),bounds=(0,4))
        params.append(fit.x)
        baseline.append(np.mean(pred(np.ones(4),~te,ridge)[te]**2)**.5)
        free.append(np.mean(pred(fit.x,~te,ridge)[te]**2)**.5)
    print('ridge, fixed pooled/worst, free pooled/worst, free wall factor range',ridge,np.linalg.norm(baseline)/np.sqrt(len(folds)),max(baseline),np.linalg.norm(free)/np.sqrt(len(folds)),max(free),np.min(params,axis=0),np.max(params,axis=0))
