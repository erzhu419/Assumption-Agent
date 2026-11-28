import pickle, itertools, numpy as np
from scipy.optimize import minimize
import model
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; y=model.calibrate(s); z=s['Z_nuisance']; print('diagnostics',type(item.get('campaign_diagnostics')), 'banks',s['row_bank_id'][:16]); print('aux singular values',np.linalg.svd(s['aux_manifold_telemetry'],compute_uv=False)); print('aux mean',s['aux_manifold_telemetry'].mean(0));
rob=np.repeat([s['robin_x'],s['robin_y']],2); folds=np.tile(np.repeat(np.arange(8)%4,4),3)
def prof(q,train=None,cv=False):
 a,_=model.chamber_response(s,rob*np.asarray(q)); tr=np.ones(len(y),bool) if train is None else train
 zz=z[tr]; sy=np.c_[a[tr],y[tr]]; b=np.linalg.solve(zz.T@zz+s['nuisance_ridge']*np.eye(z.shape[1]),zz.T@sy); v=sy-zz@b; A,r=v[:,:-1],v[:,-1]; w,nu=model.box_simplex(A.T@A+s['source_ridge']*np.eye(A.shape[1]),A.T@r,s['u_box'],s['total_budget']); e=a@w+z@(b[:,-1]-b[:,:-1]@w)-y; obj=(e[tr]@e[tr]+s['source_ridge']*(w@w))/2
 return (obj,e,w) if cv else obj
rows=[]
for q in itertools.product((0,1),repeat=4):
 obj,e,w=prof(q,cv=True); scores=[np.sqrt(np.mean(prof(q,folds!=i,True)[1][folds==i]**2)) for i in range(4)]; rows.append((obj, np.sqrt(np.mean(e**2)), max(scores),q,w.round(3)))
for row in sorted(rows,key=lambda r:r[0]): print(row)
for tr in [None]+[folds!=i for i in range(4)]:
 fit=minimize(lambda q:prof(q,tr),[.5]*4,method='L-BFGS-B',bounds=[(0,2)]*4,options={'ftol':1e-13}); print('continuous',fit.x.round(4),fit.fun, 'test RMSE',None if tr is None else np.sqrt(np.mean(prof(fit.x,tr,True)[1][~tr]**2)))
