import importlib.util, pickle, warnings
import numpy as np
from sklearn.model_selection import KFold
warnings.filterwarnings('ignore', category=UserWarning)
spec=importlib.util.spec_from_file_location('batch_solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train'];n=len(tr['y']);ops=m._operators(s)
for i,j in [(np.arange(52),np.arange(52,n)),(np.arange(56),np.arange(56,n)),*KFold(4,shuffle=True,random_state=13).split(tr['x'])]:
 x=tr['x'][np.r_[i,j]];y=m._scalar(x,tr['y'][i],ops);t=m._trajectory(x,tr['traj'][i],ops);e=abs(t-tr['traj'][j]);print('MAE',round(abs(y-tr['y'][j]).mean(),6),round(e.mean(),6),np.round(e.mean(0),5))
z=m.solve_batch(s);assert z['state_pred'].shape==(36,8) and z['y_hat'].shape==(36,);assert all(np.isfinite(v).all() for v in z.values());print('diagnostics',z['diagnostics_vec'])
