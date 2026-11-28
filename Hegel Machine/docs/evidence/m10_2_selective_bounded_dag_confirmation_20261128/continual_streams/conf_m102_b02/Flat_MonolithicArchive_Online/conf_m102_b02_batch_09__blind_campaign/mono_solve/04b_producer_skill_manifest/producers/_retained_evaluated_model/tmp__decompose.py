import sys
sys.path.insert(0,'/holdout_task')
from batch_solver import _features, _intercept, _ls
from panel import s, least_squares, np
z,h,c=_features(s,s['train']['x']); D=_intercept(np.c_[z,h,c]); tr=s['train']['traj']; y=s['train']['y']; t=s['eval_times']
print('Y coeff',_ls(D,y)); print('state coefficient matrix',_ls(D,tr).T)
base=_intercept(np.c_[z,h,c[:,[0,5,10,15]].sum(1)]); print('y base coeff and rms',_ls(base,y),np.std(base@_ls(base,y)-y))
