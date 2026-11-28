import sys
sys.path.insert(0,'/holdout_task')
from batch_solver import _design
import numpy as np
from scipy.optimize import least_squares
A=np.load('/tmp/paths.npz'); z=A['z'];T=A['T'];Y=A['Y'];t=A['t'];n=52;D=_design(z); H=np.c_[Y,T]
np.set_printoptions(precision=5,suppress=True,linewidth=145)
W=np.linalg.lstsq(D[:n],H[:n],rcond=None)[0];r=H-D@W;print('CV errors',abs(r[n:]).mean(0)); print('fit errors',abs(r[:n]).mean(0)); print('SVD T/H/R',*(np.linalg.svd(M,compute_uv=False) for M in [T,H,r[:n]]));print('coeff',W)
for j,k in [(0,5),(5,9),(9,21),(0,21)]:
 Z=W[j:k];print('block singular',j,k,np.linalg.svd(Z,compute_uv=False));print('time rows V',np.linalg.svd(Z,full_matrices=False)[2][:3]); print('y v time coeff corr',np.corrcoef(Z.T)[0]);
print('residual corr',np.corrcoef(r[:n].T)[0]);print('tele',A['t'])
