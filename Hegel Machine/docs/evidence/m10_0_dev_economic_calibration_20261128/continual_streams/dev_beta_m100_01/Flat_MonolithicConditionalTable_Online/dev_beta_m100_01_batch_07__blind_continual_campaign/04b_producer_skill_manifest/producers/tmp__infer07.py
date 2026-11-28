from probe07 import m,s
import numpy as np
from itertools import product
y=m._calibrate(s); Z=s['Z_nuisance']; budget=s['total_budget']; cap=s['u_box']
from scipy.optimize import lsq_linear
for walls in product(['open','reflect','periodic'],repeat=2):
 S,U=m._forward(s,walls); B=np.c_[S,Z]; w0=np.full(S.shape[1],budget/S.shape[1]); T=np.eye(S.shape[1])[:,:-1]; T[-1]=-1
 t=np.linalg.lstsq(np.c_[S@T,Z],y-S@w0,rcond=None)[0]; w=w0+T@t[:T.shape[1]]; err=y-S@w-Z@t[T.shape[1]:]; print(walls,'unpenalized RMSE',np.sqrt(np.mean(err**2)),'bounds',w.min(),w.max(),'gamma',t[T.shape[1]:].round(4),'shift RMSE',np.sqrt(np.mean(err.reshape(3,-1)**2,axis=1)))
