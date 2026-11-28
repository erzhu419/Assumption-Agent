import pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a=s['panel_a']; U=a['u']; V=a['v']; X=np.vstack([s['train']['x'],s['test']['x']]); np.set_printoptions(precision=4,suppress=True)
fs={'affine':lambda u:np.c_[u*0+1,u], 'cubic':lambda u:np.c_[u*0+1,u,u**3], 'quadratic_cubic':lambda u:np.c_[u*0+1,u,u*u,u**3], 'signed_quadratic':lambda u:np.c_[u*0+1,u,u*abs(u)],'hinge':lambda u:np.c_[u*0+1,u,np.maximum(-u,0)],'asym_cubic':lambda u:np.c_[u*0+1,u,np.minimum(u,0)**3,np.maximum(u,0)**3], 'asym_quadratic':lambda u:np.c_[u*0+1,u,np.minimum(u,0)**2,np.maximum(u,0)**2]}
for name,f in fs.items():
 print(name)
 for u,v in zip(U.T,V.T):
  d=f(u); b=least_squares(lambda b:d@b-v,np.linalg.lstsq(d,v,rcond=None)[0],loss='cauchy',f_scale=.06).x; r=v-d@b
  print(b, 'neg-pos median',np.median(abs(r[u<0])),np.median(abs(r[u>=0])), 'sorted',np.round(np.sort(abs(r))[::6],4))
# Check whether calibration residuals agree rowwise or are per-coordinate.
d=np.c_[np.ones(len(U)),U,U**2,U**3,*[U[:,i]*U[:,j] for i in range(4) for j in range(i)]]
for v in V.T:
 b=least_squares(lambda b:d@b-v,np.linalg.lstsq(d,v,rcond=None)[0],loss='cauchy',f_scale=.06).x
 print('joint',b,np.sort(abs(v-d@b))[::6])
