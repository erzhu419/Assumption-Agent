import sys,pickle,numpy as np
sys.path.insert(0,'/tmp/hegel_continual_archive')
import conf_m101_a03_batch_04_solver as a
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=52; x=np.r_[tr['x'],s['test']['x']]; y=tr['y']; T=tr['traj']; fit=a._fit; aff=a._affine; prod=a._products
c,keep=a._calibration(s['panel_a']); print('cal',keep.sum(),np.round(c,4)); z=a._coordinates(s['panel_a'],x)
b=a._operator_b(s['panel_b'],z); p=s['panel_b']; print('b panel RMSE',np.sqrt(np.mean((a._operator_b(p,p['u'])-p['v'])**2)))
print('trajectory sv',np.linalg.svd(T,compute_uv=False))
for k,v in [('x',x),('z',z),('b',b),('zb',np.c_[z,b])]:
 for name,f in [('lin',aff(v)),('quad',np.c_[aff(v),prod(v)] if v.shape[1]==4 else np.c_[aff(v),prod(z),prod(b)]),('tanh',np.c_[aff(v),np.tanh(v)]),('rat',np.c_[aff(v),v/(1+.75*v*v)])]:
  pred=f[n:72] @ fit(f[:n],np.c_[y,T][:n]); e=np.abs(pred-np.c_[y,T][n:]); print(k,name,tuple(e.mean(axis=0)[[0,1,-1]]),e[:,1:].mean())
for j in ('v1','v2'):
 p=s['panel_c']; u=p['u']; v=p[j]
 for name,f in [('quad',np.c_[aff(u),prod(u)]),('rat',np.c_[aff(u),u/(1+.75*u*u)]),('tanh',np.c_[aff(u),np.tanh(u)])]:
  h=f @ np.linalg.pinv(f); e=(v-h@v)/np.diag(np.eye(len(u))-h)[:,None]; print('panel_c',j,name,np.mean(abs(e)))
print('corr',np.corrcoef(np.c_[y,T,z[:72],b[:72]].T)[0]); print('range z,b',np.max(abs(z),axis=0),np.max(abs(b),axis=0))
