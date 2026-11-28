import pickle,numpy as np
from itertools import combinations_with_replacement as cwr
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; t=s['eval_times']
p=s['panel_c']; u=p['u']; v=p['v1']; w=p['v2']; b=s['panel_b']; B=np.linalg.lstsq(np.c_[np.ones(len(b['u'])),b['u']],b['v'],rcond=None)[0]; print('B A SV',np.linalg.svd(B[1:],compute_uv=False))
def fit(x,y):
 c=np.linalg.lstsq(x,y,rcond=None)[0]; e=y-x@c; return c,np.sqrt(np.mean(e**2)),np.max(np.abs(e))
def poly(x,d):return np.column_stack([np.ones(len(x))]+[np.prod(x[:,idx],axis=1) for j in range(1,d+1) for idx in cwr(range(x.shape[1]),j)])
for src,xx in [('u',u),('v1',v),('B(u)',u@B[1:]+B[0]),('unrot-v1',(v-B[0])@np.linalg.inv(B[1:]))]:
 for dst,yy in [('v1',v),('v2',w)]:
  if src=='v1' and dst=='v1':continue
  for basis,xxf in [('lin',poly(xx,1)),('diag2',np.c_[poly(xx,1),xx**2]),('quad',poly(xx,2)),('diag3',np.c_[poly(xx,1),xx**2,xx**3]),('tanh',np.c_[np.ones(len(xx)),np.tanh(xx)]),('sat',np.c_[poly(xx,1),np.expm1(-.2962*xx**2)]),('trig',np.c_[poly(xx,1),np.sin(xx),np.cos(xx)]),('odd',np.c_[np.ones(len(xx)),xx,xx**3,np.tanh(xx),np.sin(xx)])]:
   c,r,m=fit(xxf,yy); j=np.arange(len(xx))%4==0; cc=fit(xxf[~j],yy[~j])[0]; print(src,dst,basis,'%.6f %.6f %.6f'%(r,m,np.sqrt(np.mean((yy[j]-xxf[j]@cc)**2))))
   if src=='u' and dst=='v1' and basis in ['quad','diag2']:print(np.round(c,4))
def prof(r):
 q=-np.expm1(-t[:,None]*r); return (tr['traj'].T-q@np.linalg.lstsq(q,tr['traj'].T,rcond=None)[0]).ravel()
r=least_squares(prof,[.392,1.857],bounds=(.01,10),gtol=1e-12).x; print('rates',r,'rms',np.sqrt(np.mean(prof(r)**2)))
