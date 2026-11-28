import pickle
from itertools import combinations_with_replacement
import numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x,y,z=[s['train'][k] for k in ('x','y','traj')]; t=s['eval_times']; np.set_printoptions(precision=4,suppress=True,linewidth=140)
def ls(a,b): return np.linalg.lstsq(a,b,rcond=None)[0]
def basis(k):return -np.expm1(-t[:,None]*k)
def poly(x,d):return np.column_stack([np.ones(len(x))]+[np.prod(x[:,q],axis=1) for j in range(1,d+1) for q in combinations_with_replacement(range(4),j)])
a=poly(x,4); n=len(x); rng=np.random.default_rng(152); radial=np.linalg.norm(x,axis=1); splits=[np.arange(52,72)]+list(np.array_split(rng.permutation(n),4))+[np.argsort(radial)[-18:]]
for va in splits:
 tr=np.setdiff1d(np.arange(n),va); zz=z[tr].T
 k=least_squares(lambda k:(basis(k)@ls(basis(k),zz)-zz).ravel(),[1.5,.25],bounds=([.7,.02],[4,.65])).x; b=basis(k); hd=np.c_[y,ls(b,z.T).T]; yy=np.c_[y,z]
 if va[0]==52:
  for d in (2,3):
   aa=a[:,:{2:15,3:35}[d]]; c=ls(aa[tr],hd[tr]); print('readout coefficients degree',d,c); print('fit rmse',np.sqrt(np.mean((aa[tr]@c-hd[tr])**2,axis=0)))
  for d in (1,2,3):
   f=np.c_[a[:,:{1:5,2:15,3:35}[d]],hd[:,1:]]; c=ls(f[tr],y[tr]); print('readout from measured modes',d,'MAE',np.mean(abs(f[va]@c-y[va])),'mode weights',c[-2:])
 print('split',len(va),va[:3],k,'oracle basis',np.mean(abs(hd[va,1:]@b.T-z[va])))
 for quadratic in (False,True):
  for scale,reg in (('cubic',0.),('cubic',.05),('cubic',.2),('quartic',.1),('quartic',.35),('gauss',.001),('gauss',.01),('gauss',.1)):
   if scale=='gauss':
    g=np.exp(-.23*np.sum((x[:,None]-x[None,:])**2,axis=2)); p=a[:,:15]; G=g[np.ix_(tr,tr)]+reg*np.eye(len(tr)); C=ls(np.block([[G,p[tr]],[p[tr].T,np.zeros((15,15))]]),np.r_[hd[tr],np.zeros((15,3))]); ph=np.c_[g[np.ix_(va,tr)],p[va]]@C
   else:
    mat=a[:,:35 if scale=='cubic' else 70]; pen=np.r_[np.zeros(15),np.ones(20),np.full(mat.shape[1]-35,3.)]*reg
    ph=mat[va]@ls(np.r_[mat[tr],np.diag(pen)],np.r_[hd[tr],np.zeros((len(pen),3))]);
   if quadratic: ph[:,2]=a[va,:15]@ls(a[tr,:15],hd[tr,2])
   e=abs(np.c_[ph[:,0],ph[:,1:]@b.T]-yy[va]); print(scale,reg,quadratic,np.round([e[:,0].mean(),e[:,1:].mean(),e[:,1:].mean(0).max()],5))
