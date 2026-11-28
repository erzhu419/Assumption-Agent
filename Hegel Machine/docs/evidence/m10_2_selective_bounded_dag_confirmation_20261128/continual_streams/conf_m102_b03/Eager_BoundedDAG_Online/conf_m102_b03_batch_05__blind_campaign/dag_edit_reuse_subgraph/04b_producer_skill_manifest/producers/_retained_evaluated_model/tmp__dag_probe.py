import pickle, numpy as np, importlib.util
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
p='/tmp/hegel_continual_archive/conf_m102_b03_batch_00_solver.py'; spec=importlib.util.spec_from_file_location('old',p); old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
def ls(A,y):return np.linalg.lstsq(A,y,rcond=None)[0]
def ones(x):return np.c_[np.ones(len(x)),x]
def poly(u,d):return np.column_stack([np.ones(len(u))]+[np.prod(u[:,c],axis=1) for k in range(1,d+1) for c in combinations_with_replacement(range(u.shape[1]),k)])
def robust(A,y):
 p=least_squares(lambda z:A@z-y,ls(A,y),loss='soft_l1',f_scale=.06).x
 r=np.abs(A@p-y); good=r<min(1.5,max(.035,5*np.median(r))); return ls(A[good],y[good]),good
if __name__=='__main__':
 for name in ['panel_a','panel_b','panel_c']:
  p=s[name];u=p['u'];print(name)
  for k,v in p.items():
   if k=='u':continue
   for d in [1,2,3]:
    A=poly(u,d); rs=[]
    for j in range(4):
     if name=='panel_a': z,g=robust(A,v[:,j]); r=np.abs(A@z-v[:,j]);rs.append(np.round(np.quantile(r,[.5,.75,.85]),5));
     else:r=np.abs(A@ls(A,v[:,j])-v[:,j]);rs.append(np.round(np.quantile(r,[.5,.75,.85]),5))
    print(k,d,rs)
   if name=='panel_a':
    for name2,A in [('linear',ones(u)),('diag3',np.c_[ones(u),u**2,u**3]),('diag5',np.c_[ones(u),u**2,u**3,u**4,u**5])]:
     for j in range(4):
      z,g=robust(A,v[:,j]);r=A@z-v[:,j]; print(name2,j,'coef',np.round(z,3),'inliers',sum(g),'abs residual percentiles',np.round(np.quantile(abs(r),[.25,.5,.7,.8,.85,.9,.95]),4))
 if True:
  p=s['panel_c'];u=p['u']
  for j in [1,2]:
   v=p['v'+str(j)]; r=old._branch(p,u,j)-v; print('archive branch',j,np.max(abs(r)),np.sqrt(np.mean(r*r)))
 ts=s['train']; y,t=old._predict(s,slice(52),ts['x'][52:]);print('archive cv',np.mean(abs(y-ts['y'][52:])),np.mean(abs(t-ts['traj'][52:])))
