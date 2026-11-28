import numpy as np,pickle,runpy
from scipy.optimize import minimize_scalar
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a=runpy.run_path('/tmp/hegel_continual_archive/dev_m102_c04_batch_00_solver.py'); A=a['_affine']; L=a['_ls']; C=s['panel_c']; u=C['u']; v=C['v1']; b,m=a['_calibration'](s['panel_b']); d,g=a['_calibration'](s['panel_a']); np.set_printoptions(precision=4,suppress=True)
for nm,z in [('u',u),('Bf',u@m+b),('Bi',(u-b)@np.linalg.inv(m)),('Af',u@g+d),('Ai',(u-d)@np.linalg.inv(g))]:
 def err(F):
  w=L(F,v); return np.linalg.norm(v-F@w)
 def fit(fn):
  grid=np.linspace(.01,16,161); es=np.array([err(fn(k)) for k in grid]); inds=np.where((es[1:-1]<=es[:-2]) & (es[1:-1]<=es[2:]))[0]+1; opts=[(es[0],grid[0]),(es[-1],grid[-1])];
  for j in inds:
   r=minimize_scalar(lambda k:err(fn(k)),bounds=(grid[j-1],grid[j+1]),method='bounded'); opts.append((r.fun,r.x))
  return sorted(opts)[:2]
 print(nm,'lin',err(A(z)),'quad',err(A(np.c_[z,*[z[:,i]*z[:,j] for i,j in combinations_with_replacement(range(4),2)]])))
 for name,fn in [('sin+cos',lambda k:A(np.c_[z,np.sin(k*z),np.cos(k*z)])),('sin*cos',lambda k:A(np.c_[z,np.sin(k*z)*np.roll(np.cos(k*z),1,axis=1)])),('sinSq',lambda k:A(np.c_[z,np.sin(k*z*z)])),('cosSq',lambda k:A(np.c_[z,np.cos(k*z*z)])),('cosCross',lambda k:A(np.c_[z,np.cos(k*z)*np.roll(z,1,axis=1)])),('crossSin',lambda k:A(np.c_[z,np.sin(k*z)*np.roll(z,1,axis=1)])),('harmonic',lambda k:A(np.c_[z,np.sin(k*z),np.sin(2*k*z)])),('rat',lambda k:A(np.c_[z,1/(1+k*z*z),z/(1+k*z*z)])),('quad+sin',lambda k:A(np.c_[z,z*z,np.sin(k*z)]))]: print(name,fit(fn))
 r=v-A(z)@L(A(z),v); print('resid SVD',np.linalg.svd(r,compute_uv=False)); print('linear coefficients',L(A(z),v))
