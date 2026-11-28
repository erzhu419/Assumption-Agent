import sys,pickle,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
from node_00 import _calibrate,_linear,_interactions,_curvature
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; print('study keys',s.keys()); n=len(s['train']['x']); x=np.vstack([s['train']['x'],s['test']['x']]); y=np.c_[s['train']['y'],s['train']['traj']]; z=_calibrate(s['panel_a'],x); np.set_printoptions(precision=5,suppress=True)
def poly(a,p):
 from itertools import combinations_with_replacement as cr
 return np.column_stack([np.ones(len(a))]+[np.prod(a[:,j],axis=1) for k in range(1,p+1) for j in cr(range(a.shape[1]),k)])
b=s['panel_b']; u,v=b['u'],b['v']; print('B first',u[:4],v[:4]); print('B fit forward and reverse')
for U,V in [(u,v),(v,u)]:
 for h in range(1,4):
  F=poly(U,h); P=np.linalg.pinv(F); r=F@P@V-V; print(h,np.sqrt(np.mean(r*r)),np.linalg.svd(F,compute_uv=False));
  if h<3: print('LOO',np.sqrt(np.mean((r/(1-np.sum(F*P.T,axis=1))[:,None])**2)))
 print(_linear(poly(U,1),V))
print('x,z samples',x[:8],z[:8]); print('x/z ranges',np.quantile(x,[0,.1,.9,1],axis=0),np.quantile(z,[0,.1,.9,1],axis=0)); print('singular x,z cubic',np.linalg.svd(poly(x,3),compute_uv=False),np.linalg.svd(poly(z,3),compute_uv=False));
# Compare independent feature families with the exact frozen regressor.
import importlib.util
sp=importlib.util.spec_from_file_location('reg','/holdout_task/_frozen_nodes/node_02.py'); M=importlib.util.module_from_spec(sp);sp.loader.exec_module(M)
c=s['panel_c']; U=c['u']; q=.25376
for name,a in {'raw':x,'Ainv':z,'Blinear':np.c_[np.ones(len(x)),x]@_linear(poly(u,1),v),'Blinear_Ainv':np.c_[np.ones(len(x)),z]@_linear(poly(u,1),v)}.items():
 f1=_interactions(a)@_linear(_interactions(U),c['v1']); f2=_curvature(a,q)@_linear(_curvature(U,q),c['v2']);
 for key,w in {'a':a,'a2':poly(a,2)[:,1:],'m':np.c_[f1,f2],'am':np.c_[a,f1,f2],'a2m':np.c_[poly(a,2)[:,1:],f1,f2]}.items():
  errs=[]
  for scale in (.25,.5,1,2):
   p=54; f=scale*(w-w[:p].mean(0))/np.maximum(w[:p].std(0),.1); r=M.run_node({'node_00':{'features':f[:n], 'n_train':p},'node_01':{'targets':y[:p]}})['fit']-y[p:]; errs.append((np.mean(abs(r[:,0])),np.mean(abs(r[:,1:]))))
  print(name,key,np.round(errs,4).tolist())
