import os,pickle,runpy,numpy as np
from scipy.optimize import minimize_scalar
from itertools import combinations_with_replacement
os.chdir('/holdout_task');s=pickle.load(open('public_batch.pkl','rb'))[0]['study']; en=runpy.run_path('_frozen_nodes/node_00.py')['run_node']({'__input__':s})
def fit(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def aff(x):return np.c_[np.ones(len(x)),x]
def panel(k,v,fn,freq):
 p=s[k]; D=aff(fn(freq*p['u'])); C=fit(D,p[v]); print(k,v,freq,'RMSE',np.sqrt(np.mean((D@C-p[v])**2)),'C',np.round(C,4)); return lambda x: aff(fn(freq*x))@C
b=s['panel_b']; B=fit(aff(b['u']),b['v']);u=np.linalg.solve(B[1:].T,(en['z']-B[0]).T).T
for f in (1.25,2.15,2.5,2.8,2.9,3.):panel('panel_b','v',np.sin,f)
f=minimize_scalar(lambda f: np.mean((aff(np.sin(f*b['u']))@fit(aff(np.sin(f*b['u'])),b['v'])-b['v'])**2),bounds=(.1,3.5),method='bounded').x
z=panel('panel_b','v',np.sin,f)(u)
v=panel('panel_c','v1',np.tanh,2.15)(z); w=panel('panel_c','v2',np.cos,1.25)(z)
T=s['train']['traj']; _,_,vt=np.linalg.svd(T[:52],full_matrices=False); a=T@vt[0]; tar=np.c_[s['train']['y'],a];print('vt0',vt[0]);print('base correlations',np.round(np.corrcoef(np.c_[tar,u,z,v,w].T)[:2],3))
def powx(x,d):return np.column_stack([np.prod(x[:,ix],axis=1) for ix in combinations_with_replacement(range(x.shape[1]),d)])
def check(n,x):
 C=fit(x[:52],tar[:52]); e=tar[52:]-x[52:]@C
 print(n,x.shape[1], 'cv',np.round(np.mean(abs(e),axis=0)*[1,np.mean(abs(vt[0]))],5),'fit',np.round(np.mean(abs(tar[:52]-x[:52]@C),axis=0)*[1,np.mean(abs(vt[0]))],5),'coef',np.round(C.T,3) if x.shape[1]<=10 else '')
for name,x in [('u',u),('affz',en['z']),('z',z),('v',v),('w',w)]:
 A=aff(x);check(name,A);check(name+' quad',np.c_[A,powx(x,2)]);check(name+' cubic',np.c_[A,powx(x,2),powx(x,3)]);
 for h in [.22,1.25,2.15,3]:
  check(name+' tanh'+str(h),np.c_[A,np.tanh(h*x)]);check(name+' tanh-only'+str(h),aff(np.tanh(h*x)))
check('joint',np.c_[aff(z),v,w]);check('joint raw cross',np.c_[aff(z),v,w,z*v,z*w,v*w])
