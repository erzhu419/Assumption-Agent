import pickle,numpy as np,importlib.util
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def load(p):
 spec=importlib.util.spec_from_file_location('m',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
old=load('/tmp/hegel_continual_archive/dev_m102_c02_batch_00_solver.py'); new=load('/tmp/hegel_continual_archive/dev_m102_c02_batch_08_solver.py'); sink=load('/holdout_task/_frozen_nodes/node_01.py')
a,b=s['panel_a'],s['panel_b']; u,v=a['u'],a['v']; x=s['train']['x']; lin=new._lin; ls=new._ls
m=old._panel_maps(a,b); z0=(x-m[0])/m[1];z1=new._cal(a,x)
print('A diagonal parameters:',m[:2]);print('A diagonal residual sorted:',np.sort(np.linalg.norm(v-m[0]-u*m[1],axis=1)).round(4))
w=np.roll(u,-1,axis=1); d=np.c_[lin(u),u*w,u**3]; e=np.linalg.norm(v-d@ls(d,v),axis=1);keep=np.argsort(e)[:-3]
for name,ds in [('diag',lambda j:lin(u[:,j,None])),('cyclic',lambda j:np.c_[np.ones(len(u)),u[:,j],w[:,j],u[:,j]*w[:,j],u[:,j]**3]),('full',lambda j:d)]:
 pred=np.column_stack([ds(j)@ls(ds(j)[keep],v[keep,j]) for j in range(4)]); print('A',name,np.sort(np.linalg.norm(v-pred,axis=1)).round(4))
print('calibrated extremes:',z0.min(0),z0.max(0),z1.min(0),z1.max(0))
u,v=b['u'],b['v']
for name,D in [('affine',lambda z,a:lin(z)),('rational',lambda z,a:lin(z/(1+a*z*z))),('rat+linear',lambda z,a:np.c_[lin(z/(1+a*z*z)),z]),('atan',lambda z,a:lin(np.arctan(a*z))),('tanh',lambda z,a:lin(np.tanh(a*z))),('poly3',lambda z,a:np.c_[lin(z),z**3])]:
 def obj(a):
  d=D(u,a); h=np.sum(np.linalg.qr(d,mode='reduced')[0]**2,axis=1);return np.mean(((v-d@ls(d,v))/(1-h[:,None]))**2)
 a=minimize_scalar(obj,bounds=(.03,10),method='bounded').x; d=D(u,a); beta=ls(d,v);print('B',name,'a',round(a,5),'fit, loo',np.sqrt(np.mean((v-d@beta)**2)),np.sqrt(obj(a)))
 for tag,z in [('A diagonal',z0),('A cyclic',z1)]:
  c=D(z,a)@beta;n=52;y=s['train']['y'];t=s['train']['traj'];times=s['eval_times']; yp=sink._scalar_readout(c[:n],c[n:],y[:n]);tp=sink._trajectory(c[:n],c[n:],t[:n],times);print(' ',tag,np.mean(abs(yp-y[n:])),np.mean(abs(tp-t[n:])))
