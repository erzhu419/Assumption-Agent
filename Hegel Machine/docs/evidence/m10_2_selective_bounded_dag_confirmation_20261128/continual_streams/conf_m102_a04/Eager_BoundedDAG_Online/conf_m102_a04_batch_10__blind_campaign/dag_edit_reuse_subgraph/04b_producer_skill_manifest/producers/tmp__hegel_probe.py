import importlib.util, pickle, numpy as np
from scipy.optimize import least_squares
np.set_printoptions(precision=4,suppress=True,linewidth=130)
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
a=load('arc','/tmp/hegel_continual_archive/conf_m102_a04_batch_00_solver.py')
f=[load('f'+str(i),f'/holdout_task/_frozen_nodes/node_0{i}.py') for i in [1,2,3]]
def study(t,v):
 return dict(p,train={k:p['train'][k][t] for k in ['x','y','traj']},test={'x':p['train']['x'][v]})
def score(w,s):
 n=len(s['train']['x']); z=f[0].run_node({'__input__':s}); fit=f[1].run_node({'node_00':{'features':w,'n_train':np.array(n)},'node_01':z}); o=f[2].run_node({'node_01':z,'node_02':fit}); return np.mean(abs(o['y_hat']-p['train']['y'][vi])),np.mean(abs(o['state_pred']-p['train']['traj'][vi]))
if __name__=='__main__':
 for k in ['panel_a','panel_b']:
  u,v=p[k]['u'],p[k]['v']; q,b=a._align(p[k]); print(k,'offset scale',q,b); print('u range',u.min(0),u.max(0),'v range',v.min(0),v.max(0)); print('diag residual quantiles',np.quantile(abs(v-u*b-q),[0,.25,.5,.65,.75,.9,.95,1],axis=0)); print('affine',np.linalg.lstsq(np.c_[np.ones(len(u)),u],v,rcond=None)[0]); print('worst rows',np.sort(np.linalg.norm(v-u*b-q,axis=1)))
 u=p['panel_c']['u']; w=a._panel_maps(p['panel_c'],u); print('archived c RMS',np.sqrt(np.mean((w-np.c_[p['panel_c']['v1'],p['panel_c']['v2']])**2,axis=0)))
 vi=np.arange(52,72); s=study(np.arange(52),vi); x=np.vstack([s['train']['x'],s['test']['x']]); qa,ba=a._align(p['panel_a']); qb,bb=a._align(p['panel_b']);
 for k,z in {'raw':x,'a inverse':(x-qa)/ba,'b inverse':(x-qb)/bb,'b direct':x*bb+qb,'a inv b inv':((x-qa)/ba-qb)/bb,'a inv b direct':((x-qa)/ba)*bb+qb,'b inv a inv':((x-qb)/bb-qa)/ba,'b direct a inv':(x*bb+qb-qa)/ba}.items():
  for name,w in [('raw',z),('square',np.c_[z,z*z]),('archived',a._panel_maps(p['panel_c'],z)),('archived + raw',np.c_[z,a._panel_maps(p['panel_c'],z)])]:
   w=(w-w[:52].mean(0))/np.maximum(w[:52].std(0),.1);print(k,name,score(w,s))
