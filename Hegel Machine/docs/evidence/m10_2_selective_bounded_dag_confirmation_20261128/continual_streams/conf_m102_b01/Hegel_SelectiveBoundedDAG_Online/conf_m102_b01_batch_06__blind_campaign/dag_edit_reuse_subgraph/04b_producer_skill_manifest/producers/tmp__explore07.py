import runpy,pickle,numpy as np
from scipy.optimize import least_squares
st=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; w=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')['run_node']({'__input__':st})['w']; x=st['train']['x']; y=st['train']['y']; tr=st['train']['traj']; t=st['eval_times']; p=st['panel_c']; np.set_printoptions(precision=4,suppress=True,linewidth=160)
print('w',w.min(0),w.max(0),w[:12]); print('svd traj',np.linalg.svd(tr,compute_uv=False)); print('sample',np.column_stack([y,tr])[:16]); print('y vs traj',np.linalg.lstsq(np.c_[np.ones(len(y)),tr],y,rcond=None)[0]); print('y vs tlast',np.linalg.lstsq(np.c_[np.ones(len(y)),tr[:,-1]],y,rcond=None)[0]);
for key in ['v1','v2']:
 u=p['u']; v=p[key]; print(key,'range',v.min(0),v.max(0));
 for label,U in [('raw',u),('tanh',np.tanh(u)),('sinh',np.sinh(u)),('odd',np.c_[u,u**3,u**5]),('zabs',u*np.abs(u))]:
  a=np.c_[np.ones(len(u)),U]; b=np.linalg.lstsq(a,v,rcond=None)[0]; print(label,np.sqrt(np.mean((a@b-v)**2)),np.abs(b).max());
