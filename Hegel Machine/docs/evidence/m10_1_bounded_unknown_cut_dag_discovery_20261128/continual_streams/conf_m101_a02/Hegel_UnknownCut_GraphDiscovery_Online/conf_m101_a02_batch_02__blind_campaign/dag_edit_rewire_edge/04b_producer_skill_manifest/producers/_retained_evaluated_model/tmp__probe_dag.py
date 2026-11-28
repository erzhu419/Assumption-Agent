import sys,pickle,itertools as it,numpy as np
sys.path.insert(0,'/holdout_task/_frozen_nodes')
import node_00,node_01
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a=node_00.run_node({'__input__':s}); b=node_01.run_node({'node_00':a}); p=s['panel_c']; x=a['cal_tr']; z=b['z_tr']; y=s['train']['y']; r=s['train']['traj']; t=s['eval_times']
def aff(x): return np.column_stack((np.ones(len(x)),x))
def powers(x,d): return np.column_stack([np.prod(x[:,i],axis=1) for j in range(d+1) for i in it.combinations_with_replacement(range(x.shape[1]),j)])
def fit(A,B): return np.linalg.lstsq(A,B,rcond=None)[0]
def loo(A,B):
 e=B-A@fit(A,B); h=np.sum(np.linalg.qr(A,mode='reduced')[0]**2,axis=1); return np.mean(abs(e/(1-h).reshape((-1,)+(1,)*(B.ndim-1))),axis=0)
m=fit(aff(s['panel_b']['u']),s['panel_b']['v']); q=p['u']; zz=aff(q)@m
np.set_printoptions(precision=5,suppress=True)
print('mixer',m); print('C sample',np.column_stack([q,p['v1'],p['v2']])[:7]);print('corr',np.corrcoef(np.column_stack([q,zz,p['v1'],p['v2']]).T)[0:8,8:]); print('trajectory singular values',np.linalg.svd(r,compute_uv=False)); print('trajectory sample',r[:8]);print('train association y, trajectory',np.corrcoef(np.column_stack([x,z,y,r]).T)[8,:]);
for d in (1,2,3):
 print('poly',d)
 for v in ('v1','v2'):
  A=powers(q,d); B=p[v]; print(v,'max fit',np.max(abs(B-A@fit(A,B)),axis=0),'loo',loo(A,B) if A.shape[1]<len(A) else 'saturated')
 A=powers(x,d);print('y',loo(A,y),'traj',loo(A,r));
 if d==2: print('y quad coefs',fit(A,y))
print('y correlations with trajectory residual beyond affine',np.corrcoef(np.column_stack([y-aff(x)@fit(aff(x),y),r-aff(x)@fit(aff(x),r)]).T)[0,1:])
