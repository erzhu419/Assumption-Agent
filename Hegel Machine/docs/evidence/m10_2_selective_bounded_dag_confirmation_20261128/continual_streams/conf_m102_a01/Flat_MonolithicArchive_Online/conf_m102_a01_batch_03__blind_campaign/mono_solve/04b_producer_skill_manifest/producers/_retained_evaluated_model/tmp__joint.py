import sys,pickle,numpy as np
from itertools import combinations_with_replacement as comb
from scipy.linalg import svd
sys.path.insert(0,'/tmp/hegel_continual_archive'); import conf_m102_a01_batch_00_solver as prior
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; T=s['train']['traj']; z,h=prior._coordinates(s,x); n=52
np.set_printoptions(precision=4,suppress=True)
def poly(x,d):return np.stack([np.prod(x[:,i],axis=1) for k in range(d+1) for i in comb(range(x.shape[1]),k)],1)
def test(F,b=y):
 C=np.linalg.lstsq(F[:n],b[:n],rcond=None)[0]; E=F@C-b;return np.mean(np.abs(E[n:])),np.std(E[:n]),C
_,_,V=svd(T[:n],full_matrices=False); U=T@V[:2].T; Q=poly(x,2);
for name,F in [('Q',Q),('Q+U',np.c_[Q,U]),('Q+U2',np.c_[Q,poly(U,2)[:,1:]]),('Q+xU',poly(np.c_[x,U],2)),('Q+tanhU',np.c_[Q,np.tanh(U)]),('Q+U+xU',np.c_[Q,U,(x[:,:,None]*U[:,None,:]).reshape(len(x),-1)])]:
 e,r,c=test(F);print(name,e,r,c[-6:])
for tag,w in [('raw',x),('uncal',z),('inv',h)]:
 p=s['panel_b']; B=np.linalg.lstsq(poly(p['u'],1),p['v'],rcond=None)[0]; fw=poly(w,1)@B
 for label,W in [(tag,w),(tag+'B',fw)]:
  for nm,F in [('pure3',W**3),('squares*x',np.einsum('ni,nj->nij',W,W**2).reshape(len(W),-1)),('pure3+4',np.c_[W**3,W**4]),('pure5',np.c_[W**3,W**4,W**5]),('softsign3', W/(1+.85*abs(W))),('rational cross',np.column_stack([(W[:,i]*W[:,j])/(1+.22*W[:,k]**2) for i,j in comb(range(4),2) for k in range(4)]))]:
   print(label,nm,np.round(test(np.c_[Q,F])[:2],5),np.round(test(np.c_[Q,F],U)[:2],5))
print('quad U residual axes',np.linalg.svd(U-Q@np.linalg.lstsq(Q,U,rcond=None)[0],compute_uv=False));print('U from linear',test(poly(x,1),U)[:2]);print('yU coeff',test(np.c_[Q,U])[2]); print('T spectral',V[:2]);
