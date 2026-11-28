import pickle,numpy as np
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; t=s['train']['traj']; n=52
np.set_printoptions(precision=5,suppress=True,linewidth=130)
def affine(p,k='v',robust=False):
 u=p['u']; v=p[k]; a=np.c_[np.ones(len(u)),u]; ix=np.arange(len(u));
 for _ in range(6 if robust else 1):
  b=np.linalg.lstsq(a[ix],v[ix],rcond=None)[0]; r=np.linalg.norm(v-a@b,axis=1); ix=np.flatnonzero(r<=max(.0007,np.median(r)*3))
 return b,r
A,ra=affine(s['panel_a'],robust=True); B,rb=affine(s['panel_b']); print('times',s['eval_times'],'telemetry',s['telemetry_vec']); print('A',A,'resid',np.sort(ra)); print('B',B,'svd',np.linalg.svd(B[1:],compute_uv=False)); print('B offset',B[0]); z=(x-A[0])@np.linalg.inv(A[1:]); U,V=np.linalg.svd(t,full_matrices=False)[::2]; T=t@V[:2].T; print('temporal V',V[:2]);
def poly(q,d): return np.column_stack([np.ones(len(q))]+[np.prod(q[:,ids],axis=1) for h in range(1,d+1) for ids in combinations_with_replacement(range(q.shape[1]),h)])
def check(f,Y):
 b=np.linalg.lstsq(f[:n],Y[:n],rcond=None)[0]; r=Y-f@b; return [np.mean(abs(r[:n]),axis=0),np.mean(abs(r[n:]),axis=0)]
for key,q in [('raw',x),('inverse_a',z),('raw_b',x@B[1:]+B[0]),('cal_b',z@B[1:]+B[0])]:
 print(key,'q range',q.min(0),q.max(0));
 for d in range(1,5):print('poly',d,check(poly(q,d),np.c_[y,T]));
 for k in [.5,1,1.5,2,2.5,3,4]:
  f=np.c_[poly(q,1),np.sin(k*q),np.cos(k*q),np.tanh(k*q)]; print('univ',k,check(f,np.c_[y,T]));
print('linear x slopes Y,T',np.linalg.lstsq(poly(x,1),np.c_[y,T],rcond=None)[0])
