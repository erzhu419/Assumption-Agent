import numpy as np,pickle
from itertools import combinations_with_replacement as cw
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; t=s['train']['traj']; Y=np.c_[y,t]; np.set_printoptions(precision=4,suppress=True,linewidth=150)
def feat(x,d): return np.column_stack([np.prod(x[:,ix],axis=1) for k in range(d+1) for ix in cw(range(x.shape[1]),k)])
def fit(x,Y,d=1): return np.linalg.lstsq(feat(x,d),Y,rcond=None)[0]
def cv(F,Y,l=0):
 C=[]
 for i in range(4):
  tr=np.arange(72)<52 if i==0 else np.arange(72)//18!=i-1
  a,b=F[tr],Y[tr]; c=np.linalg.lstsq(a,b,rcond=None)[0] if not l else np.linalg.solve(a.T@a+l*np.eye(a.shape[1]),a.T@b); e=np.abs(F[~tr]@c-Y[~tr]); C.append([e[:,0].mean(),e[:,1:].mean()])
 return np.asarray(C).T
for p in (s['panel_a'],s['panel_b'],s['panel_c']):
 u=p['u'];
 for k,v in p.items():
  if k=='u': continue
  for d in (1,2):
   a=fit(u,v,d); print('panel',k,d,np.sqrt(np.mean((feat(u,d)@a-v)**2,axis=0))); print(a if d==1 else '')
print('std',Y.std(0),'trajectory sv',np.linalg.svd(t,compute_uv=False),'end+traj sv',np.linalg.svd(Y,compute_uv=False)); print('ranges',x.min(0),x.max(0),s['test']['x'].min(0),s['test']['x'].max(0))
for d in range(1,5):
 F=feat(x,d); print('raw',d,'ols',cv(F,Y),'ridge',[(l,cv(F,Y,l).max(1)) for l in (.003,.03,.3,3)])
 if d<3:
  print('coef',fit(x,Y,d)); print('resid sv',np.linalg.svd(F@fit(x,Y,d)-Y,compute_uv=False))
for k in (1,2,3,4,6,8):
 _,_,b=np.linalg.svd(t,full_matrices=False); print('y from traj modes',k,[(d,cv(feat(t@b[:k].T,d),Y)[0]) for d in range(1,4)])
print('horizontal first rows',Y[:12])
