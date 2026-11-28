import sys,pickle,numpy as np
from itertools import combinations_with_replacement
sys.path.insert(0,'/tmp/hegel_continual_archive')
import conf_m102_a03_batch_01_solver as old
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; t=s['train']['traj']; fit=old._fit
np.set_printoptions(precision=5,suppress=True)
def poly(X,d):
 return np.column_stack([np.ones(len(X))]+[np.prod(X[:,i],axis=1) for n in range(1,d+1) for i in combinations_with_replacement(range(X.shape[1]),n)])
def cv(F,Y=y):
 yh=F[52:]@fit(F[:52],Y[:52]);return np.mean(abs(yh-Y[52:]))
A=old._calibration(s['panel_a']); print('A',A)
u=s['panel_a']['u']; v=s['panel_a']['v']; R=v-old._cal_features(u)@A if hasattr(old,'_cal_features') else None
z=old._decode(x,A); print('z ranges',z.min(0),z.max(0))
pb=s['panel_b']; B=fit(poly(pb['u'],1),pb['v']); w=poly(z,1)@B
for k,F in [('raw',x),('decode',z),('operator',w)]:
 print(k,'poly y',[(d,cv(poly(F,d))) for d in [1,2,3]]);print(k,'poly traj',[(d,cv(poly(F,d),t)) for d in [1,2,3]]);print(k,'y weights',fit(poly(F,2)[:52],y[:52]));print(k,'traj coeff last',fit(poly(F,1)[:52],t[:52,-1])); print(k,'bilinear',cv(old._readout_features(F)))
pc=s['panel_c']; U=pc['u']
for field in ('v1','v2'):
 V=pc[field]; C=fit(poly(U,2),V); print(field,'poly2 LOO mae',np.mean(abs(poly(U,2)@C-V)/(1-np.diag(poly(U,2)@np.linalg.pinv(poly(U,2))))[:,None]));print(C)
 for k,F in [('raw',x),('decode',z),('operator',w)]:
  Q=poly(F,2)@C;print(k,field,'C-only and combined y',cv(poly(Q,1)),cv(np.c_[poly(F,1),Q]),'traj',cv(poly(Q,1),t),cv(np.c_[poly(F,1),Q],t))
