from spatial06 import *
from itertools import combinations_with_replacement as cwr
z,w,((a,b),_)=geom(s); y=s['train']['y']; S=s['train']['traj'];t=s['eval_times'];n=len(y);I=np.arange(56);J=np.arange(56,n)
dot=np.sum(a*b,1);cross=a[:,0]*b[:,1]; F=aff(np.c_[a,b,dot,cross]);
T=lambda p:-np.expm1(-t[:,None]*p)
p=least_squares(lambda p:(ls(T(p),S[I].T).T@T(p).T-S[I]).ravel(),[.4,1.8,5]).x; B=T(p);C=ls(B,S.T).T;Y=np.c_[y,C];
for mode in ['base','linear','cubic','sym','powers','fourth','triple_and_w','triple_and_wsin','sin_and_cube']:
 D={}
 def add(key,A):
  for j,x in enumerate(np.asarray(A).reshape(len(z),-1).T):D[key+str(j)]=x
 if mode!='base':add('z',z)
 if mode in ['cubic','fourth','powers','sym','sin_and_cube']:
  for k in [2,3]+([4] if mode=='fourth' else []):
   for idx in cwr(range(4),k):
    if mode=='powers' and len(set(idx)) not in (1,k):continue
    if mode=='sym' and k==3 and len(set(idx))==2:continue
    D['z'+''.join(map(str,idx))]=np.prod(z[:,idx],axis=1)
 if mode.startswith('triple'):
  add('zt',np.array([np.prod(z[:,i],1) for i in combinations(range(4),3)]).T)
  add('w^3',w**3);add('z^3',z**3)
  if mode.endswith('wsin'):add('w-sin',np.sin(2*w));add('w-cos',np.cos(2*w))
 if mode=='sin_and_cube':add('sin-w',np.sin(2*w));add('cos-w',np.cos(2*w));add('sin-z',np.sin(2*z));add('cos-z',np.cos(2*z))
 H=np.c_[F,*D.values()] if D else F
 A=ls(H[I],Y[I]);E=H[:n]@A-Y;print(mode,H.shape[1], 'train',np.round(np.std(E[I],axis=0),5),'y/trajectory MAE',np.mean(abs(E[J,0])),np.mean(abs(E[J,1:]@B.T)))
 if mode in ('cubic','triple_and_wsin','triple_and_w'):
  print(sorted(zip(D,np.round(A[11:,0],4)),key=lambda q:-abs(q[1]))[:24]); print('resid eig',np.linalg.svd(E[I],compute_uv=False)); print('resid corr',np.round(np.corrcoef(E[I].T),3))
# Directly infer linear shared driver relations using observed targets and temporal modes.
for features in (F,aff(np.c_[a,b]),aff(np.c_[a,b,dot]),aff(np.c_[a,b,dot,cross,z])):
 H=np.c_[features[:n], y]; Q=ls(H[I],C[I]);P=H@Q;print('conditional',features.shape[1], 'coefficients',np.round(Q,4),'mae',np.mean(abs((P-C)[J]),0))
p=s['panel_b'];A=ls(aff(p['u']),p['v']);print('B affine',A, 'rms',np.mean((aff(p['u'])@A-p['v'])**2)**.5);print('telemetry',s['telemetry_vec'])
