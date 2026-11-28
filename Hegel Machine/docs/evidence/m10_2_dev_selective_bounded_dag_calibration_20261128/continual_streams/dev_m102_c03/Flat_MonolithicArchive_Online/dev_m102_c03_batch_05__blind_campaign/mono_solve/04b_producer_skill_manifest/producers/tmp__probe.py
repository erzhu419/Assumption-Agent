import pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
from scipy.linalg import qr
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; d=s['train']; x=d['x']; Y=np.c_[d['y'],d['traj']]; t=s['eval_times']; fit=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-9)[0]; aff=lambda z:np.c_[np.ones(len(z)),z]; mae=lambda a:np.mean(abs(a),axis=0)
ps={}
for panel in ('panel_a','panel_b','panel_c'):
 u=s[panel]['u']; A=aff(u)
 for key,v in s[panel].items():
  if key=='u':continue
  mask=np.ones(len(u),bool)
  for _ in range(12):
   B=fit(A[mask],v[mask]); err=np.linalg.norm(A@B-v,axis=1); mask=err<=max(.001,3*np.median(err))
  print(panel,key,'affine med',np.median(err),'inliers',mask.sum(),'B',np.round(B,4)); ps[panel]=B
  for p in (2,3):
   P=np.column_stack([A]+[np.prod(u[:,I],axis=1) for k in range(2,p+1) for I in combinations_with_replacement(range(4),k)]); print('poly',p,P.shape,np.median(abs(P@fit(P,v)-v)), 'loo',np.mean([abs(P[i]@fit(np.delete(P,i,0),np.delete(v,i,0))-v[i]).mean() for i in range(len(u))]))
A=ps['panel_a']; B=ps['panel_b']; z=(x-A[0])@np.linalg.inv(A[1:]); b=aff(z)@B
q=lambda p:-np.expm1(-t[:,None]*p); traj=Y[:,1:]; rates=least_squares(lambda p:(q(p)@fit(q(p),traj[:52].T)-traj[:52].T).ravel(),[.42,1.83],bounds=([.01,.8],[1.2,8])).x; print('rates',rates,mae((q(rates)@fit(q(rates),traj.T)-traj.T).T)); Q=q(rates); amps=fit(Q,traj.T).T; yy=np.c_[Y[:,0],amps];
for name,a in [('x',x),('z',z),('b',b),('b_raw',aff(x)@B),('b_inv',(z-B[0])@np.linalg.inv(B[1:]))]:
 for kind in ['aff','quad','cubic','tanh','soft','sin','cos','mix']:
  V={'aff':aff(a),'quad':np.c_[aff(a),*[a[:,i]*a[:,j] for i,j in combinations_with_replacement(range(4),2)]],'cubic':np.c_[aff(a),*[np.prod(a[:,I],axis=1) for p in (2,3) for I in combinations_with_replacement(range(4),p)]],'tanh':np.c_[aff(a),np.tanh(a)],'soft':np.c_[aff(a),a/(1+abs(a))],'sin':np.c_[aff(a),np.sin(a)],'cos':np.c_[aff(a),np.cos(2.4*a)],'mix':np.c_[aff(a),np.tanh(a),np.cos(2.4*a),*[a[:,i]*a[:,j] for i,j in combinations_with_replacement(range(4),2)]]}[kind]; pred=V[52:]@fit(V[:52],yy[:52]); print(name,kind,np.round(mae(pred-yy[52:]),4),round(mae(pred[:,1:]@Q.T-traj[52:]).mean(),4))
