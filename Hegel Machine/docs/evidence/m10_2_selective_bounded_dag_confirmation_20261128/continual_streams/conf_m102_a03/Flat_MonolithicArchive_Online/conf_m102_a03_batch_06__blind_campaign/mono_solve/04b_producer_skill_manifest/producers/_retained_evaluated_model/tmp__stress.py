import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task'); import batch_solver as b
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=S['train']; y=tr['y']; T=tr['traj']; z=b._decode(tr['x'],b._calibration(S['panel_a'])); ix=np.arange(52); val=np.arange(52,72)
from itertools import combinations_with_replacement
pairs=list(combinations_with_replacement(range(4),2))
def pq(h):return np.column_stack([h[:,i]*h[:,j] for i,j in pairs])
def setup(i,j):
 def pp(p):
  q=b._mixed(z[i],b._warp(z[i],p)); return q@b._fit(q,y[i])-y[i]
 p=min((least_squares(pp,k,bounds=([0,.5],[.75,4]),max_nfev=100) for k in ([.22,1.25],[.35,1.5],[.5,2])),key=lambda r:r.cost).x
 h=b._warp(z,p); h0=b._warp(z,[.42,1.25]); Q=b._mixed(z,h); G=np.c_[b._mixed(z,h0),pq(z)]; D={'profile':b._ridge(Q[i],Q[j],y[i],.006),'fixed':b._ridge(G[i],G[j],y[i],.03)}
 def fit(Q,a=.006):return b._ridge(Q[i],Q[j],y[i],a)
 for a in (0,.002,.01,.03,.1):
  for k,X in [('mixed',Q),('expand',np.c_[Q,pq(z)]),('simple',np.c_[Q[:,:19],pq(z)]),('hybrid',np.c_[Q,pq(z),h0])]:
   sc=np.sqrt((X[i]**2).mean(0)).clip(1e-8); xx=X/sc; D[k,a]=xx[j]@np.linalg.lstsq(xx[i].T@xx[i]+a*np.eye(xx.shape[1]),xx[i].T@y[i],rcond=1e-12)[0]
 # Adaptive shrinkage of higher-order mixed interactions.
 for k,X in [('mix',Q),('full',np.c_[Q,pq(z)]),('fixed',G)]:
  sc=np.sqrt((X[i]**2).mean(0)).clip(1e-8); xx=X/sc; I=xx[i]; J=xx[j]; c=np.linalg.lstsq(I.T@I+.03*np.eye(xx.shape[1]),I.T@y[i],rcond=None)[0]
  for a in (.0003,.003,.03,.1):
   for power in (1,2):
    cc=c.copy()
    for _ in range(5):
     rr=a/(np.abs(cc)+.035)**power; rr[:9]*=.1; cc=np.linalg.solve(I.T@I+np.diag(rr),I.T@y[i])
    D['adapt',k,a,power]=J@cc
 return D
D=[]; splits=[(ix,val),*KFold(5,shuffle=True,random_state=41).split(z),*KFold(5,shuffle=True,random_state=76).split(z)]
for i,j in splits:
 d=setup(i,j); h=b._warp(z,[.35,1.5]); q=b._mixed(z,h); rho=b._ridge(q[i],q[j],T[i],.015); print('baseline',np.mean(abs(d['profile']-y[j])),np.mean(abs(rho-T[j])), 'max-time',abs(rho-T[j]).mean(0).max()); D.append((d,y[j]))
R=[]
for k in D[0][0]:
 for l in ('profile','fixed'):
  for a in (0,.25,.5,.75,1):
   e=[np.mean(abs((1-a)*d[k]+a*d[l]-v)) for d,v in D]; R.append((e[0],np.mean(e[1:]),max(e[1:]),str(k),l,a))
print('gate',*sorted(R)[:10],sep='\n');print('cv',*sorted(set(R),key=lambda x:x[1])[:16],sep='\n');print('minimax',*sorted(set(R),key=lambda x:max(x[:3]))[:10],sep='\n')
