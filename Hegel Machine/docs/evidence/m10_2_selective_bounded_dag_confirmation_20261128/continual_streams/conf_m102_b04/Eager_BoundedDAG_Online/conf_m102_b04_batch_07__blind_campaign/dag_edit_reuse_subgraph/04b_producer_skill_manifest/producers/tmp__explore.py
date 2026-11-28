import numpy as np,pickle
from itertools import combinations_with_replacement as cwr
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; z=s['train']['traj']; t=s['eval_times']; Y=np.c_[y,z]
def basis(x,d): return np.column_stack([np.ones(len(x))]+[np.prod(x[:,j],axis=1) for k in range(1,d+1) for j in cwr(range(x.shape[1]),k)])
def fit(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
a,b,c=(s['panel_'+k] for k in 'abc'); A=basis(a['u'],1); fa=fit(A,a['v']); B=basis(b['u'],1); fb=fit(B,b['v']); C=basis(c['u'],1); fc=fit(C,c['v1']); fd=fit(C,c['v2']);
for _ in range(10):
 ii=np.argsort(np.linalg.norm(A@fa-a['v'],axis=1))[:39]; fa=fit(A[ii],a['v'][ii]);
xa=(x-fa[0])/fa[1:].diagonal(); sq=lambda q,k=.09:q/np.sqrt(1+k*q*q)
def cv(F,l):
 e=[]
 for j in range(4):
  idx=np.random.default_rng(j).permutation(len(x));
  for va in np.array_split(idx,4):
   tr=np.setdiff1d(idx,va); ff=F[tr]; P=F[va]@np.linalg.solve(ff.T@ff+l*np.eye(F.shape[1]),ff.T@Y[tr]); e.append([np.abs(P[:,0]-y[va]).mean(),np.abs(P[:,1:]-z[va]).mean()]);
 return np.round(np.array(e).mean(0),5)
for label,u in [('x',x),('xa',xa),('B',basis(x,1)@fb),('aB',basis(xa,1)@fb),('C',basis(x,1)@fc),('aC',basis(xa,1)@fc),('D',basis(x,1)@fd),('aD',basis(xa,1)@fd),('sqx',sq(x)),('sqB',sq(basis(x,1)@fb)),('sqaB',sq(basis(xa,1)@fb)),('sqC',sq(basis(x,1)@fc)),('sqaC',sq(basis(xa,1)@fc)),('tanhB',np.tanh(basis(x,1)@fb)),('tanhaB',np.tanh(basis(xa,1)@fb)),('tanhC',np.tanh(basis(x,1)@fc)),('tanhaC',np.tanh(basis(xa,1)@fc))]:
 print(label,[cv(basis(u,d),l) for d,l in [(1,.001),(2,.001),(3,.001),(3,.03),(4,.03)]]);
for k in ['v1','v2']:
 for d in [1,2]:
  U=basis(c['u'],d); H=U@np.linalg.pinv(U); print('panel',k,d,'rms LOO',np.sqrt(np.mean(((c[k]-H@c[k])/(1-np.diag(H))[:,None])**2)), 'train',np.sqrt(np.mean((c[k]-H@c[k])**2)))
print('x range',x.min(0),x.max(0)); print('train affine coef',fit(basis(x,1),Y)[:,[0,1,8]]); print('time',t)
