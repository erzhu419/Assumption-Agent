import pickle,numpy as np,itertools
from scipy.linalg import svd
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; T=s['train']['traj']; n=len(x); Y=np.c_[y,T]; np.set_printoptions(precision=4,suppress=True,linewidth=140)
def aff(u): return np.c_[np.ones(len(u)),u]
def ls(x,y): return np.linalg.lstsq(x,y,rcond=None)[0]
def poly(x,d): return np.array([np.prod(x[:,p],axis=1) for k in range(d+1) for p in itertools.combinations_with_replacement(range(x.shape[1]),k)]).T
a=s['panel_a']; U=aff(a['u']); V=a['v']; ids=np.arange(len(U));
for _ in range(8):
 A=ls(U[ids],V[ids]); ids=np.argsort(np.linalg.norm(V-U@A,axis=1))[:36]
print('A',A,'A residual row norms',np.sort(np.linalg.norm(V-U@A,axis=1))); z=aff(x)@A
b=s['panel_b']; B=ls(aff(b['u']),b['v']); print('B singular values',svd(B[1:],compute_uv=False));
c=s['panel_c']; C=[]
for k in ['v1','v2']:
 for d in [1,2,3]:
  F=poly(c['u'],d); w=ls(F[:19],c[k][:19]); print(k,d,np.sqrt(np.mean((F[19:]@w-c[k][19:])**2)), 'coeff',w.T if d==2 else '')
 C.append(ls(aff(c['u']),c[k]));
P=poly(z,2); w=ls(P[:52],Y[:52]); print('quadratic in calibrated coordinates',w[:,[0,8]].T,'residual sv',svd(Y[:52]-P[:52]@w,compute_uv=False));
print('quadratic y residual by input:'); print(np.c_[z,y,y-P@w[:,0]][np.argsort(y-P@w[:,0])][::4]);
Xs={'raw':x,'a':z,'b':aff(z)@B,'c1':aff(z)@C[0],'c2':aff(z)@C[1]};
splits=[(np.arange(52),np.arange(52,72))]+[(np.setdiff1d(np.arange(72),j),j) for j in np.array_split(np.random.default_rng(13).permutation(72),4)]
def score(F,pen):
 # unpenalized quadratic trend, optionally regularized higher order or reduced-rank features
 Q=poly(z,2); F=np.asarray(F,float); F=F/np.maximum(F.std(0),1e-8); K=F@F.T; E=[]
 for i,j in splits:
  q,r=np.linalg.qr(Q[i],mode='complete'); N=q[:,Q.shape[1]:]; v=N@np.linalg.solve(N.T@K[np.ix_(i,i)]@N+pen*np.eye(N.shape[1]),N.T@Y[i]); beta=ls(Q[i],Y[i]-K[np.ix_(i,i)]@v); err=np.abs(Q[j]@beta+K[np.ix_(j,i)]@v-Y[j]); E.append([err[:,0].mean(),err[:,1:].mean(),err[:,-1].mean()]);
 return np.array(E)
for k,d in Xs.items():
 for label,F in [('diag3',d**3),('diag34',np.c_[d**3,d**4]),('poly4',poly(d,4)[:,15:]),('trig',np.c_[np.sin(d),np.cos(d),np.sin(1.8*d),np.cos(1.8*d)]),('pairs',np.c_[*[np.sin(d[:,i,None]+d[:,j,None]*np.array([-1,1])) for i,j in itertools.combinations(range(4),2)],*[np.cos(d[:,i,None]+d[:,j,None]*np.array([-1,1])) for i,j in itertools.combinations(range(4),2)]])]:
  out=[score(F,p) for p in [.001,.03,.3,3]]; print(k,label,[(a[0].round(4).tolist(),a[1:].max(0).round(4).tolist()) for a in out]);
