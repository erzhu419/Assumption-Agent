exec(open('/tmp/structure11.py').read().split("A=np.c_[np.ones(72)")[0])
P=s['panel_b']; U,V=P['u'],P['v'];
def db(x,k): return np.c_[np.ones(len(x)),x,np.cos(k*x),np.cos(k*prod(x))]
def goal(k):
 a=db(U,k);return np.mean((a@ls(a,V)-V)**2)
k=minimize_scalar(goal,bounds=(2.3,2.9),method='bounded',options={'xatol':1e-13}).x
W=ls(db(U,k),V); print('panel b',k,goal(k)**.5,'coef', W.round(5),sep='\n')
ije=np.abs(W).max(1)>.002; W2=ls(db(U,k)[:,ije],V); print('trim',ije,np.std(db(U,k)[:,ije]@W2-V,0)); B=db(X,k)[:,ije]@W2
for n in [52,60]:
 for name,a in [('b',poly(B,1)),('bquad',poly(B,2)),('bcos',np.c_[poly(B,1),np.cos(k*X)]),('rawbasis',db(X,k)),('evenraw',np.c_[np.ones(len(X)),np.cos(k*X),np.cos(k*prod(X))]),('bcoord',np.c_[poly(B,1),np.cos(k*X),prod(np.cos(k*X))])]:
  o=a@ls(a[:n],np.c_[Y,T][:n]); e=o-np.c_[Y,T]; print(n,name,np.mean(abs(e[:n]),0).round(5),np.mean(abs(e[n:]),0).round(5))
  if name in ('b','evenraw'): print('coeff',ls(a[:n],np.c_[Y,Z][:n]).round(5)); print('temporal residual SVD',np.linalg.svd(e[:n,1:],compute_uv=False).round(5))
print('raw T SVD',np.linalg.svd(T,compute_uv=False)); print('time means',T.mean(0), 'y mean/std',Y.mean(),Y.std()); print('summary/telemetry',tr['summary'],s['telemetry_vec']); print('panel calibration',D)
