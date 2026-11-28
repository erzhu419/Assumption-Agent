import numpy as np, pickle
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True,linewidth=145)
def feats(x,inds): return np.column_stack([np.prod(x[:,i],axis=1) for i in inds])
lin=[(),(0,),(1,),(2,),(3,)]; q=list(combinations_with_replacement(range(4),2)); diag=[(i,i) for i in range(4)]; pair=[(0,1),(0,3),(1,2),(2,3)]
def fit(X,Y): return np.linalg.lstsq(X,Y,rcond=None)[0]
def loo(X,Y):
 p=np.linalg.pinv(X); return np.mean(abs((X@(p@Y)-Y)/(1-(X*p.T).sum(1)[:,None])),axis=0)
a,b,c=[s['panel_'+k] for k in 'abc']; A=feats(a['u'],lin); w=np.ones(len(A));
for _ in range(25):
 ca=fit(A*w[:,None]**.5,a['v']*w[:,None]**.5); r=np.linalg.norm(A@ca-a['v'],axis=1); w=1/(1+(r/(2*np.median(r)))**2)
print('A robust',ca,'residual quantiles',np.quantile(r,[0,.5,.9,.94,.96,1]));
for p,k in [(b,'v'),(c,'v1'),(c,'v2')]:
 for tag,inds in [('lin',lin),('diag',lin+diag),('pair',lin+pair),('quad',lin+q),('pure pair', [()]+pair)]: print(k,tag,loo(feats(p['u'],inds),p[k]))
 cb=fit(feats(p['u'],lin+q),p[k]); print(k,'basis',list(zip(lin+q,cb)))
 t=p['u']; print(k,'sine diag and tanh quad sweep')
 for scale in [.25,.5,.75,1,1.25,1.5,2]:
  print(scale,*[loo(np.column_stack([feats(t,inds), f(scale*t)]),p[k]).round(4) for inds in (lin,lin+diag) for f in (np.sin,np.tanh)])
x=s['train']['x']; tr=s['train']['traj']; Y=np.c_[s['train']['y'],tr]; z=(x-ca[0])@np.linalg.inv(ca[1:]);
def cv(name,X):
 e=[]
 for it in [np.arange(54,72)]+[np.arange(j,72,6) for j in range(6)]:
  ir=np.setdiff1d(np.arange(72),it); er=abs(X[it]@fit(X[ir],Y[ir])-Y[it]); e.append([er[:,0].mean(),er[:,1:].mean()]);
 print(name,np.round(e[0],5),np.round(np.max(e,axis=0),5))
for B_inds in (lin,lin+diag,lin+q):
 g=feats(z,B_inds)@fit(feats(b['u'],B_inds),b['v']);
 for cg in (z,g,np.tanh(g)):
  F=feats(cg,lin+pair)@fit(feats(c['u'],lin+pair),c['v1']); G=feats(cg,lin+diag)@fit(feats(c['u'],lin+diag),c['v2']);
  for k,P in [('g',feats(cg,lin+q)),('V',np.c_[np.ones(len(g)),F,G]),('nonlinear V',np.c_[np.ones(len(g)),np.tanh(F),np.tanh(G)]),('mixed',np.c_[feats(cg,lin),np.tanh(F),np.tanh(G)])]: cv((len(B_inds),id(cg)==id(z),id(cg)==id(g),k),P)
