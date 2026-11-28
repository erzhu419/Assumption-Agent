exec(open('/tmp/discover.py').read().split('for p,k in')[0])
from scipy.optimize import least_squares
from scipy.linalg import qr
tr=s['train']; x=np.r_[tr['x'],s['test']['x']]; Y=np.c_[tr['y'],tr['traj']]; n=len(Y)
def invert(x):
 U,V=a['u'],a['v']; m=np.argsort(r)[:-3]; W=np.c_[np.ones(len(U)),U]; G=np.roll(U,-1,axis=1)*(U-1); D=U**3
 Q=np.zeros((len(U),4,10)); Q[:,:,0]=G; Q[:,:,1]=D
 for i in range(4): Q[:,i,2+i]=1; Q[:,i,6+i]=U[:,i]
 cc=fit(Q[m].reshape(-1,10),V[m].ravel()); print('A shared',cc,'rms',np.mean((Q[m]@cc-V[m])**2)**.5)
 g,d=cc[:2]; off,sc=cc[2:6],cc[6:]; u=(x-off)/sc
 for _ in range(25):
  R=off+sc*u+g*np.roll(u,-1,axis=1)*(u-1)+d*u**3-x
  J=np.zeros((len(x),4,4)); i=np.arange(4); J[:,i,i]=sc+g*np.roll(u,-1,axis=1)+3*d*u*u; J[:,i,(i+1)%4]=g*(u-1)
  u-=np.clip(np.linalg.solve(J,R[...,None])[...,0],-.45,.45)
 print('inverse range and max forward residual',u.min(0),u.max(0),abs(R).max()); return u
z=invert(x)
B=[]
for name,fn in [('sin',np.sin),('tanh',np.tanh),('atan',np.arctan),('soft',lambda t:t/np.sqrt(1+t*t)),('cos',np.cos)]:
 for dg in (False,True):
  U=b['u']; V=b['v']
  def phi(t,p): return np.c_[feats(t,lin),fn(t*p),t*t] if dg else np.c_[feats(t,lin),fn(t*p)]
  for shared in (True,False):
   def res(p):
    F=phi(U,p); return (F@fit(F,V)-V).ravel()
   o=min([least_squares(res,np.full(1 if shared else 4,k),bounds=(.2,4),max_nfev=130) for k in (1.1,1.7,2.7)],key=lambda o:o.cost)
   F=phi(U,o.x); coef=fit(F,V); score=np.mean(loo(F,V)); print('B',name,dg,shared,o.x,np.mean(o.fun**2)**.5,score)
   B.append((score,name+str(dg)+str(shared),phi(z,o.x)@coef,coef))
B.sort(key=lambda row:row[0]); print('best B coefficients',B[0][1],B[0][3]);
F1=lambda t:feats(t,lin+pair); F2=lambda t:feats(t,lin+diag)
c1=fit(F1(c['u']),c['v1']); c2=fit(F2(c['u']),c['v2']); print('C1',c1)
folds=[np.arange(n-20,n)]+[np.arange(j,n,6) for j in range(6)]
res=[]
def cv(tag,X):
 for lam in ([0,.005,.04] if X.shape[1]>20 else [0,.005]):
  e=[]
  for it in folds:
   ir=np.setdiff1d(np.arange(n),it); A=X[ir]; V=Y[ir]; sc=np.maximum(A.std(0),1e-4); mu=A.mean(0); A=(A-mu)/sc; Z=(X[it]-mu)/sc
   if lam: cf=np.linalg.solve(A.T@A+lam*n*np.eye(A.shape[1]),A.T@(V-V.mean(0)))
   else: cf=fit(A,V-V.mean(0))
   E=np.abs(Z@cf+V.mean(0)-Y[it]); e.append([E[:,0].mean(),E[:,1:].mean()]);
  res.append((max(np.max(e,axis=0)),e[0],np.max(e,axis=0),str(tag),lam))
for bn,g in [('z',z),('x',x)]+[(row[1],row[2]) for row in B[:10]]:
 for gn,gg in [('id',g),('tanh',np.tanh(g)),('plus',z+g),('minus',z-g)]:
  F,G=F1(gg)@c1,F2(gg)@c2; L=feats(gg,lin); Q=feats(gg,lin+q)
  for k,P in [('lin',L),('quad',Q),('V',np.c_[L,F,G]),('Vonly',np.c_[L[:,0],F,G]),('Vtanh',np.c_[L,np.tanh(F),np.tanh(G)]),('Vtanhs',np.c_[L,F,G,np.tanh(F),np.tanh(G)]),('Vmix',np.c_[L,F,G, F*G]),('quadV',np.c_[Q,np.tanh(F),np.tanh(G)]),('quadBoth',np.c_[Q,feats(z,lin+q)[:,1:]])]: cv((bn,gn,k),P)
for row in sorted(res,key=lambda t:t[0])[:24]: print('best',row)
np.savez('/tmp/cascade.npz',z=z,x=x,Y=Y,**{'b'+k:g for _,k,g,_ in B})
