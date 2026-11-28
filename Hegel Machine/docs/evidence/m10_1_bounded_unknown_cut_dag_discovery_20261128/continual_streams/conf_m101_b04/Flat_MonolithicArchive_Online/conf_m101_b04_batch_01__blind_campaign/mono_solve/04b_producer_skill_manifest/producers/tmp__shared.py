from dagprobe import *
from scipy.linalg import solve
x=s['train']['x']; y=s['train']['y']; T=s['train']['traj']; n=len(x); ii=np.arange(n); splits=[(ii[:52],ii[52:])]+[(np.setdiff1d(ii,j),j) for j in np.array_split(np.random.default_rng(13).permutation(n),4)]; D=blocks(x,s); Q=poly(x,2); t=s['eval_times']; mode=-np.expm1(-.85*t); m=T@mode/(mode@mode); print('A diag',calibrate(s)(np.r_[np.zeros((1,4)),np.eye(4)])); a=s['panel_a']; A=ls(aff(a['u']),a['v']); r=np.linalg.norm(a['v']-aff(a['u'])@A,axis=1); print('A full trimmed',ls(aff(a['u'])[np.argsort(r)[:36]],a['v'][np.argsort(r)[:36]])); print('y-m corr',np.corrcoef(y,m)[0,1]);
for i,j in splits:
 for lab,F in [('Q',Q),('Q+m',np.c_[Q,m]),('Q+m2',np.c_[Q,m,m*m]),('Q+m*x',np.c_[Q,m,m[:,None]*x]),('Q+m*t',np.c_[Q,T])]:
  r=y-F@ls(F[i],y[i]); print(lab,round(np.std(r[i]),4),round(abs(r[j]).mean(),4),end='; ')
 print()
K0=np.exp(-.23*cdist(x,x,'sqeuclidean')); rows=[]
for name,z in D.items():
 P=poly(z,4); c=z-z.mean(0); tri=np.array(list(it.combinations_with_replacement(range(4),3))); d3=P[:,15:35];
 bases={'cub':d3,'triple':d3[:,[(len(set(k))==3) for k in tri]],'mixed':d3[:,[(len(set(k))>1) for k in tri]],'pure':z**3,'cubquart':P[:,15:],'mult':np.einsum('ni,nj->nij',x,z).reshape(n,-1),'linear':z,'square':poly(z,2)[:,1:]};
 for bn,F in bases.items():
  K=(F-F.mean(0))@(F-F.mean(0)).T; K/=np.mean(K.diagonal())
  for pen in [.0003,.003,.03,.15]:
   for g in [0,.2,1]:
    for gp in [False,True]:
     es=[]
     for i,j in splits:
      N=np.linalg.qr(Q[i],mode='complete')[0][:,15:]; k=K+g*K0; ki=k[np.ix_(i,i)]; W=N.T@ki@N+pen*np.eye(N.shape[1]); Z=np.c_[y,m][i]; a=N@solve(W,N.T@Z,assume_a='pos'); b=ls(Q[i],Z-ki@a); proj=Q[j]@b+k[np.ix_(j,i)]@a; err=proj[:,0]-y[j];
      if gp:
       # Estimate a common latent component from the local trajectory residual.
       ai=N@solve(W,N.T@(Z[:,0]),assume_a='pos'); aj=N@solve(W,N.T@(Z[:,1]),assume_a='pos'); beta=(ai@aj)/(aj@aj+1e-12); kt=.55*K0+.45*K; vi=kt[np.ix_(i,i)]; ab=N@solve(N.T@vi@N+.003*np.eye(N.shape[1]),N.T@Z[:,1],assume_a='pos'); bb=ls(Q[i],Z[:,1]-vi@ab); pm=Q[j]@bb+kt[np.ix_(j,i)]@ab; err+=np.clip(beta,-1.6,1.6)*(pm-proj[:,1]);
      es.append(np.mean(abs(err)))
     rows.append((np.array(es),name,bn,pen,g,gp))
for key in [lambda r:r[0][0],lambda r:max(r[0]),lambda r:np.mean(r[0])]:
 print('rank');
 for r in sorted(rows,key=key)[:9]: print(r[0].round(5),r[1:]);
