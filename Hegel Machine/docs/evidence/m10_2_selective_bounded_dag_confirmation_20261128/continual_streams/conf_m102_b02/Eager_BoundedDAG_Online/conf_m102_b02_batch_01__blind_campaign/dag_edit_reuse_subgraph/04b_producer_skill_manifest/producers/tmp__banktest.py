exec(open('/tmp/compose.py').read().split('results=[]')[0].replace("print(p,k,fam,o.x,o.fun**.5,'coef',c.tolist());",''))
from itertools import combinations
from scipy.spatial.distance import cdist
from scipy.linalg import solve

def cv(G,ls):
 res=[]
 for lam in ls:
  rr=[]
  for j in folds:
   i=np.setdiff1d(np.arange(n),j); g=G[np.ix_(i,i)].copy(); g.flat[::len(i)+1]+=lam*np.diag(g).mean(); p=G[np.ix_(j,i)]@solve(g,tar[i],assume_a='pos'); r=abs(p-tar[j]);rr.append([r[:,0].mean(),r[:,1:].mean()])
  res.append((max(np.max(rr,axis=0)),lam,np.r_[rr[0],np.max(rr,axis=0)]))
 return min(res,key=lambda a:a[0])
for name in ('vodd','vtanh','v1asinh','v2even','v1','v11','v21'):
 if name in bank:
  f=bank[name]; print('MAP',name,'zero',f(np.zeros((1,4))).round(3),'derivative',((f(.001*np.eye(4))-f(-.001*np.eye(4)))/.002).round(3),'diag',f(np.eye(4)).round(3))
z0=x[:n]; vb=bank['vodd'](x)[:n]; v1=bank['v1asinh'](x)[:n];v2=bank['v2even'](x)[:n]
bases={'x':z0,'b':vb,'c1':v1,'c2':v2,'b12':np.c_[vb,v1,v2],'all':np.c_[z0,vb,v1,v2], 'odd':np.c_[z0,vb,v1]}
for name,v in bases.items():
 print('concat',name,[(sc,score(v*sc).round(4)) for sc in (.05,.15,.3,.6,1)])
ls=np.r_[1e-8,1e-6,1e-5,1e-4,1e-3,.01,.05,.2]
for name,z in bases.items():
 z=z-z.mean(0); z=z/z.std(0); Q=z@z.T/z.shape[1]; L=cdist(z,z,'sqeuclidean')/z.shape[1];
 for d in (1,2,3,4): print('poly',name,d,cv((1+Q)**d,ls))
 for a in (.05,.2,.5,1,2): print('rbf',name,a,cv(np.exp(-L*a),ls))
for basis in ('axis','all'):
 for scale in (.3,.6,1.0,1.5,2.5,4.,6.):
  u=np.c_[np.arcsinh(scale*z0),np.tanh(scale*z0),np.sin(scale*z0),1/(1+(scale*z0)**2)]
  f=np.c_[z0,u] if basis=='axis' else np.c_[z0,u,vb,v1,v2]; f-=f.mean(0); f/=f.std(0); Q=f@f.T/f.shape[1]
  print('nonlin',basis,scale,[cv((1+Q)**d,ls) for d in (1,2,3)])
