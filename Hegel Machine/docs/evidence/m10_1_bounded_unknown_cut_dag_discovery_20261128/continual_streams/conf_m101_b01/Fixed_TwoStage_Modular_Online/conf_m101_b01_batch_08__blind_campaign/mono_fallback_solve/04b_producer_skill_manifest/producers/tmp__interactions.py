from compose import *
from sklearn.linear_model import ElasticNet, ARDRegression
from itertools import combinations_with_replacement
from scipy.spatial.distance import cdist
p=s['train']; Y=np.c_[p['y'],p['traj']]; z=inverse(afit(s)['diagsecond'],p['x']); op=operators(s)
u,v=[s['panel_b'][k] for k in ('u','v')]
def Bfit(par):
 f,a,g,h=par
 fun=lambda x:x+a*np.sin(f*x)+g*np.tanh(h*x)
 w=ls(aff(fun(u)),v)
 return lambda x:aff(fun(x))@w
op['B']=Bfit(least_squares(lambda par:(Bfit(par)(u)-v).ravel(),[3.,-.3,-.05,2.],max_nfev=150).x)
P={'z':z}
for k in ['b','c','d','bc','bd','cb','cd','db','dc','bcd','bdc']:
 P[k]=op[k[-1].upper()](P[k[:-1] or 'z'])
folds=[np.arange(52,72)]+[np.random.default_rng(i).permutation(72)[:20] for i in range(64,73)]
res=[]
for keys in [('z','bc','bd'),('z','b','bc','bd'),('z','bc','bd','bcd'),('z','b','bc','bd','bcd'),('z','bc','bcd'),('z','bc','bd','bdc'),('z','bc','bd','c','d'),('z','c','d'),('b','bc','bd'),('z','b','bc'),('z','bc'),('z','bd')]:
 for typ in ['full','cross','within','matched','cyclic','kernel']:
  A=np.column_stack([P[k] for k in keys]); N=A.shape[1]; ii,jj=np.triu_indices(N); I,J=ii//4,jj//4; i,j=ii%4,jj%4
  sel={'full':np.ones(len(i),bool),'cross':I!=J,'within':I==J,'matched':(I==J)|(i==j),'cyclic':(I==J)|(i==j)|((i-j)%4==1)|((j-i)%4==1),'kernel':np.ones(len(i),bool)}[typ]; ii,jj=ii[sel],jj[sel]
  for model in (['k0.0002','k0.001','k0.005','k0.02'] if typ=='kernel' else ['r0.005','r0.05','r0.4','r2','e0.0005','e0.002']):
   e=[]
   for te in folds:
    tr=np.setdiff1d(np.arange(72),te); a=(A-A[tr].mean(0))/A[tr].std(0); bn=Y[tr].mean(0); yy=Y[tr]-bn; reg=float(model[1:]);
    if model[0]=='k':
     K=(1+a@a.T/(N*1.5))**2; pr=K[np.ix_(te,tr)]@np.linalg.solve(K[np.ix_(tr,tr)]+reg*np.eye(len(tr)),yy)+bn
    else:
     d=np.c_[a,a[:,ii]*a[:,jj]]; d=(d-d[tr].mean(0))/d[tr].std(0).clip(1e-8); X=d[tr];
     if model[0]=='r':pr=d[te]@X.T@np.linalg.solve(X@X.T+reg*np.eye(len(tr)),yy)+bn
     else:
      _,_, vt=np.linalg.svd(yy[:,1:],full_matrices=False); v0=vt[0]; q=np.c_[yy[:,0],yy[:,1:]@v0]; est=ElasticNet(alpha=reg,l1_ratio=.92,max_iter=3500,tol=.0003); yp=[est.fit(X,col).predict(d[te]) for col in q.T]; pr=np.c_[yp[0],yp[1][:,None]*v0]+bn
    E=abs(pr-Y[te]); e.append([E[:,0].mean(),E[:,1:].mean()]);
   res.append((keys,typ,model,np.array(e)))
for j in range(2):
 for sort in [lambda e:e[1:].mean(0)[j],lambda e:e[:,j].max(),lambda e:e[0,j]]:
  print(j,[(r[:3],r[-1][0].round(4).tolist(),r[-1][1:].mean(0).round(4).tolist(),r[-1].max(0).round(4).tolist()) for r in sorted(res,key=lambda r:sort(r[-1]))[:6]])
A=np.column_stack([P[k] for k in ('z','bc','bd')]); i,j=np.triu_indices(12); F=np.c_[A,A[:,i]*A[:,j]]; sc=F.std(0); F=(F-F.mean(0))/sc; names=[f'{k}{j}' for k in ('z','c','d') for j in range(4)]; names=names+[names[x]+'*'+names[y] for x,y in zip(i,j)];
for target in [Y[:,0],Y[:,-1]]:
 for alpha in [.0004,.002]:
  m=ElasticNet(alpha=alpha,l1_ratio=.95,max_iter=8000).fit(F,target); print('coeff',alpha,[(names[k],round(m.coef_[k]/sc[k],3)) for k in np.argsort(abs(m.coef_))[-16:][::-1]])
