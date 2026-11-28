exec(open('/tmp/probe.py').read().split('res=[]')[0])
from scipy.spatial.distance import cdist
from scipy.linalg import cho_solve,cho_factor
from itertools import product
n=52
B=fit(ext(b['u']),b['v']); C=fit(ext(c['u']),c['v1']);
for name,(u,v) in {'b':(b['u'],b['v']),'c1':(c['u'],c['v1']),'c2':(c['u'],c['v2'])}.items():
 w=fit(ext(u),v); print(name,'matrix',w.round(3),'singular',np.linalg.svd(w[1:],compute_uv=False).round(4))
res=[]; results={}
for label,z in list(coords.items())[:3]:
 for name,w in [('none',z),('B',(z-B[0])@np.linalg.pinv(B[1:])),('b',ext(z)@B),('C',(z-C[0])@np.linalg.pinv(C[1:])),('c',ext(z)@C)]:
  F={}
  for k in (.4,.7,1.,1.4,1.9,2.5,3.2):
   F['sin'+str(k)]=np.c_[np.sin(k*w),np.cos(k*w)]
   F['tanh'+str(k)]=np.tanh(k*w)
   F['atan'+str(k)]=np.arctan(k*w)
   F['rat'+str(k)]=np.c_[w/(1+k*w*w),w*w/(1+k*w*w)]
  K0=1+z@z.T
  for bas,kg in [('lin',K0),('poly',K0+(z@z.T)**2/4),('rbf12',K0+np.exp(-cdist(z,z,'sqeuclidean')/12)),('rbf3',K0+np.exp(-cdist(z,z,'sqeuclidean')/3))]:
   for f,G in list(F.items())+[('null',np.zeros((len(z),1)))]:
    for p in (1,2):
     if p==2 and f=='null':continue
     K=kg+((1+G@G.T/4)**p-1)*3
     for l in (1e-6,1e-4,.002,.015):
      H=cho_solve(cho_factor(K[:n,:n]+l*np.eye(n)),np.eye(n)); Y=np.c_[y,T]; a=H@Y[:n]; e=a/np.diag(H)[:,None]; q=K[n:,:n]@a; cv=np.mean(abs(q-Y[n:]),axis=0); tr=np.mean(abs(e),axis=0)
      res.append((tr[0],cv[0],np.mean(tr[1:]),np.mean(cv[1:]),(label,name,bas,f,p,l))); results[res[-1][-1]]=q
for j in range(4):
 print('best by',j,*sorted(res,key=lambda e:e[j])[:7],sep='\n')
for j in [0,2]:
 best=sorted(res,key=lambda e:e[j]);
 for m in (3,10,30,100):
  q=np.mean([results[r[-1]] for r in best[:m]],axis=0); print('ensemble',j,m,np.mean(abs(q[:,0]-y[n:])),np.mean(abs(q[:,1:]-T[n:])))
