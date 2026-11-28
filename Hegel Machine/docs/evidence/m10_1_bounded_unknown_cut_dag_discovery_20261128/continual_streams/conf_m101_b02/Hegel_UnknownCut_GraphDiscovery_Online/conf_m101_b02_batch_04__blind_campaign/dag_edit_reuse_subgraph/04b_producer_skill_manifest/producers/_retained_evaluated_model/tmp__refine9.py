exec(open('/tmp/explore.py').read().split('funcs=')[0])
from scipy.spatial.distance import cdist
from scipy.linalg import solve
from scipy.optimize import differential_evolution
ns={};exec(open('/holdout_task/node_01.py').read(),ns)
c,d=ns['_calibrate'](s['panel_c'],z)
u0,v0=s['panel_b']['u'],s['panel_b']['v']
def warp(x,k,c):return x*(1-c*np.expm1(-(k*x)**2))
def prof(k):
 X=design(u0, u0*(-np.expm1(-(k*u0)**2))); p=fit(X,v0);return np.mean((X@p-v0)**2)**.5,p
for k0 in (.1,.4,.7,1.,1.5,2.,3.,5.):
 print('B profile',k0,prof(k0)[0]);
q=minimize_scalar(lambda k:prof(k)[0],bounds=(.03,5),method='bounded').x
err,p=prof(q);print('B expanded',q,err,p.round(3));print('B ranges',u0.min(),u0.max());print('B direction ratios',np.sum(p[1:5]*p[5:9],axis=1)/np.sum(p[1:5]**2,axis=1))
Y=np.c_[y,traj]; z,b,c,d=(a[:n] for a in (z,b,c,d)); bplus=design(z,z*(-np.expm1(-(q*z)**2)))@p
folds=list(KFold(5,shuffle=True,random_state=13).split(y))+[(np.arange(n-20),np.arange(n-20,n))]
for desc,M,B in [('current',b,b),('B-expanded',bplus,bplus),('B-latent',z*(-np.expm1(-(q*z)**2)),bplus)]:
 for block,Q in [('zb',np.c_[z,M]),('quadratic',np.c_[z,M,c,d,quad(z),quad(B)]),('zbc',np.c_[z,M,c,d])]:
  for mode in ('rbf','poly','matern','blend'):
   specs=[]
   for width in (.65,1,1.55,2.4,3.7,5.7):
    for reg in (1e-6,1e-5,1e-4,.001):
     vals=[]
     for i,j in folds:
      H=(Q-Q[i].mean(0))/(.1+Q[i].std(0));G=(H@H.T)/H.shape[1];D=cdist(H,H,'euclidean')/(width*np.sqrt(H.shape[1])); K=1+G
      if mode=='rbf':K+=np.exp(-D**2/2)
      if mode=='poly':K+=(1+G/width)**3-1
      if mode=='matern':K+=(1+D+D**2/3)*np.exp(-D)
      if mode=='blend':K+=np.exp(-D**2/2)+.15*G**2
      E=abs(K[np.ix_(j,i)]@solve(K[np.ix_(i,i)]+reg*np.eye(len(i)),Y[i],assume_a='pos')-Y[j]);vals.append([E[:,0].mean(),E[:,1:].mean()]);
     Mx=np.array(vals);specs.append((Mx.max(0),Mx[-1],Mx.mean(0),width,reg))
   print(desc,block,mode,*[min((r for r in specs if r[1][j]<.038),key=lambda r:r[0][j],default=None) for j in (0,1)])
