import pickle,numpy as np,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True)
def a(x):return np.c_[np.ones(len(x)),x]
def ls(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
for k in ('panel_b','panel_c'):
 p=s[k];u=p['u'];D=a(u);V=p.get('v',p.get('v1'));print(k,'U,V');print(np.c_[u,V]);
 for g in [1,2.15,3,4,5,6.28,8,10]:
  si=np.sin(g*u);co=np.cos(g*u)
  print('g',g,[(round(np.mean(abs(M@ls(M,V)-V)),5),round(np.mean(abs((M@ls(M,V)-V)/(1-np.sum(np.linalg.qr(M,mode='reduced')[0]**2,axis=1))[:,None])),5)) for M in [a(si),a(co),a(np.c_[si,co]),a(np.c_[u,si,co]),a(np.c_[u, np.tanh(g*u)])]])
 rng=np.random.default_rng(10)
 for name,func in [('sin',np.sin),('tanh',np.tanh),('asin',np.arcsinh),('atan',np.arctan)]:
  best=[]
  for j in range(4):
   y=V[:,j];r0=ls(D,y);bs=None
   for i in range(24 if name=='sin' else 8):
    init=np.r_[0.,1.,rng.normal(size=5)*2] if i else np.r_[0.,1.,r0*2]
    fit=least_squares(lambda q:q[0]+q[1]*func(D@q[2:])-y,init,max_nfev=450)
    if bs is None or fit.cost<bs.cost:bs=fit
   best.append((round(np.sqrt(2*bs.cost/len(D)),6),bs.x.round(3).tolist()))
  print(name,best)
