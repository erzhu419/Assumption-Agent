import sys,pickle,numpy as np
from scipy.optimize import least_squares
from scipy.special import erf
sys.path.insert(0,'/holdout_task'); import batch_solver as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=m._calibrate(s['panel_a'],tr['x']); t=s['eval_times']; T=np.c_[-np.expm1(-1.813*t),t/(1+.405*t)]; Y=np.c_[tr['y'],tr['traj']]; Z=np.c_[tr['y'],tr['traj']@np.linalg.pinv(T.T)]
def panel(p,key,g,starts):
 u,v=s[p]['u'],s[p][key]
 def calc(q):
  a=np.c_[np.ones(len(u)),g(u,*q)]; w=np.linalg.lstsq(a,v,rcond=None)[0]; return a@w-v,w,a
 o=min((least_squares(lambda q:calc(q)[0].ravel(), q,max_nfev=300) for q in starts),key=lambda o:o.cost); r,w,a=calc(o.x); rm=np.sqrt(np.mean((r/(1-(a*np.linalg.pinv(a).T).sum(1))[:,None])**2));return (rm,o.x,lambda xx:np.c_[np.ones(len(xx)),g(xx,*o.x)]@w)
cs=panel('panel_c','v1',lambda z,k,a,b:z*abs(z)/(1+np.exp(k)*abs(z))+a*z**3+b*z,[[-.42,-.02,-1.08]]); print('C',cs[:2]); c=cs[2]
bf={
 'erf':(lambda z,k,a:erf(np.exp(k)*z)+a*z,[[.38,-1.54]]),
 'ratquad':(lambda z,k,a,b:z**3/(1+np.exp(k)*z**4)+a*z**3+b*z,[[-.1,.57,-.16]]),
 'tancubed':(lambda z,k,a,b:np.tanh(np.exp(k)*z**3)+a*z**3+b*z,[[.49,1.3,-.29]]),
 'soft':(lambda z,k,a,b:z*abs(z)/(1+np.exp(k)*abs(z))+a*z**3+b*z,[[0,0,-1],[-1,.2,-.2]]),
 'ratmix':(lambda z,k,l,a,b:z**3/(1+np.exp(k)*z*z+np.exp(l)*z**4)+a*z**3+b*z,[[0,0,.5,-.16],[-4,0,.5,-.16]]),
 'tanhmix':(lambda z,k,a,b,c:np.tanh(np.exp(k)*z+a*z**3)+b*z**3+c*z,[[0,0,0,-1],[0,1,0,-1],[-2,1,.4,-.3],[.4,-1,2,-1]]),
 'erfmix':(lambda z,k,a,b,c:erf(np.exp(k)*z+a*z**3)+b*z**3+c*z,[[0,0,0,-1],[0,1,0,-1],[-2,1,.4,-.3],[.4,-1,2,-1]]),
 'sinmix':(lambda z,k,a,b,c:np.sin(np.exp(k)*z+a*z**3)+b*z**3+c*z,[[0,0,0,-1],[0,1,0,-1],[-2,1,.4,-.3],[.4,-1,2,-1]]),
 'poly':(lambda z,*a:z+sum(k*z**(2*j+3) for j,k in enumerate(a)),[[-1,1,0,0,0],[-2,0,0,0,0]]),
 'tanhabs':(lambda z,k,a,b:z*np.tanh(np.exp(k)*abs(z))+a*z**3+b*z,[[0,0,-1]])}
rng=np.random.default_rng(417); splits=[np.arange(72)]+[rng.permutation(72) for j in range(24)]+[np.argsort(x[:,j]*sgn) for j in range(4) for sgn in [-1,1]]
def scores(b):
 for jc,ch in enumerate([c(b),c(b),c(b)]):
  E=[]
  for p in splits:
   i,j=p[:52],p[52:]; X=x/x[i].std(0); C=ch/ch[i].std(0); A=m._design(X,C)
   if jc==1:A=np.c_[A[:,:9],A[:,17:]]
   if jc==2:A=np.c_[A[:,:13],A[:,17:]]
   a=A[i]; w=np.linalg.solve(a.T@a+.003*np.eye(a.shape[1]),a.T@Z[i]); d=A[j]@w; e=np.abs(np.c_[d[:,0],d[:,1:]@T.T]-Y[j]); E.append([e[:,0].mean(),e[:,1:].mean(),e[:,-1].mean()])
  e=np.array(E); print(jc,e[0].round(5),e[1:25].max(0).round(5),e[25:].max(0).round(5),end=' ')
 print()
for k,(g,starts) in bf.items():
 r,q,b=panel('panel_b','v',g,starts); print(k,r,q,end=' ');scores(b(x))
