from probe import *
from scipy.linalg import lstsq
p=S['panel_c']; u=p['u']; Y=np.c_[p['v1'],p['v2']]; D=design(u,1)
def assess(F):
 w=np.linalg.lstsq(F,Y,rcond=1e-10)[0]; r=Y-F@w; h=np.sum(np.linalg.svd(F,full_matrices=False)[0]**2,axis=1); return np.sqrt(np.mean(r*r,axis=0)),np.sqrt(np.mean((r/np.maximum(1-h,1e-10)[:,None])**2,axis=0))
ops={'sin':np.sin,'cos':np.cos,'tanh':np.tanh,'atan':np.arctan,'cubic':lambda x:x**3,'cube':lambda x:x**3,'xabs':lambda x:x*abs(x),'softsign':lambda x:x/(1+abs(x)),'square':np.square,'abs':np.abs}
for name,op in ops.items():
 for scale in ([.6,1,1.6,2.4,3.2,4.5,6] if name in ('sin','cos','tanh','atan','softsign') else [1]):
  print(name,scale,*[np.round(q,5) for q in assess(np.c_[D,op(u*scale)])])
for name,op in ops.items():
 if name=='cube':continue
 for dest in (slice(0,4),slice(4,8)):
  y=Y[:,dest]; square=dest.start==4
  def F(u,p):
   return np.c_[design(u,1),u*u if square else np.empty((len(u),0)),op(u*p[:4]+p[4:])]
  def fun(p):
   X=F(u,p); a=np.linalg.lstsq(X,y,rcond=1e-10)[0];return (X@a-y).ravel()
  sols=[least_squares(fun,np.r_[np.full(4,k),np.zeros(4)],bounds=(np.r_[np.full(4,.15),np.full(4,-4)],np.r_[np.full(4,8),np.full(4,4)]),max_nfev=200) for k in (.8,1.6,3,5)]
  best=min(sols,key=lambda p:p.cost); print('optimized',name,dest.start,np.round(np.std(best.fun),6),np.round(best.x,3))
