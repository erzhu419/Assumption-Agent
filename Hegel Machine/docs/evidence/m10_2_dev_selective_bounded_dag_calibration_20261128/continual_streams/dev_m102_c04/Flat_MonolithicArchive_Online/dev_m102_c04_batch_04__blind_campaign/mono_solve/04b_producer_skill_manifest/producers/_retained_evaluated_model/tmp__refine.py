exec(open('/tmp/graph.py').read().split('rows=[]')[0])
from itertools import product
funcs={'rational':lambda x,k:x*x/(1+k*x*x),'log':lambda x,k:np.log1p(k*x*x)/k,'sqrt':lambda x,k:2*np.expm1(.5*np.log1p(k*x*x))/k,'exp':lambda x,k:-np.expm1(-k*x*x)/k,'tanh':lambda x,k:np.tanh(k*x*x)/k,'atan':lambda x,k:np.arctan(k*x*x)/k}
fits=[]
for name,fn in funcs.items():
 def design(x,k,c):return aff(x+c*fn(x,k))
 def residual(p):
  F=design(U,*p);return (F@ls(F,W,rcond=None)[0]-W).ravel()
 sol=least_squares(residual,[.22,.55],bounds=([.001,.01],[3,2]),gtol=1e-13,ftol=1e-13,xtol=1e-13)
 F=design(U,*sol.x); w=ls(F,W,rcond=None)[0]; R=design(q,*sol.x)@w; fits.append((np.std(sol.fun),name,sol.x,R));print(name,sol.x,np.std(sol.fun))
P=F1(q)
for _,name,_,R in sorted(fits,key=lambda v:v[0])[:3]:
 D=(P*R).sum(1); C=P[:,0]*R[:,1]; G=aff(np.c_[P,R,D]); E=np.c_[G,C]; fast=G; slow=aff(np.c_[R,C]);
 for k,(F,S) in enumerate([(G,slow),(E,slow),(G,E),(E,E)]):
  def design(r,idx):
   H=-np.expm1(-t[:,None]*r);return np.concatenate([(F[idx,None,:]*H[None,:,0,None]),(S[idx,None,:]*H[None,:,1,None])],axis=2).reshape(-1,F.shape[1]+S.shape[1]);
  i=np.arange(52); j=np.arange(52,72)
  def err(r):
   A=design(r,i);return A@ls(A,T[i].ravel(),rcond=None)[0]-T[i].ravel()
  opt=least_squares(err,[1.795,.42],bounds=([1.3,.1],[2.3,.8]),gtol=1e-13); A=design(opt.x,i); w=ls(A,T[i].ravel(),rcond=None)[0]; pe=abs(design(opt.x,j)@w-T[j].ravel()).reshape(-1,8); ye=abs(G[j]@ls(G[i],y[i],rcond=None)[0]-y[j]); print(name,k,'rates',opt.x,'T train SD',np.std(opt.fun),'T test MAE',pe.mean(),'time max MAE',pe.mean(0).max(),'y test MAE',ye.mean(),'w',np.round(w,5))
