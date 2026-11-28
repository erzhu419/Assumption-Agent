from probe import *
from scipy.optimize import least_squares
from itertools import product

def fit_vp(u,v,base,extra,starts):
 def err(h):
  a=np.c_[base,extra(u,h)]; return a@fit(a,v)-v
 sol=min((least_squares(lambda h:err(h).ravel(),h,bounds=(.12,12.),max_nfev=180) for h in starts),key=lambda z:z.cost)
 a=np.c_[base,extra(u,sol.x)]; return np.std(err(sol.x),axis=0),sol.x,fit(a,v)

if __name__=='__main__':
 for p,k in [('a','v'),('b','v'),('c','v1'),('c','v2')]:
  u,v=[s['panel_'+p][j] for j in ('u',k)]; good=np.linalg.norm(v,axis=1)<5; u,v=u[good],v[good]; print('\n',p,k)
  pairs=u*np.roll(u,-1,axis=1)
  for desc,b,extra in [('linear+sin/cos',pol(u,1),lambda u,h:np.c_[np.sin(u*h),np.cos(u*h)]),('quad+sin',pol(u,2),lambda u,h:np.sin(u*h)),('cyclic+sin/cos',np.c_[pol(u,1),pairs],lambda u,h:np.c_[np.sin(u*h),np.cos(u*h)]),('linear+sin',pol(u,1),lambda u,h:np.sin(u*h)),('linear+sin(PAIR)',pol(u,1),lambda u,h:np.sin((u*np.roll(u,-1,axis=1))*h)),('sin/cos',np.ones((len(u),1)),lambda u,h:np.c_[np.sin(u*h),np.cos(u*h)]),('tanh/sq',np.c_[np.ones(len(u)),u*u],lambda u,h:np.tanh(u*h)),('lin+tanh',pol(u,1),lambda u,h:np.tanh(u*h))]:
   er,h,c=fit_vp(u,v,b,extra,[[a]*4 for a in [1.,2.,3.,4.,6.]]); print(desc,er,h); 
   if max(er)<.009: print(c.T)
  if p=='a':
   for j in range(4):
    m=(j+1)%4; x=u[:,j]; w=u[:,m]; V=v[:,j]; A=np.c_[np.ones(len(u)),x,w,x*w]; c=fit(A,V); r=V-A@c; print('A',j,c, np.std(r)); print('A r corr',np.corrcoef(np.c_[r,u,u*u,u**3,np.sin(2*u),np.cos(2*u),pairs,pairs**2,pairs**3].T)[0,1:].reshape(-1,4));
