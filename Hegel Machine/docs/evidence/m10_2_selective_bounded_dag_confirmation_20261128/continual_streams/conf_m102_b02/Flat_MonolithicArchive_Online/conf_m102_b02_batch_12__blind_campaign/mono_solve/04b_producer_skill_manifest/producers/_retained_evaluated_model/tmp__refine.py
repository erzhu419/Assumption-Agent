from stress import *
from scipy.optimize import least_squares
u,v=s['panel_b']['u'],s['panel_b']['v']; print('B ranges',u.min(0),u.max(0)); print('B rows',np.round(np.c_[u,v],3))
F={'softsign':lambda x,k:x/(1+k*abs(x)),'softroot':lambda x,k:x/np.sqrt(1+k*x*x),'atan':lambda x,k:np.arctan(k*x),'logsoft':lambda x,k:np.sign(x)*np.log1p(k*abs(x)),'exptaper':lambda x,k:x*np.exp(-k*x*x),'rat':lambda x,k:x/(1+k*x*x),'sine':lambda x,k:np.sin(k*x),'asinh':lambda x,k:np.arcsinh(k*x),'tanh':lambda x,k:np.tanh(k*x),'expsat':lambda x,k:np.sign(x)*(-np.expm1(-k*abs(x))),'tanhabs':lambda x,k:x*np.tanh(k*abs(x)),'xabs':lambda x,k:x*abs(x),'cubic':lambda x,k:x**3,'atanabs':lambda x,k:x*np.arctan(k*abs(x))}
records=[]
for name,g in F.items():
 for mode in ('free','direct','tied'):
  def basis(x,p):
   a=g(x,p[0]);return np.c_[x,a] if mode=='free' else (a if mode=='direct' else x+p[1]*a)
  def design(x,p):return np.c_[np.ones(len(x)),basis(x,p)]
  def res(p):
   a=design(u,p);return (a@m._ls(a,v)-v).ravel()
  cfg=([.01,-20],[20,20]) if mode=='tied' else ([.01],[20]); params=min((least_squares(res,[k,l] if mode=='tied' else [k],bounds=cfg,max_nfev=120) for k in (.25,1.25,3,6) for l in ((-.4,.4) if mode=='tied' else (0,))),key=lambda f:f.cost).x
  a=design(u,params); U,_,_=np.linalg.svd(a,full_matrices=False); h=(U[:,:np.linalg.matrix_rank(a)]**2).sum(1); records.append((np.mean((res(params).reshape(v.shape)/(1-h[:,None]))**2)**.5,name,mode,params))
print('B sorted', sorted(records,key=lambda r:r[0])[:16]); z,h=m._coords(s,xx); print('B nested range',h.min(0),h.max(0)); D=z*h; E=z[:,1]*h[:,0]; Xs=[np.c_[z,h,D,E],np.c_[z,h,D.sum(1),E],np.c_[z,h,D.sum(1)+E]]
for name,i,j in slices:
 vals=[]
 for X in Xs:
  get=lambda ids:np.c_[np.ones(len(ids)),X[ids]]
  pp=get(j)@m._ls(get(i),y[i]); e=np.abs(pp-y[j]); vals.append(np.round([e[:,0].mean(),e[:,1:].mean(),e[:,1:].mean(0).max()],6))
 print(name,vals)
