import sys,pickle,numpy as np,itertools
from scipy.optimize import minimize_scalar,least_squares
sys.path.insert(0,'/holdout_task');import node_00 as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['panel_c']['u']; y=s['panel_c']['v2']; pairs=list(itertools.combinations(range(4),2)); ring=[(0,1),(1,2),(2,3),(0,3)]; a=np.c_[np.ones(len(x)),x]; b=lambda xs,ps: np.column_stack([xs[:,i]*xs[:,j] for i,j in ps]); p=m._ls(np.c_[np.ones(len(x)),b(x,ring)],s['panel_c']['v1']); print('V1 ring RMSE',np.std(np.c_[np.ones(len(x)),b(x,ring)]@p-s['panel_c']['v1'])); print('B affine rmse',np.std(np.c_[np.ones(24),s['panel_b']['u']]@m._ls(np.c_[np.ones(24),s['panel_b']['u']],s['panel_b']['v'])-s['panel_b']['v']));
for name,base in [('constant',a[:,:1]),('affine',a),('ring',np.c_[a[:,:1],b(x,ring)]),('affring',np.c_[a,b(x,ring)]),('cross',np.c_[a,b(x,pairs)]),('quad',np.c_[a,b(x,pairs),x*x])]:
 for f,fn in [('rat',lambda z,k:z*z/(1+k*z*z)),('sqrt',lambda z,k:np.sqrt(1+k*z*z)-1),('log',lambda z,k:np.log1p(k*z*z)),('cos',lambda z,k:1-np.cos(k*z)),('abs',lambda z,k:z*z/(1+k*np.abs(z)))]:
  def res(k,details=False):
   z=np.c_[base,fn(x,k)]; w=m._ls(z,y); e=z@w-y
   return (w,e) if details else e.ravel()
  opt=least_squares(res,[.25],bounds=(.001,5),max_nfev=200,gtol=1e-13,ftol=1e-13,xtol=1e-13); opt2=least_squares(res,np.repeat(opt.x,4),bounds=(.001,5),max_nfev=200,gtol=1e-13,ftol=1e-13,xtol=1e-13); w,e=res(opt2.x,True);print(name,f,'shared',np.std(opt.fun),opt.x,'sep',np.std(e),opt2.x); 
  if f=='rat' and name in ['quad','cross','affring']:print(np.round(w,4))
