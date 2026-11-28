import pickle, numpy as np, importlib.util
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
from scipy.linalg import svd
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']
p='/tmp/hegel_continual_archive/conf_m101_b03_batch_03_solver.py'; m=importlib.util.module_from_spec(sp:=importlib.util.spec_from_file_location('old',p));sp.loader.exec_module(m)
def A(*a):return np.column_stack((np.ones(len(a[0])),*a))
def fit(a,y):return np.linalg.lstsq(a,y,rcond=1e-10)[0]
def products(x,d):return np.column_stack([x[:,j].prod(1) for j in combinations_with_replacement(range(x.shape[1]),d)])
z=m._calibrated_coordinates(s['panel_a'],tr['x']); b=s['panel_b']; u=b['u']; bc=fit(A(u),b['v']); b=A(z)@bc
u=s['panel_c']['u']; f0=s['panel_c']['v1'];g0=s['panel_c']['v2']; fd=lambda x:A(x,products(x,2)); fc=fit(fd(u),f0); f=fd(z)@fc
print('C1 coef\n',np.round(fc,4)); print('B coef\n',np.round(bc,4));print('z extent',z.min(0),z.max(0))
for name,fun,p0 in [('gauss',lambda x,k:np.exp(-k*x*x),.55),('cos',lambda x,k:np.cos(k*x),1.28),('rational',lambda x,k:1/(1+k*x*x),.2)]:
 for d in (1,4):
  def design(x,k):return A(x,fun(x,k))
  def res(k):
   a=design(u,k);return (a@fit(a,g0)-g0).ravel()
  opt=least_squares(res,[p0]*d,bounds=(.0001,5),gtol=1e-13)
  print('C2',name,opt.x,np.sqrt(np.mean(opt.fun**2))); 
  if name=='gauss' and d==4:
   gc=fit(design(u,opt.x),g0);print('C2 coef\n',np.round(gc,4));g=design(z,opt.x)@gc
T=tr['traj'];t=s['eval_times']; basis=lambda r:-np.expm1(-t[:,None]*r)
for starts in [[.4,2.],[.4,2.,5.],[.4,2.,5.,9.]]:
 def res(r):
  h=basis(r);return (h@fit(h,T[:52].T)-T[:52].T).ravel()
 o=least_squares(res,starts,bounds=(.01,20),gtol=1e-12);print('rates',o.x,'rmse',np.sqrt(np.mean(o.fun**2)))
 if len(starts)==2: h=basis(o.x)
y=np.c_[tr['y'],T]; n=52
features={'z':z,'b':b,'f':f,'g':g,'z+b':z+b,'z+f':z+f,'z+g':z+g,'f+g':f+g}
sets={}
for nm,w in features.items():
 sets[nm+'^2']=A(w,products(w,2));sets[nm+'^3']=A(w,products(w,2),products(w,3));
for nm,w in [('zf',np.c_[z,f]),('zg',np.c_[z,g]),('fg',np.c_[f,g]),('zfg',np.c_[z,f,g]),('bfg',np.c_[b,f,g])]:
 sets[nm+' full2']=A(w,products(w,2));sets[nm+' diag']=A(z,products(z,2),f,g,*[features[i]*features[j] for i,j in combinations_with_replacement(nm,2)])
for nm,a in sets.items():
 v=a[:,1:]-a[:n,1:].mean(0);v=v/np.maximum(v[:n].std(0),1e-8); U,S,V=svd(v[:n],full_matrices=False); result=[]
 for lam in (1e-8,1e-5,.0001,.001,.01,.1,1):
  yp=y[:n].mean(0)+v[n:]@(V.T*(S/(S*S+lam)))@U.T@y[:n]; e=np.abs(yp-y[n:]);result.append((lam,round(e[:,0].mean(),5),round(e[:,1:].mean(),5)))
 print(nm, min(result,key=lambda e:e[1]),min(result,key=lambda e:e[2]))
