import numpy as np,pickle,runpy,itertools
from scipy.linalg import solve
from scipy.optimize import minimize_scalar
from sklearn.linear_model import ARDRegression, OrthogonalMatchingPursuit
m=runpy.run_path('/holdout_task/node_00.py'); s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; y=s['train']['y']; Y=s['train']['traj']; z=m['_sensor'](s['panel_a'],s['train']['x']); n=len(z)
def lin(A,b):return np.linalg.lstsq(A,b,rcond=1e-8)[0]
def base(u):return np.c_[np.ones(len(u)),u]
maps={}; bases={}
for p,k,fun,rg in [('panel_b','v',np.sin,(2,3.5)),('panel_b','v',np.tanh,(.7,2)),('panel_b','v',lambda a:a/(1+a*a),(.4,1.4)),('panel_c','v1',np.arctan,(1.5,3)),('panel_c','v2',np.cos,(.9,1.6))]:
 u,v=s[p]['u'],s[p][k]; name=k+fun.__name__; 
 for mode in range(4):
  def funs(u,f):
   q=fun(f*u)
   if mode==0:return np.c_[np.ones(len(u)),q]
   if mode==1:return np.c_[base(u),q]
   if mode==2:return np.c_[base(u),q,u*u]
   return np.c_[base(u),q,np.cos(f*u)] if k!='v2' else np.c_[base(u),q,np.sin(f*u)]
  def obj(f):
   A=funs(u,f); return np.mean((A@lin(A,v)-v)**2)
  opt=minimize_scalar(obj,bounds=rg,method='bounded'); A=funs(u,opt.x); B=lin(A,v); F=funs(z,opt.x)@B; print('panel',name,mode,round(opt.x,5),round(opt.fun**.5,6),flush=True); maps[name+str(mode)]=F
  if mode==1: print('coeff',np.round(B,4)); bases[name]=fun(opt.x*z)
G,gn=m['_groups'](s,z); tr=np.arange(52); te=np.arange(52,n)
V=np.linalg.svd(Y,full_matrices=False)[0][:,:3]; print('correlations y state',np.corrcoef(np.c_[y,V].T)[0]);
def make(fs,degree=2,cross=True):
  aa=[]; nn=[]
  for k,g in fs.items():
   aa.append(g); nn.extend(f'{k}{i}' for i in range(g.shape[1]));
  H=np.column_stack(aa); out=[H]; names=list(nn)
  if cross:
   for d in range(2,degree+1):
    ix=np.array(list(itertools.combinations_with_replacement(range(H.shape[1]),d))); out.append(np.prod(H[:,ix],axis=2)); names.extend('*'.join(nn[i] for i in I) for I in ix)
  return np.column_stack(out),names
sets=[('z',{'z':z}),('bases',{'z':z,**bases})]+[(k,{k:v}) for k,v in maps.items()]+[(k+'z',{'z':z,k:v}) for k,v in maps.items()]+[(k+l,{k:maps[k],l:maps[l]}) for k,l in itertools.combinations(['vsin1','vsin2','v1arctan1','v2cos1'],2)]; results=[]
for tag,fs in sets:
 for deg in (1,2,3):
  if len(fs)>3 and deg==3:continue
  F,nn=make(fs,deg); mu=F[tr].mean(0); sd=np.maximum(F[tr].std(0),1e-7); W=(F-mu)/sd
  # compare broad feature regularization to individual-term selection
  if W.shape[1]<300:
   A=ARDRegression(max_iter=180,threshold_lambda=1000) if 'max_iter' in ARDRegression().get_params() else ARDRegression(n_iter=180,threshold_lambda=1000)
   A.fit(W[tr],y[tr]); yp=A.predict(W); nz=np.abs(A.coef_)>.025; results.append((np.mean(abs(yp[te]-y[te])),np.mean(abs(yp[tr]-y[tr])),tag,deg,'ard',[(nn[i],round(A.coef_[i],3)) for i in np.flatnonzero(nz)]))
  for k in (5,8,12,18):
   if k>=W.shape[1]:continue
   A=OrthogonalMatchingPursuit(n_nonzero_coefs=k).fit(W[tr],y[tr]); yp=A.predict(W); results.append((np.mean(abs(yp[te]-y[te])),np.mean(abs(yp[tr]-y[tr])),tag,deg,f'omp{k}',[(nn[i],round(A.coef_[i],3)) for i in np.flatnonzero(A.coef_)]))
for line in sorted(results)[:32]:print(line)
