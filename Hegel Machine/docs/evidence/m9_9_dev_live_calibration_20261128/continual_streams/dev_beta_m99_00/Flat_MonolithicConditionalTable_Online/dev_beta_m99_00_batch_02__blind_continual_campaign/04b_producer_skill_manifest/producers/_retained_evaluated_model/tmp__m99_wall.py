import importlib.util,pickle,numpy as np
from scipy.sparse import csc_matrix
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('prior','/tmp/hegel_continual_archive/dev_beta_m99_00_batch_01_solver.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
it=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0];s=it['study'];p=it['campaign_diagnostics'];R=np.array([v['y_pulse_raw'] for v in p]);W=np.array([v['w_pulse'] for v in p]).T
Y=m._calibrate(dict(s,y_obs_raw=R.ravel())).reshape(R.shape)
Z=s['Z_nuisance'];q=np.linalg.qr(Z)[0];base=m._axis
nobs=Y.shape[1]
print('shifts',s['mu_shifts'])
for k in 'xy': print(k,'endpoint vel',s['vel_'+k][[0,-1]],'kap',s['kappa_'+k][[0,-1]],'rob',s['robin_'+k])
def pred(a):
 seq=iter(zip('xy',a.reshape(2,2,3)))
 def axis(k,v,h):
  key,ends=next(seq); d=base(k,v,h).toarray();n=len(k)
  for i,j,o,(scale,leak,wrap) in zip((0,n-1),(1,n-2),(n-1,0),ends):
   D=scale*k[i]/h**2; C=wrap*k[i]/h**2; B=leak*s['robin_'+key]/h
   d[i,i]+=D+C+B;d[i,j]-=D;d[i,o]-=C
  return csc_matrix(d)
 m._axis=axis
 return (m._forward(s)[0]@W).T
def opt(cols,train):
 mask=np.zeros(12,dtype=bool);mask[cols]=1
 def fill(x):
  a=np.zeros(12);a[mask]=x;return a
 def errors(a):
  e=pred(a)-Y;return e-e@q@q.T
 def fun(x):return errors(fill(x))[train].ravel()
 if len(cols):
  lo=np.array([-.99,-3,0]*4)[mask];hi=np.array([12,5,12]*4)[mask];r=least_squares(fun,np.maximum(1e-7,lo+1e-7),bounds=(lo,hi),ftol=1e-12,gtol=1e-12,xtol=1e-12,max_nfev=200);a=fill(r.x)
 else:a=fill([])
 e=errors(a);return a,np.sqrt(np.mean(e*e,axis=1)),np.sqrt(np.mean((pred(a)-Y)**2,axis=1))
for name,cols in [('sealed',[]),('leak',[1,4,7,10]),('diff',[0,3,6,9]),('wrap',[2,5,8,11]),('diff+leak',[0,1,3,4,6,7,9,10]),('all',list(range(12)))]:
 a,r,raw=opt(cols,[0,1,2]);cv=[opt(cols,[j for j in range(3) if j!=i])[1][i] for i in range(3)] if cols else r
 print(name,'params',np.round(a.reshape(4,3),7).tolist(),'pulse',r,'raw',raw,'LOPO',cv,flush=True)
