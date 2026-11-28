import numpy as np,pickle,itertools
from scipy.optimize import least_squares
from scipy.special import erf
c=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']['calibration_load_cells']; x,y=c['true_channel_loads'],c['bridge_raw']; ok=np.max(abs(y),axis=1)<5; x,y=x[ok],y[ok]
F={'sin':np.sin,'atan':np.arctan,'tanh':np.tanh,'erf':erf,'rat2':lambda z:z/(1+z*z),'rat4':lambda z:z/(1+z**4),'sqrt':lambda z:z/np.sqrt(1+z*z),'soft4':lambda z:z/(1+z**4)**.25}
def design(x,j,key,k,nd,nc,mix):
 z=x[:,j]; cross=np.delete(x,j,axis=1); cols=[np.ones(len(z))]+[z**m for m in range(1,nd+1)]+[cross**m for m in range(1,nc+1)]
 if mix:cols += [x[:,a]*x[:,b] for a in range(4) for b in range(a+1,4)]
 cols.extend(F[f](v*z) for f,v in zip(key,k)); return np.column_stack(cols)
def fit(x,y,j,key,nd,nc,mix=False):
 def vp(k,ret=False):
  a=design(x,j,key,k,nd,nc,mix); p=np.linalg.lstsq(a,y,rcond=1e-10)[0]; return p if ret else a@p-y
 if key:
  ans=min((least_squares(vp,s,bounds=(.12,4),max_nfev=100) for s in itertools.product(*([(.5,1.3,2.55)]*len(key)))),key=lambda r:r.cost); k=ans.x
 else:k=np.array([])
 return k,vp(k,True)
if __name__=='__main__':
 rec=[]
 for key in [()]+[(f,) for f in F]+[('tanh','sin'),('tanh','rat2'),('sin','rat2'),('sin','atan')]:
  for nd,nc,mix in ([(d,3,False) for d in [3,5,7,9]] if not key else list(itertools.product([1,3],[2,3],[False,True]))):
   cfg=(key,nd,nc,mix); errs=[]; loo=[]; params=[]
   for j in range(4):
    k,p=fit(x,y[:,j],j,*cfg); a=design(x,j,key,k,nd,nc,mix); r=y[:,j]-a@p; t=np.column_stack([a]+[(design(x,j,key,k+e*1e-5,nd,nc,mix)@p-a@p)/1e-5 for e in np.eye(len(k))]); u,s,_=np.linalg.svd(t,full_matrices=False); hh=(u[:,s>1e-8]**2).sum(1); errs.append(r); loo.append(r/(1-hh)); params.append((k,p))
   rec.append((max(np.mean(np.array(loo)**2,axis=1)**.5),np.max(abs(np.array(loo))),cfg,np.std(errs,axis=1),params))
 rec.sort(key=lambda a:a[0]);
 for row in rec[:18]:print(row[:4]);
 for row in rec[:3]:
  print('PARAMS',row[2]);
  for k,p in row[-1]: print(np.round(k,5),np.round(p,5))
