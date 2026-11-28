import numpy as np,pickle
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
with open('/holdout_task/public_batch.pkl','rb') as f: d=pickle.load(f)[0]['dataset']
c=d['calibration_load_cells']; x=c['true_channel_loads']; y=c['bridge_raw']; clean=np.max(abs(y),axis=1)<4; x,y=x[clean],y[clean]
ix,jx=np.array(list(combinations_with_replacement(range(4),2))).T
func={'tanh':np.tanh,'atan':np.arctan,'sin':np.sin,'rat':lambda z:z/(1+z*z),'root':lambda z:z/np.sqrt(1+z*z),'erf':__import__('scipy').special.erf,'asinh':np.arcsinh}
def basis(x,s,f,degree,linear=True):
 blocks=[np.ones((len(x),1)),func[f](x*s)]
 if linear: blocks.append(x)
 if degree>=2: blocks.append(x[:,ix]*x[:,jx])
 if degree>=3: blocks.append(np.array([x[:,i]*x[:,j]*x[:,k] for i,j,k in combinations_with_replacement(range(4),3)]).T)
 return np.column_stack(blocks)
def fit(x,y,f,deg,lin=True):
 def prof(s,ret=False):
  a=basis(x,s,f,deg,lin); b=np.linalg.lstsq(a,y,rcond=None)[0]; return b if ret else (a@b-y).ravel()
 o=least_squares(prof,[1.]*4,bounds=(.15,4.),max_nfev=60); return o.x,prof(o.x,True)
for deg in (1,2,3):
 for lin in (False,True):
  for f in func:
   s,b=fit(x,y,f,deg,lin); r=basis(x,s,f,deg,lin)@b-y
   if np.sqrt(np.mean(r*r))>.025: continue
   errs=[]
   for ii in np.array_split(np.random.default_rng(23).permutation(len(x)),8):
    m=np.ones(len(x),bool);m[ii]=0;ss,bb=fit(x[m],y[m],f,deg,lin); errs.extend(zip(ii,basis(x[ii],ss,f,deg,lin)@bb-y[ii]))
   errs=np.array([e for _,e in sorted(errs,key=lambda a:a[0])]); print(f,deg,lin,'scales',np.round(s,4),'fit',round(np.sqrt(np.mean(r*r)),7),'oof RMSE',np.round(np.sqrt(np.mean(errs*errs,0)),6),'max',round(np.max(abs(errs)),6))
   if np.sqrt(np.mean(r*r))<.006:
    print(np.round(b,4))
