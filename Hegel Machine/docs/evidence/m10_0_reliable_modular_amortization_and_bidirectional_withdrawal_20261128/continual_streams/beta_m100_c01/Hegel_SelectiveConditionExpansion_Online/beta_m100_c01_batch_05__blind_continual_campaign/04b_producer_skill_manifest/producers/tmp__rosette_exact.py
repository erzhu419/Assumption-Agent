from rosette_models import X,Y,axis
import numpy as np
from scipy.special import erf
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
rng=np.random.default_rng(812); fold=np.empty(len(X),int)
for mask in (axis,~axis): fold[mask]=rng.permutation(mask.sum())%5
fs={'tanh':np.tanh,'erf':erf,'sin':np.sin,'atan':np.arctan,'rat':lambda x:x/(1+x*x),'soft':lambda x:x/np.sqrt(1+x*x),'exp':lambda x:x*np.exp(-x*x)}
def d(x,k,sp):
 u=x;v=x[:,[2,3,0,1]]
 cols=[np.ones_like(u),v,np.sin(k[-1]*v)**2];
 for i,m in enumerate(sp):cols.append(u if m=='lin' else fs[m](k[i]*u))
 return np.stack(cols,axis=-1)
def fit(x,y,sp):
 def cf(k):
  a=d(x,k,sp); return np.stack([np.linalg.lstsq(a[:,j],y[:,j],rcond=1e-10)[0] for j in range(4)])
 def err(k):return (np.einsum('njk,jk->nj',d(x,k,sp),cf(k))-y).ravel()
 opts=[least_squares(err,p,bounds=(.08,3.5),max_nfev=120,gtol=1e-11) for p in starts[len(sp)]]
 o=min(opts,key=lambda r:r.cost);return np.std(o.fun),o.x,cf(o.x)
starts={1:[[1,1],[.6,1]],2:[[1,1,1],[.5,1.5,1],[1.5,.5,1],[.7,.85,1]]}
if __name__=='__main__':
 results=[]
 for sp in [(f,) for f in fs]+list(combinations_with_replacement([*fs,'lin'],2)):
  rms,k,c=fit(X,Y,sp);results.append((rms,sp,k,c))
 for rms,sp,k,c in sorted(results,key=lambda r:r[0])[:10]:
  pred=np.empty_like(Y)
  for i in range(5):
   m=fold!=i; _,kk,cc=fit(X[m],Y[m],sp);pred[~m]=np.einsum('njk,jk->nj',d(X[~m],kk,sp),cc)
  e=pred-Y;print(sp,'train',rms,'OOF max/rms',np.abs(e).max(),np.sqrt(np.mean(e**2,axis=0)), 'axis/mixed',*[np.sqrt(np.mean(e[m]**2)) for m in (axis,~axis)]);print('scales',k,'coefficients',c.round(7))
 for j in range(4):
  u=X[:,j];v=X[:,j^2]; a=np.column_stack([np.ones(len(u)), *[u**i for i in range(1,12,2)],v,np.sin(v)**2]); c=np.linalg.lstsq(a,Y[:,j],rcond=None)[0];print('odd+sin2',j,c,'rms',np.std(a@c-Y[:,j]))
