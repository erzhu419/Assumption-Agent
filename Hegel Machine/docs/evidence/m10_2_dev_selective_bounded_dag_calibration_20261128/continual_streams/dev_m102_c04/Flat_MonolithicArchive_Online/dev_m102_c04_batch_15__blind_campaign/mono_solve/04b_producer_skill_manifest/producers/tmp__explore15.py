import numpy as np,pickle,itertools as it
from scipy.optimize import least_squares,minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; ls=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-9)[0]; aff=lambda *x: np.column_stack((np.ones(len(x[0])),*x)); rm=lambda a:np.mean(a*a)**.5
u,v=s['panel_a'].values(); good=np.max(abs(u-v),axis=1)<3;u,v=u[good],v[good]; print('clean',len(u));eye=np.broadcast_to(np.eye(4),(len(u),4,4));
def cal_d(u,j,kind,k):
 r=np.roll(u,j,axis=1)
 if kind=='poly': return u*r,u**3
 if kind=='trig': return np.tanh(k[0]*u),np.sin(k[1]*r)
 if kind=='trig2': return np.sin(k[0]*u),np.tanh(k[1]*r)
 if kind=='trig3': return np.tanh(k[0]*u),np.cos(k[1]*r)
for j in (1,2,3):
 for kind in ('poly','trig','trig2','trig3'):
  for free in (False,True):
   def fit(k):
    h,g=cal_d(u,j,kind,k);d=np.concatenate((eye,eye*u[:,:,None],*((eye*h[:,:,None],eye*g[:,:,None]) if free else (h[:,:,None],g[:,:,None]))),axis=2).reshape(len(u)*4,-1);w=ls(d,v.ravel());return d@w-v.ravel(),w
   k=least_squares(lambda k:fit(k)[0],[1.8,2.4],bounds=(.15,6)).x if kind!='poly' else []
   e,w=fit(k);print('a',j,kind,free,rm(e),np.round(k,4),np.round(w,4) if rm(e)<.002 else '')
U,V,W=s['panel_c'].values(); pairs=lambda x:np.column_stack([x[:,i]*x[:,j] for i,j in it.combinations(range(4),2)]); funcs={'sat':lambda x,k:x/(1+k*abs(x)), 'rat':lambda x,k:x/(1+k*x*x),'root':lambda x,k:x/np.sqrt(1+k*x*x),'tanh':lambda x,k:np.tanh(k*x),'sin':lambda x,k:np.sin(k*x),'cos':lambda x,k:np.cos(k*x),'log':lambda x,k:np.log1p(k*x*x),'exp':lambda x,k:np.exp(-k*x*x),'abs':lambda x,k:abs(x),'sq':lambda x,k:x*x}
for v in (V,W):
 out=[]
 for name,f in funcs.items():
  for typ in range(5):
   def d(k):
    h=f(U,k); return aff(h,*[(),(U,),(pairs(U),),(U,pairs(U)),(pairs(h),)][typ]);
   def obj(k):
    a=d(k);return rm(a@ls(a,v)-v)
   o=minimize_scalar(obj,bounds=(.01,5),method='bounded'); out.append((o.fun,name,typ,o.x))
 print('c best',sorted(out)[:12]);
 for mode in (U,pairs(U),aff(U,pairs(U)),aff(U,U**2,pairs(U))): print('base',mode.shape,rm(mode@ls(mode,v)-v))
