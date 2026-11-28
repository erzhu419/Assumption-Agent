import pickle, numpy as np
from scipy.optimize import least_squares
from itertools import combinations
x0,y0=(lambda c:(c['true_channel_loads'],c['bridge_raw']))(pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']['calibration_load_cells'])
keep=np.max(abs(y0-x0),axis=1)<2; x,y=x0[keep],y0[keep]; pair=list(combinations(range(4),2)); js=np.arange(4); np.set_printoptions(precision=6,suppress=True,linewidth=145)
F={'tanh':np.tanh,'atan':np.arctan,'soft':lambda v:v/np.sqrt(1+v*v),'rational':lambda v:v/(1+v*v),'sin':np.sin,'erf':__import__('scipy.special',fromlist=['erf']).erf}
rng=np.random.default_rng(405); folds=np.array_split(rng.permutation(len(x)),5)
def features(x,j,p,kind,cross,ix):
 k,l=p; other=js!=j; z=x[:,j]; u=x[:,other]; h=F[kind](k*z)/k
 cols=[np.ones(len(x)),h,z] if ix!=3 else [np.ones(len(x)),h]; cols+=[u]
 if cross=='poly': cols += [u**2,u**3,u**4]
 if cross=='sin': cols += [np.sin(l*u),np.cos(l*u)-1]
 if cross=='tanh': cols += [np.tanh(l*u),u**2]
 if cross=='rat': cols += [u/(1+(l*u)**2),u**2/(1+(l*u)**2)]
 if ix in (1,2): cols += [np.column_stack([x[:,a]*x[:,b] for a,b in pair])]
 if ix==2: cols += [np.column_stack([x[:,a]**2*x[:,b] for a in js for b in js if a!=b])]
 return np.column_stack(cols)
def fit(t,v,j,kind,cross,ix):
 def pred(p,vv,ret=False):
  a=features(x[t],j,p,kind,cross,ix); b=np.linalg.lstsq(a,y[t,j],rcond=1e-10)[0]; return (features(x[vv],j,p,kind,cross,ix)@b,b) if ret else a@b-y[t,j]
 p=least_squares(pred,[1.,1.],args=(t,),bounds=([.08,.08],[4.,4.]),ftol=1e-9,gtol=1e-10,max_nfev=55).x
 return (*pred(p,v,True),p)
n=np.arange(len(x)); allr=[]
for kind in F:
 for cross in ('poly','sin','tanh','rat'):
  for ix in (0,1,2,3):
   out=np.zeros_like(y)
   for v in folds:
    t=np.setdiff1d(n,v)
    for j in js: out[v,j]=fit(t,v,j,kind,cross,ix)[0]-y[v,j]
   allr.append((np.mean(out**2)**.5,np.max(abs(out)),kind,cross,ix))
for entry in sorted(allr)[:16]: print(entry)
for _,_,kind,cross,ix in sorted(allr)[:3]:
 print('best',kind,cross,ix)
 for j in js:
  yy,b,p=fit(n,n,j,kind,cross,ix); print('ch',j,'p',p,'beta',b,'RMSE',np.mean((yy-y[:,j])**2)**.5,'max',max(abs(yy-y[:,j])),'axis err',(yy-y[:,j])[:13])
