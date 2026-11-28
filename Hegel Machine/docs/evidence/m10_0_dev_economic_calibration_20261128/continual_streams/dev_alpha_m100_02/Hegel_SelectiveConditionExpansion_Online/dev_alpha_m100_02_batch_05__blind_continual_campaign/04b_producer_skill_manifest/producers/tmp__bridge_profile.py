import pickle
import numpy as np
from scipy.optimize import least_squares
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=D['calibration_load_cells']; x=c['true_channel_loads']; y=c['bridge_raw']; sweep=c['single_channel_sweeps']
axis=(abs(x)>1e-9).sum(1)<=1
print('axis versus mixed rows',axis.sum(),(~axis).sum()); print('first load/raw rows',np.c_[x[:8],y[:8]].round(4)); print('sweeps by lab',np.c_[sweep['true_stress'],sweep['cmd_stress']].reshape(4,-1,2).round(4))

def link(z,b,k):
 if k=='cubic':return z-b*z**3
 if k=='tanh':return np.tanh(b*z)/b
 if k=='atan':return np.arctan(b*z)/b
 if k=='root':return z/np.sqrt(1+b*z*z)
 if k=='rational':return z/(1+b*z*z)
 if k=='sinh_inverse':return np.arcsinh(b*z)/b
 if k=='erf':
  from scipy.special import erf
  return erf(b*z)*np.sqrt(np.pi)/(2*b)

def features(x,j,b,k,terms):
 z=x[:,j]; other=[i for i in range(4) if i!=j]; a=[np.ones(len(x)),link(z,b,k),*x[:,other].T]
 if terms=='adj': a.extend(x[:,other].T**2)
 if terms=='diag2':a.extend(x.T**2)
 if terms=='full2':a.extend(x[:,i]*x[:,l] for i in range(4) for l in range(i,4))
 return np.column_stack(a)

def fit(x,y,k,terms):
 model=[]
 for j in range(4):
  A=features(x,j,.55,k,terms); p=np.r_[.55,np.linalg.lstsq(A,y[:,j],rcond=None)[0]]
  def residual(p):return features(x,j,p[0],k,terms)@p[1:]-y[:,j]
  for scale,loss in [(.12,'soft_l1'),(.025,'cauchy'),(.008,'cauchy')]:
   p=least_squares(residual,p,bounds=(np.r_[.0001,np.full(len(p)-1,-np.inf)],np.r_[4.,np.full(len(p)-1,np.inf)]),loss=loss,f_scale=scale,max_nfev=100).x
  model.append(p)
 return model

def predict(x,model,k,terms):return np.column_stack([features(x,j,p[0],k,terms)@p[1:] for j,p in enumerate(model)])

def score(r,folds):return np.round([np.median(r),np.quantile(r,.85),max(np.median(r[f]) for f in folds)],5)
if __name__=='__main__':
 rng=np.random.default_rng(870); random=np.array_split(rng.permutation(len(x)),5); radial=np.array_split(np.argsort(np.max(np.abs(x),axis=1)),5)
 for k in ('cubic','tanh','atan','root','rational','sinh_inverse','erf'):
  for terms in ('adj','diag2','full2'):
   summary=[]
   for folds in (random,radial):
    r=np.zeros(len(x));
    for f in folds:
     i=np.setdiff1d(np.arange(len(x)),f); r[f]=np.linalg.norm(predict(x[f],fit(x[i],y[i],k,terms),k,terms)-y[f],axis=1)
    summary.append(score(r,folds))
   print(k,terms,*summary)
 for k in ('tanh','atan','root','rational','erf'):
  m=fit(x,y,k,'diag2'); r=np.linalg.norm(predict(x,m,k,'diag2')-y,axis=1); print(k,'coeffs',np.round(m,4),'largest rows',[(int(i),round(r[i],4)) for i in np.argsort(r)[-8:]])
