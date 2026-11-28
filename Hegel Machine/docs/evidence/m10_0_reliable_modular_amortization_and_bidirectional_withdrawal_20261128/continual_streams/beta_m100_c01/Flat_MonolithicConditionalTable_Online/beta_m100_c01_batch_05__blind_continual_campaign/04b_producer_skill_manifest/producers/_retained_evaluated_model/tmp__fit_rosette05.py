import pickle
import numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; sw=s['sensor_calibration_sweeps']; X=sw['ref_rosette']; Y=sw['raw_rosette']; b=sw['single_bank_sweeps']; good=np.max(abs(Y),axis=1)<5; axis=np.count_nonzero(X,axis=1)==1

def own_fit(j):
 q=(b['bank_id']==j)&(abs(b['raw_readout'])<5); r=good&axis&(X[:,j]!=0); x=np.r_[b['ref_signal'][q],X[r,j]]; y=np.r_[b['raw_readout'][q],Y[r,j]]
 def pred(p,x): return p[0]+p[1]*x+p[2]*np.sin(p[3]*x)
 p=least_squares(lambda p:pred(p,x)-y,[0,1,.25,2.5],bounds=([-np.inf,-np.inf,-np.inf,1],[np.inf,np.inf,np.inf,4]),gtol=1e-12).x
 return p,pred,np.sqrt(np.mean((pred(p,x)-y)**2))

for j in range(4):
 p,own,rms=own_fit(j); q,n,o,m=X[:,[j,(j+1)%4,(j+2)%4,(j-1)%4]].T; e=Y[:,j]-own(p,q); print('bank',j,'own',p,'axis rmse',rms); k=good&axis&(o!=0); print('opposite',np.c_[o[k],e[k]])
 for name, design in [('opp sine',lambda f:np.c_[o,np.sin(f*o)]),('opp sine + pairs',lambda f:np.c_[o,np.sin(f*o),q*o,q*n,q*m,n*o,m*o,n*m]),('opp sine + poly',lambda f:np.column_stack([o,np.sin(f*o),n,m]+[X[:,i]*X[:,k] for i in range(4) for k in range(i,4)])),('sin mix',lambda f:np.c_[o,np.sin(f*o),q*o,q*n,q*m,n*o,m*o,n*m,np.sin(f*q)*o,q*np.sin(f*o),np.sin(f*q)*np.sin(f*o)])]:
  def fit(train):
   def ev(f):
    A=design(f); coef=np.linalg.lstsq(A[train],e[train],rcond=None)[0]; return e-A@coef,coef
   ff=least_squares(lambda f:ev(f[0])[0][train],[2.5],bounds=(.5,5),gtol=1e-11).x[0]; return ff,*ev(ff)
  f,res,coef=fit(good); folds=[]
  for l in range(5):
   test=good&(np.arange(len(X))%5==l)&~axis; _,rr,_=fit(good&~test); folds.append(np.sqrt(np.mean(rr[test]**2)))
  print(name,round(f,7),'rmse',np.sqrt(np.mean(res[good]**2)),'worst fold',max(folds),'coef',coef.round(6))
 print('residual quantiles on mixed sweeps',np.quantile(e[good&~axis],[0,.25,.5,.75,1]))
