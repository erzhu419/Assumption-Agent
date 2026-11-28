import numpy as np
from scipy.optimize import least_squares
from ros_probe import x,y,fit
from ros_joint_input import bank
valid=fit(x,y,'affine')[1]
folds=[np.array([],int),*np.array_split(np.argsort(np.linalg.norm(x,axis=1)),5)]
def design(x,j,k,mode):
 a,b=x[:,j],x[:,j^2]; u=np.tanh(k[0]*a); v=np.tanh(k[0]*b); d=[np.ones(len(a)),a,u,b,1-np.cos(k[1]*b)]
 if 'r' in mode:d += [b/(1+b*b)]
 if 'i' in mode:d += [a*b,u*b,a*v,u*v]
 if 'm' in mode:d += [np.tanh(k[0]*a+k[2]*b)-u]
 if 'g' in mode:d += [a/(1+a*a)]
 if 'e' in mode:d += [b*b/(1+b*b)*a,b*b/(1+b*b)*u,a*a/(1+a*a)*b]
 if 'o' in mode:d += [np.sin(2*a),np.sin(2*b)]
 if 'n' in mode:d += [x[:,j^1],x[:,j^3]]
 return np.column_stack(d)
for mode in ('r','rg','ri','re','rie','rm','rme','rgo','rgien'):
 es=[];Bs=[]
 for ids in folds:
  tr=valid.copy(); tr[ids]=False; va=ids[valid[ids]] if ids.size else np.flatnonzero(valid); errs=[];Bs0=[]
  for j in range(4):
   ss=(bank['bank_id']==j)&(np.abs(bank['raw_readout'])<4); xx=np.zeros((ss.sum(),4));xx[:,j]=bank['ref_signal'][ss]; xx=np.vstack([x[tr],xx]); yy=np.r_[y[tr,j],bank['raw_readout'][ss]]
   def project(k):
    d=design(xx,j,k,mode); p=np.linalg.lstsq(d,yy,rcond=None)[0];return d@p-yy,p
   k=least_squares(lambda k:project(k)[0],[1.35,2.,.12] if 'm' in mode else [1.35,2.],bounds=([.4,.5,-.8],[3.,4.,.8]) if 'm' in mode else ([.4,.5],[3.,4.]),max_nfev=70).x
   _,p=project(k);Bs0.append(np.r_[k,p]);errs.append(y[va,j]-design(x[va],j,k,mode)@p)
  es.append(np.sqrt(np.mean(np.array(errs)**2)));Bs.append(np.array(Bs0))
 print(mode,np.round(es,6)); print(np.round(Bs[0],5))
print('clean sweep residual geometry', len(x), np.flatnonzero(~valid), x[np.argmax(np.linalg.norm(x,axis=1))])
