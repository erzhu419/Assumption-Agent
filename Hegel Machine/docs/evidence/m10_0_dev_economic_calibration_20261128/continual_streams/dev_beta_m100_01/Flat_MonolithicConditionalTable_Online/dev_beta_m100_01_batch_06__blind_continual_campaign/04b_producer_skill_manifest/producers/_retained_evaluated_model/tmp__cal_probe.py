import numpy as np,pickle
from scipy.optimize import least_squares
from scipy.special import erf
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; X,Y=c['ref_rosette'],c['raw_rosette']; clean=np.max(np.abs(Y),axis=1)<4; X,Y=X[clean],Y[clean]; q=c['single_bank_sweeps']; B=q['bank_id']; G=q['ref_signal']; T=q['raw_readout']; plus=np.eye(4)[B]*G[:,None]; forms={'tanh':np.tanh,'erf':erf,'atan':np.arctan,'sqrt':lambda z:z/np.sqrt(1+z*z),'rational':lambda z:z/(1+z*z)}
axis=np.count_nonzero(X,axis=1)==1
for j in range(4):
 a=X[:,j]; b=X[:,(j+2)%4]; m=(B==j)&(abs(T)<4); U=np.r_[X,plus[m]]; v=np.r_[Y[:,j],T[m]]; alone=np.r_[axis,np.ones(sum(m),bool)]; fold=np.arange(len(U))%4
 print('channel',j, 'clean axis median',np.median(Y[axis&(a==0)&(b==0),j]))
 for k,f in forms.items():
  scores=[]
  for typ in range(4):
   def design(P):
    aa,bb=U[:,j],U[:,(j+2)%4]; z=aa+P[1]*bb; scale=np.exp(P[0]); cols=[np.ones(len(U)),f(scale*(z if typ<2 else aa)),aa,aa**2,aa**3,bb,bb**2,bb**3,aa*bb];
    if typ%2: cols.extend([aa**2*bb,aa*bb**2,U[:,(j+1)%4],U[:,(j-1)%4],U[:,(j+1)%4]*U[:,(j-1)%4]])
    if typ>=2: cols.append(f(np.exp(P[1])*bb))
    return np.column_stack(cols)
   def fit(train):
    def lin(P):
     D=design(P); bet=np.linalg.lstsq(D[train],v[train],rcond=None)[0];return D@bet
    p=least_squares(lambda p:(lin(p)-v)[train],[0,.15],bounds=([-2,-3],[2,2])).x
    return lin(p)-v,p
   err=np.zeros(len(U))
   for i in range(4):
    pred,_=fit(fold!=i); err[fold==i]=pred[fold==i]
   res,p=fit(np.ones(len(U),bool)); scores.append((typ,[round(t,7) for t in (np.sqrt(np.mean(err[alone]**2)),np.sqrt(np.mean(err[~alone]**2)),np.max(abs(res)))],p.round(3).tolist()))
  print(k,scores)
