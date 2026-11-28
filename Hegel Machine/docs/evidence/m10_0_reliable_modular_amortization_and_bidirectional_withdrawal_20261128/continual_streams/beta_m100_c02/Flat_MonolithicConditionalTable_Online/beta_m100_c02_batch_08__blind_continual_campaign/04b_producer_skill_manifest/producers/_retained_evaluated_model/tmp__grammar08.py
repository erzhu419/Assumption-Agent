import pickle,numpy as np
from scipy.optimize import least_squares
from scipy.special import erf
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; d=s['sensor_calibration_sweeps']; X,Y=d['ref_rosette'],d['raw_rosette']; D=d['single_bank_sweeps']; good=(abs(Y)<5).all(1); X,Y=X[good],Y[good]; print('clean rows',len(X),'axis',np.sum(np.count_nonzero(X,axis=1)==1),'raw extrema',Y.min(0),Y.max(0))
funcs={'tanh':np.tanh,'atan':np.arctan,'soft':lambda x:x/np.sqrt(1+x*x),'erf':erf,'asinh':np.arcsinh,'gud':lambda x:2*np.arctan(np.tanh(x/2))}
for i in [0,1]:
 a=D['ref_signal'][(D['bank_id']==i)&(abs(D['raw_readout'])<5)]; y=D['raw_readout'][(D['bank_id']==i)&(abs(D['raw_readout'])<5)]; ix=(X[:,i]!=0)&(np.count_nonzero(X,axis=1)==1); a=np.r_[a,X[ix,i]]; y=np.r_[y,Y[ix,i]]; cases=[]
 for k,f in funcs.items():
  for degrees in [(),(1,),(3,),(1,3),(1,3,5)]:
   def pred(p,t): return p[0]+p[1]*f(p[2]*t)+sum(q*t**j for j,q in zip(degrees,p[3:]))
   p=least_squares(lambda p:pred(p,a)-y,[0,1.5,1,*[0]*len(degrees)],max_nfev=1500).x
   errs=[]
   for j in range(len(y)):
    mask=np.arange(len(y))!=j; q=least_squares(lambda q:pred(q,a[mask])-y[mask],p,max_nfev=200).x; errs.append(pred(q,a[j])-y[j])
   cases.append((np.sqrt(np.mean(np.square(errs))),k,degrees,np.sqrt(np.mean((pred(p,a)-y)**2)),p.round(6)))
 print('own',i,*sorted(cases,key=lambda x:x[0])[:12],sep='\n')
print('first clean representative mixed rows',np.c_[X,Y][np.count_nonzero(X,axis=1)>1][:10].round(5))
for i in range(4):
 x=X[:,i]; z=X[:,(i+2)%4]; l=X[:,(i-1)%4]; r=X[:,(i+1)%4]; y=Y[:,i]; A=np.column_stack([np.ones(len(x)),x,x**3,x**5,z,z*z,z**3,z**4,z**5,x*z,x*z*z,x*x*z,x*x*z*z,l,r,l*r,x*l,x*r,z*l,z*r]); p=np.linalg.lstsq(A,y,rcond=None)[0]; print('poly',i,np.std(A@p-y),p.round(4))
