import pickle,numpy as np
from scipy.optimize import least_squares
from numpy.polynomial.chebyshev import chebvander
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x=c['ref_rosette']; y=c['raw_rosette']; d=c['single_bank_sweeps']; ids=d['bank_id']; ref=d['ref_signal']; raw=d['raw_readout']; keep=np.max(abs(y),axis=1)<6; bk=abs(raw)<6
bank=np.r_[np.tile(np.arange(4),keep.sum()),ids[bk]]; u=np.r_[x[keep].ravel(),ref[bk]]; v=np.r_[np.roll(x[keep],2,axis=1).ravel(),np.zeros(bk.sum())]; z=np.r_[y[keep].ravel(),raw[bk]]
def diag(t,a):
 q=t*t; return t*(1+q*(a[0]+q*a[1]))/(1+q*(a[2]+q*(a[3]+q*a[4])))
for deg in (7,9,11):
 for shared in (False,True):
  C=chebvander(v/1.8,deg)[:,1:]-chebvander(0.,deg)[:,1:]; M=np.eye(4)[bank]; p0=np.r_[[-.19,.066,.21,.024,.0024],np.zeros(deg)] if shared else [-.19,.066,.21,.024,.0024]
  def calc(p, train, ret=False):
   q=diag(u,p[:5]); H=np.c_[M,M*(q+C@p[5:])[:,None]] if shared else np.c_[M,M*q[:,None],(M[:,:,None]*C[:,None,:]).reshape(len(u),-1)]; b=np.linalg.lstsq(H[train],z[train],rcond=None)[0]; P=H@b
   return (P,b) if ret else (P-z)[train]
  def fit(train):
   return least_squares(calc,p0,args=(train,),bounds=(np.r_[[-1,0,0,0,0],np.full(len(p0)-5,-1)],np.r_[[1,1,2,2,1],np.full(len(p0)-5,1)]),max_nfev=100,gtol=1e-12).x
  allrows=np.ones(len(u),bool); p=fit(allrows); pred,b=calc(p,allrows,True); errors=[]; worst=[]
  for k in range(4):
   test=np.zeros(len(u),bool); test[:keep.sum()*4]=np.repeat(np.arange(keep.sum())%4==k,4); err=(calc(fit(~test),~test,True)[0]-z)[test].reshape(-1,4); errors.append(err); worst.append(np.sqrt(np.mean(err**2,0)).max())
  print(deg,shared,'train',np.sqrt(np.mean((pred-z)**2)),'worst forward fold',max(worst),'max cv',max(np.max(abs(e)) for e in errors),'shape',p.round(7),'gain',b[4:8].round(6))
