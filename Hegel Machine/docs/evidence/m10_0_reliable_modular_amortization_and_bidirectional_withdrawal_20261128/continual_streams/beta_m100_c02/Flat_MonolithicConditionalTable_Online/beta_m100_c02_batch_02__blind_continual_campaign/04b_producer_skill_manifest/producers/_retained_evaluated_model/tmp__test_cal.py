import pickle,numpy as np,itertools
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; b=c['single_bank_sweeps']; print('gross outliers',np.argwhere(abs(y)>5).tolist())
for k in range(4):
 mask=(b['bank_id']==k)&(abs(b['raw_readout'])<5); t=b['ref_signal'][mask]; z=b['raw_readout'][mask]; print('bank', k)
 for d in range(1,6):
  a=np.polynomial.polynomial.polyfit(t,z,d); print(d,np.round(a,7), np.sqrt(np.mean((np.polynomial.polynomial.polyval(t,a)-z)**2)))
for d in range(1,5):
 terms=[t for i in range(d+1) for t in itertools.combinations_with_replacement(range(4),i)]
 X=np.array([np.prod(x[:,t],axis=1) for t in terms]).T
 for k in range(4):
  good=abs(y[:,k])<5; errs=[]
  for fold in range(5):
   tr=good&(np.arange(len(x))%5!=fold); va=good&~tr
   fit=np.linalg.lstsq(X[tr],y[tr,k],rcond=None)[0]; errs.append(np.sqrt(np.mean((X[va]@fit-y[va,k])**2)))
  fit=np.linalg.lstsq(X[good],y[good,k],rcond=None)[0]; print('degree/bank/worst',d,k,max(errs),'coef',[(t,round(a,5)) for t,a in zip(terms,fit) if abs(a)>.008][:35])
