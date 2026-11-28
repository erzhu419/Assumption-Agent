import itertools,pickle,numpy as np
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=p['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; q=c['single_bank_sweeps']; keep=np.max(abs(y),axis=1)<5
for j in range(4):
 m=(q['bank_id']==j)&(abs(q['raw_readout'])<5); single=np.zeros((m.sum(),4)); single[:,j]=q['ref_signal'][m]; xx=np.vstack((np.roll(x[keep],-j,axis=1),np.roll(single,-j,axis=1))); yy=np.r_[y[keep,j],q['raw_readout'][m]]; fold=np.r_[np.arange(keep.sum())%5,np.arange(m.sum())%5]; print('BANK',j)
 for deg in [2,3,4]:
  terms=[i for i in itertools.product(range(deg+1),repeat=4) if sum(i)<=deg]; A=np.prod(xx[:,None,:]**np.array(terms)[None,:,:],axis=2)
  coef=np.linalg.lstsq(A,yy,rcond=None)[0]; print(deg,'train',np.sqrt(np.mean((A@coef-yy)**2)), 'cv',max(np.sqrt(np.mean((A[fold==k]@np.linalg.lstsq(A[fold!=k],yy[fold!=k],rcond=None)[0]-yy[fold==k])**2)) for k in range(5))); print([(terms[i],round(coef[i],7)) for i in np.argsort(-abs(coef))[:13]])
 print('single',np.column_stack((q['ref_signal'][m],q['raw_readout'][m])).round(6).tolist())
