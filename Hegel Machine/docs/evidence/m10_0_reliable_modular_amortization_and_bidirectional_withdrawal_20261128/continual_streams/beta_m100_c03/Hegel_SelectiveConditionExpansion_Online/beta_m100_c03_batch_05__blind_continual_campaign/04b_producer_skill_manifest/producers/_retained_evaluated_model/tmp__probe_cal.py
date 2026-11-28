import pickle,numpy as np
from itertools import combinations,combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x=c['ref_rosette']; y=c['raw_rosette']; good=np.max(abs(y),axis=1)<6
np.set_printoptions(precision=6,suppress=True,linewidth=130)
print('axis rows',np.c_[x[:16],y[:16]])
print('bank data',np.column_stack(list(c['single_bank_sweeps'].values())).reshape(4,12,3).transpose(1,0,2)[:, :, 1:])
i,j=np.triu_indices(4,1)
for name,terms in [('affine',[x]),('quadratic',[x,x*x,x[:,i]*x[:,j]]),('cubic',[np.column_stack([np.prod(x[:,k],axis=1) for p in range(1,4) for k in combinations_with_replacement(range(4),p)])]),('tanh',[x,np.tanh(x),x*x,x[:,i]*x[:,j]]),('sine',[x,np.sin(x),x*x,x[:,i]*x[:,j]]),('tanh+cubic',[x,np.tanh(x),x*x,x**3,x[:,i]*x[:,j]]),('rat',[x,x/(1+x*x),x*x,x[:,i]*x[:,j]])]:
 z=np.column_stack([np.ones(len(x)),*terms]); pred=[]
 for k in range(len(x)):
  train=good.copy();train[k]=False
  pred.append(z[k]@np.linalg.lstsq(z[train],y[train],rcond=None)[0])
 r=(np.asarray(pred)-y)[good];print(name,'LOO rms',np.sqrt(np.mean(r*r,axis=0)),'max',np.max(abs(r),axis=0))
 if name in ['quadratic','tanh+cubic','cubic']:
  b=np.linalg.lstsq(z[good],y[good],rcond=None)[0];print('terms',b.round(5)); print('axis residual', (z@b-y)[:13].round(5))
