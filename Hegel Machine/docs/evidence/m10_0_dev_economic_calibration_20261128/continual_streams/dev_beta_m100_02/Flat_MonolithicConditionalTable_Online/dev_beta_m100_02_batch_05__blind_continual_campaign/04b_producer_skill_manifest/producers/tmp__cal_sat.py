import itertools,pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
sw=s['sensor_calibration_sweeps']; X=sw['ref_rosette']; Y=sw['raw_rosette']; single=sw['single_bank_sweeps']; ids=np.ravel(single['bank_id']); sig=np.ravel(single['ref_signal']); z=np.ravel(single['raw_readout'])

def phi(u,f):
 if f=='tanh': return np.tanh(u)
 if f=='atan': return np.arctan(u)
 if f=='sqrt': return u/np.sqrt(1+u*u)
 if f=='rational': return u/(1+u*u)
 if f=='erf':
  from scipy.special import erf
  return erf(u)

def features(x,k):
 return np.column_stack([x[:,k],x[:,(k+2)%4],x[:,(k+1)%4],x[:,(k+3)%4],x[:,k]*x[:,(k+1)%4],x[:,k]*x[:,(k+2)%4],x[:,k]*x[:,(k+3)%4],*[x[:,i]*x[:,j] for i,j in itertools.combinations_with_replacement(range(4),2) if i!=k and j!=k]])

def data(k):
 sel=ids==k; ix=np.zeros((sum(sel),4)); ix[:,k]=sig[sel]; x=np.r_[X,ix]; y=np.r_[Y[:,k],z[sel]]; good=abs(y)<5
 return features(x[good],k),y[good],np.flatnonzero(good)

def pred(b,D,f,form):
 if form=='inside': return b[0]+b[1]*phi(b[2]+D@b[3:],f)
 if form=='outside': return b[0]+b[1]*phi(b[2]+b[3]*D[:,0],f)+D[:,1:]@b[4:]
 if form=='both': return b[0]+b[1]*phi(b[2]+D[:,:4]@b[3:7],f)+D@b[7:]
 if form=='inside_tail': return b[0]+b[1]*phi(b[2]+D@b[3:-1],f)+b[-1]*D[:,0]

def fit(D,y,f,form):
 b=np.r_[0.,max(1,np.max(abs(y))),0.,1.,-.2,np.zeros(D.shape[1]-2)]
 if form=='both': b=np.r_[b[:7],np.zeros(D.shape[1])]
 if form=='inside_tail': b=np.r_[b,0.]
 return least_squares(lambda b:pred(b,D,f,form)-y,b,loss='soft_l1',f_scale=.001,max_nfev=180).x

if __name__=='__main__':
 for k in range(4):
  D,y,idx=data(k); print('channel',k)
  for f,form in itertools.product(('tanh','atan','sqrt','rational','erf'),('inside','outside','both','inside_tail')):
   e=np.empty(len(y)); b=fit(D,y,f,form)
   for j in range(4):
    tr=idx%4!=j; e[~tr]=pred(fit(D[tr],y[tr],f,form),D[~tr],f,form)-y[~tr]
   print(f,form,'oof median/90/max',np.round(np.quantile(abs(e),[.5,.9,1]),6),'fit',np.round(b,4))
