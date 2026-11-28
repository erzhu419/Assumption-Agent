import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; sw=s['sensor_calibration_sweeps']; X,Y=sw['ref_rosette'],sw['raw_rosette']; bank=sw['single_bank_sweeps']
np.set_printoptions(precision=6,suppress=True)

def feats(x,d):
 return np.column_stack([np.prod(x[:,t],axis=1) if t else np.ones(len(x)) for k in range(d+1) for t in combinations_with_replacement(range(x.shape[1]),k)])

def fitfun(fun,q,X,Y):
 # Gross corruptions are far outside the physical sweep range.
 use=abs(Y)<5; X,Y=X[use],Y[use]; q=least_squares(lambda p:fun(X,p)-Y,q,loss='soft_l1',f_scale=.001,max_nfev=350).x
 return q

for j in range(4):
 use=bank['bank_id']==j; xb=np.zeros((use.sum(),4)); xb[:,0]=bank['ref_signal'][use]; x=X[:,(j+np.arange(4))%4]; y=Y[:,j]; z=np.r_[x,xb]; v=np.r_[y,bank['raw_readout'][use]]
 print('BANK',j)
 iso=np.count_nonzero(x[:,1:],axis=1)==0; xi=np.r_[x[iso,0],xb[:,0]]; yi=np.r_[y[iso],v[len(y):]]
 for name,f,p in [('tanh',lambda a,q:q[0]+q[1]*np.tanh(q[2]*a),[0,2,1]),('tanh3',lambda a,q:q[0]+q[1]*np.tanh(q[2]*a)+q[3]*a**3,[0,2,1,0]),('tanhlin',lambda a,q:q[0]+q[1]*np.tanh(q[2]*a)+q[3]*a,[0,2,1,0]),('tanharg',lambda a,q:q[0]+q[1]*np.tanh(q[2]*a+q[3]*a**3),[0,2,1,0]),('rational',lambda a,q:q[0]+(q[1]*a+q[3]*a**3)/(1+q[2]*a*a+q[4]*a**4),[0,2,1,0,0]),('atan3',lambda a,q:q[0]+q[1]*np.arctan(q[2]*a)+q[3]*a**3,[0,2,1,0])]:
  q=fitfun(f,p,xi,yi); print(name,q,np.quantile(abs(f(xi,q)-yi),[.5,.9]))
 for label in ('poly3','poly4','poly5','tanh3poly2','tanh3poly3','tanhar2','tanhar2poly3','tanhar3poly3','separate'):
  def design(z):
   a,b,c,d=z.T
   if label[:4]=='poly':return feats(z,int(label[-1]))
   if label=='tanh3poly2':return np.c_[a**3,feats(z,2)]
   if label=='separate':return np.c_[a**3,feats(z,3)]
   return feats(z,3)
  D=design(z); Q=np.c_[z, D]; size=D.shape[1]; p=np.r_[2.,1.1,0.,0.,0.,0.,0.,np.zeros(size)];p[7]=np.median(y)
  def f(Q,p):
   a,b,c,d=Q[:,:4].T; D=Q[:,4:]; out=D@p[7:];
   if label.startswith('poly'):return out
   t=p[1]*a
   if label.startswith('tanhar'):t=t+p[2]*b+p[3]*c+p[4]*d+p[5]*a*c+p[6]*c*c
   if label=='separate':return out+p[0]*np.tanh(t)+p[2]*np.tanh(p[3]*c)
   return out+p[0]*np.tanh(t)
  if label=='separate':p[2:4]=[.2,1.]
  es=[]
  for i in range(5):
   test=np.arange(len(z))%5==i;q=fitfun(f,p,Q[~test],v[~test]);es.extend((f(Q[test & (abs(v)<5)],q)-v[test & (abs(v)<5)]).tolist())
  print(label,np.quantile(abs(np.array(es)),[.5,.9,.99]))
