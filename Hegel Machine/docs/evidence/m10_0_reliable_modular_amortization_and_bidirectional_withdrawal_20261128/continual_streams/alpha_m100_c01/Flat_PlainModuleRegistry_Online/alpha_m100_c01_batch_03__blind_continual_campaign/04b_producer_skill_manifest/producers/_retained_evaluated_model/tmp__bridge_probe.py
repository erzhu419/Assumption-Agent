import pickle,itertools,numpy as np
from collections import Counter
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=D['calibration_load_cells']; X=c['true_channel_loads']; Y=c['bridge_raw'];
A=[(0,0,0,0)]+[tuple(int(i==j) for i in range(4)) for j in range(4)]
def terms(k):
 return [tuple(Counter(s)[i] for i in range(4)) for s in itertools.combinations_with_replacement(range(4),k)]
def phi(x,p):return np.prod(x[:,None,:]**np.array(p),axis=-1)
def fit(z,y):
 b=np.linalg.lstsq(z,y,rcond=None)[0]
 for s in [.4,.1,.02,.004,.001,.00035,.0002,.0002]:
  for _ in range(15):
   r=np.linalg.norm(y-z@b,axis=1); w=(1+(r/s)**2)**(-.5); b=np.linalg.lstsq(z*w[:,None],y*w[:,None],rcond=None)[0]
 return b
core=A+terms(2)+[k for k in terms(3) if max(k)==3]; full=A+terms(2)+terms(3)
for name,p in [('affine',A),('diag cubic only',A+[k for k in terms(3) if max(k)==3]),('quad plus diag cubic',core),('full cubic',full),('full quartic',full+terms(4))]:
 z=phi(X,p); b=fit(z,Y); r=np.linalg.norm(Y-z@b,axis=1); print(name,'rank',np.linalg.matrix_rank(z),'fit row norm quantiles',np.quantile(r,[.5,.8,.9,.95,1]).round(7));
 if name in ('quad plus diag cubic','full cubic'):
  print('terms:',[(k,np.round(v,5).tolist()) for k,v in zip(p,b) if np.max(abs(v))>.002]); print('largest row residuals:',[(int(i),round(r[i],5)) for i in np.argsort(r)[-12:]])
 if len(p)<=35:
  rs=[]
  for i in range(len(X)):
   m=np.arange(len(X))!=i; rs.append(np.linalg.norm(Y[i]-z[i]@fit(z[m],Y[m])))
  print('LOO',np.quantile(rs,[.5,.8,.9,.95,.975,1]).round(7))
print('axial rows'); a=np.count_nonzero(X,axis=1)<=1; print(np.c_[np.flatnonzero(a),X[a],Y[a]].round(5))
