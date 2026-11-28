import numpy as np,pickle
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import least_squares
with open('/holdout_task/public_batch.pkl','rb') as f: d=pickle.load(f)[0]['dataset']
c=d['calibration_load_cells']; x=c['true_channel_loads']; y=c['bridge_raw']; np.set_printoptions(precision=5,suppress=True,linewidth=160)
print('axis samples:'); print(np.c_[x[:20],y[:20]])
print('mixed ranges',x[16:].min(0),x[16:].max(0)); print('gross rows',np.where(np.max(abs(y),axis=1)>4)[0]); clean=np.max(abs(y),axis=1)<4

def fit(A,Y):
 b=np.linalg.lstsq(A,Y,rcond=None)[0]
 for s in (.3,.1,.03,.01,.003):
  for _ in range(10):
   w=1/np.sqrt(1+np.sum((A@b-Y)**2,axis=1)/s**2)
   b=np.linalg.lstsq(A*w[:,None],Y*w[:,None],rcond=None)[0]
 return b

def test(name,A):
 r=[]
 for i in np.flatnonzero(clean):
  m=np.arange(len(x))!=i; r.append(A[i]@fit(A[m],y[m])-y[i])
 r=np.array(r); print(name,'rank',np.linalg.matrix_rank(A),'OOF channel RMSE',np.sqrt(np.mean(r*r,axis=0)),'boundary max',np.max(abs(r[np.sum(x[clean]!=0,axis=1)==1]),axis=0),'mixed max',np.max(abs(r[np.sum(x[clean]!=0,axis=1)>1]),axis=0)); return fit(A,y)
for k in (1,2,3):
 p=PolynomialFeatures(k); A=p.fit_transform(x); b=test('poly'+str(k),A)
 if k in (2,3): print(*[(n,bb) for n,bb in zip(p.get_feature_names_out(),b) if np.max(abs(bb))>.015],sep='\n')
for k in (1,2,3):
 test('tanh'+str(k),np.c_[PolynomialFeatures(k).fit_transform(x),np.tanh(x), np.sin(x)])
