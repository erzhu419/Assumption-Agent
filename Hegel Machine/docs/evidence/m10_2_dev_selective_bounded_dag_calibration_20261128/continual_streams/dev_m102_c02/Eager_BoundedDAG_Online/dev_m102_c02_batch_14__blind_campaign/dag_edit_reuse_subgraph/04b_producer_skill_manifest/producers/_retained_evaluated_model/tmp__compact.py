import numpy as np,pickle
from scipy.optimize import least_squares, minimize_scalar
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True,linewidth=130)
def fit(d,y):
 b=np.linalg.lstsq(d,y,rcond=None)[0]; r=y-d@b; h=np.sum(np.linalg.qr(d,mode='reduced')[0]**2,axis=1); return np.mean(abs(r)),np.mean(abs(r)/np.maximum(1-h,1e-10).reshape((-1,)+(1,)*(r.ndim-1))),b
u=s['panel_a']['u'];v=s['panel_a']['v']; good=np.max(abs(v-u),axis=1)<2;u=u[good];v=v[good]
for j in range(4):
 a=u[:,j]; b=u[:,(j+2)%4]; y=v[:,j]; print('sensor',j)
 for lab,f in [('cube',lambda q,k:q**3),('sin',lambda q,k:np.sin(k*q)/k-q),('tanh',lambda q,k:np.tanh(k*q)/k-q),('atan',lambda q,k:np.arctan(k*q)/k-q),('rat',lambda q,k:q/(1+k*q*q)-q)]:
  for other,g in [('sin',lambda q,k:np.sin(k*q)/k-q),('tanh',lambda q,k:np.tanh(k*q)/k-q),('cube',lambda q,k:q**3)]:
   def d(k):return np.c_[np.ones(len(a)),a,b,f(a,k[0]),g(b,k[1])]
   def res(k):
    mat=d(k);return mat@np.linalg.lstsq(mat,y,rcond=None)[0]-y
   k=least_squares(res,[1.3,2.],bounds=(.03,5)).x; rr,l,co=fit(d(k),y); print(lab,other,round(rr,5),round(l,5),k,co)
 for deg in [3,5,7]:
  d=np.c_[np.ones(len(a)),*[t**i for i in range(1,deg+1,2) for t in (a,b)]];print(deg,fit(d,y))
for pn,vk in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u=s[pn]['u']; y=s[pn][vk]; print(pn,vk,len(u));
 for label,d in [('linear',np.c_[np.ones(len(u)),u]),('sep2',np.c_[np.ones(len(u)),u,u*u]),('sep3',np.c_[np.ones(len(u)),u,u*u,u**3]),('odd3',np.c_[np.ones(len(u)),u,u**3]),('full3',PolynomialFeatures(3).fit_transform(u))]:
  if len(u)<=d.shape[1]:continue
  a,b,c=fit(d,y); print(label,a,b); print(c if label in ('sep2','odd3') else '')
 for name,f in [('sin',np.sin),('tanh',np.tanh),('atan',np.arctan)]:
  def d(k):return np.c_[np.ones(len(u)),u,f(k*u)]
  def res(k):
   mat=d(k);return (mat@np.linalg.lstsq(mat,y,rcond=None)[0]-y).ravel()
  k=least_squares(res,[1.5],bounds=(.05,7)).x; a,b,c=fit(d(k),y); print(name,k,a,b)
