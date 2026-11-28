import numpy as np,pickle,itertools
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True,linewidth=160)
def ls(A,B):return np.linalg.lstsq(A,B,rcond=None)[0]
def aff(u):return np.c_[np.ones(len(u)),u]
def quad(u):return PolynomialFeatures(2).fit_transform(u)
for p in 'abc':
 d=s['panel_'+p];u=d['u']; m=np.max(abs(d['v']),axis=1)<4 if p=='a' else np.ones(len(u),bool);u=u[m]
 for k in list(d)[1:]:
  y=d[k][m]; print(p,k,len(u))
  for name,F in [('quad',quad(u)),('cube',PolynomialFeatures(3).fit_transform(u)),('trig',np.c_[aff(u),np.sin(u),np.cos(u)]),('quad+odd',np.c_[quad(u),u**3,np.sin(u)])]:
   B=ls(F,y); h=np.sum(F*np.linalg.pinv(F).T,axis=1); r=F@B-y; print(name,np.std(r,0),np.mean(abs(r)/np.maximum(1-h,1e-9)[:,None],0));
   if name=='quad':print(B)
