from readout_probe import A,c,y,old,folds,Q,pairs,study
from scipy.optimize import least_squares
import numpy as np

# Normalize links to unit local slope; a is the degree of compression.
links={
 'tanh':lambda z,a: np.tanh(a*z)/a,
 'atan':lambda z,a: np.arctan(a*z)/a,
 'asinh':lambda z,a: np.arcsinh(a*z)/a,
 'sqrt':lambda z,a:z/np.sqrt(1+(a*z)**2),
 'rational':lambda z,a:z/(1+(a*z)**2),
 'erf':lambda z,a: __import__('scipy.special',fromlist=['erf']).erf(a*z)/a*np.sqrt(np.pi)/2,
 'cubic':lambda z,a:z-a*a*z**3,
}

def fit(tr,te,link,mode):
 f=links[link]; d=np.c_[A,old]; b=np.linalg.lstsq(d[tr],y[tr],rcond=None)[0]; init=np.r_[b,0.19]
 if mode=='outside': fn=lambda p,i:p[0]+f(c[i]@p[1:5],p[-1])+old[i]*p[5]
 if mode=='inside': fn=lambda p,i:p[0]+f(d[i,1:]@p[1:6],p[-1])
 if mode=='all': fn=lambda p,i:f(d[i]@p[:6],p[-1])
 if mode=='shift':
  init=np.r_[b,0.,0.19]; fn=lambda p,i:p[0]+f(c[i]@p[1:5]+p[6],p[-1])+old[i]*p[5]
 if mode=='bilinear_rational': fn=lambda p,i:p[0]+(c[i]@p[1:5])/(1+(p[-1]*old[i])**2)+old[i]*p[5]
 p=least_squares(lambda p:fn(p,tr)-y[tr],init,bounds=(np.r_[np.full(len(init)-1,-np.inf),1e-5],np.r_[np.full(len(init)-1,np.inf),1.5]),max_nfev=300).x
 return np.mean(np.abs(fn(p,te)-y[te])),p

scores=[]
for k in links:
 for mode in ('outside','inside','all','shift'):
  e=np.array([fit(tr,te,k,mode)[0] for tr,te in folds]); scores.append((e.mean(),k,mode,e))
for _,k,m,e in sorted(scores)[:17]:
 print(k,m,'holdout/oof/worst',np.round([e[0],e[1:].mean(),e.max()],6),'holdout-fit',fit(*folds[0],k,m)[1].round(6))
print('bilinear denominator', [round(fit(tr,te,'rational','bilinear_rational')[0],6) for tr,te in folds])
p=study['panel_c']; u=p['u']; x=np.c_[np.ones(len(u)),u]; cross=np.column_stack([u[:,i]*u[:,j] for i,j in pairs]);
for name in ('v1','v2'):
 for X in (x,np.c_[x,np.sum(u[:,:2]*u[:,2:],axis=1)],np.c_[x,cross]):
  b=np.linalg.lstsq(X[4:],p[name][4:],rcond=None)[0]; print('panel_c',name,X.shape[1],np.mean(np.abs(X[:4]@b-p[name][:4])))
