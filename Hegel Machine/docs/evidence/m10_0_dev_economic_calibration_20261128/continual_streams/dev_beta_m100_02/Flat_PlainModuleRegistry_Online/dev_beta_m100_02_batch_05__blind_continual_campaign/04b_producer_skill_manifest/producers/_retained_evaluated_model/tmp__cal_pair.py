import pickle,numpy as np
from scipy.optimize import least_squares
from scipy.special import erf
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; X,Y=c['ref_rosette'],c['raw_rosette']; X,Y=X[(abs(Y)<5).all(1)],Y[(abs(Y)<5).all(1)]

def sat(x,k,kind):
 if kind=='T': return np.tanh(k*x)
 if kind=='A': return np.arctan(k*x)
 if kind=='S': return x/np.sqrt(1+k*x*x)
 if kind=='R': return x/(1+k*x*x)
 if kind=='E': return erf(k*x)
 return x/(1+k*abs(x))

def design(X,j,k,m):
 t,u=X[:,j],X[:,(j+2)%4]; v=t+k[2]*u; h=k[1]; f=sat(v,k[0],m[0]);
 A=[np.ones(len(t)),f,u,t,u*u,t*t]
 extra={'a':[abs(u),abs(t)],'r':[u/(1+h*u*u),u*u/(1+h*u*u)],'t':[np.tanh(h*u),np.tanh(h*u)**2],'p':[v**3/(1+h*v*v),u**3/(1+h*u*u)],'q':[u*u*t,u*t*t],'s':[sat(t,h,m[0]),sat(u,h,m[0])],'m':[t*u,t*t*t,u*u*u],'b':[np.sin(h*u),np.cos(h*u)],'d':[np.sin(h*u),np.sin(h*t)],'h':[v**2/(1+h*v*v),u*u/(1+h*u*u)],'w':[t/(1+h*t*t),u/(1+h*u*u)]}
 for key in m[1:]: A+=extra[key]
 return np.column_stack(A)

def fit(X,Y,j,m):
 def resid(k,coeff=False):
  A=design(X,j,k,m); p=np.linalg.lstsq(A,Y[:,j],rcond=1e-9)[0]
  return p if coeff else A@p-Y[:,j]
 start=[[1,1,c] for c in (0,-.18,.18)]; best=min((least_squares(resid,k,bounds=([.03,.03,-.8],[8,10,.8]),max_nfev=100,gtol=1e-10) for k in start),key=lambda r:r.cost)
 return best.x,resid(best.x,True)

def pred(X,m,p):return np.column_stack([design(X,j,k,m)@b for j,(k,b) in enumerate(p)])
def train(X,Y,m):return [fit(X,Y,j,m) for j in range(4)]
def rms(z):return np.sqrt(np.mean(z*z))
if __name__=='__main__':
 results=[]
 for a in 'TASREB':
  for b in ('','a','r','t','p','q','s','m','b','d','h','w','rq','rh','bh','dh'):
   m=a+b; p=train(X,Y,m); results.append((rms(pred(X,m,p)-Y),m,p))
 for e,m,p in sorted(results)[:12]:
  errs=[]
  for order in (np.arange(len(X)),np.argsort(np.max(abs(X),axis=1)),np.random.default_rng(45).permutation(len(X))):
   for te in np.array_split(order,5):
    tr=np.setdiff1d(np.arange(len(X)),te); errs.append(rms(pred(X[te],m,train(X[tr],Y[tr],m))-Y[te]))
  print(m,e,max(errs),flush=True)
  if e<.004: print([(np.round(k,5),np.round(b,5)) for k,b in p],flush=True)
