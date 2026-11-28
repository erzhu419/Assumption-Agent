import pickle,itertools,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; good=(abs(y)<5).all(1); X,Y=x[good],y[good]
np.set_printoptions(precision=6,suppress=True,linewidth=130)
print('axis output with nominal zero offset removed:');print(np.c_[x,y-[-.06,.07,.06,-.05]][:16])
def odd(t,k,m):
 if m=='T':return np.tanh(k*t)
 if m=='R':return t/(1+k*t*t)
 if m=='A':return np.arctan(k*t)
 if m=='S':return t/np.sqrt(1+k*t*t)
 if m=='L':return t

def basis(x,j,k,m,extras):
 t=x[:,j]; u=x[:,(j+2)%4]; a,b=k; p,q=m[:2]; f,g=odd(t,a,p),odd(u,b,q)
 A=[np.ones(len(t)),f,g]
 for c in m[2:]:
  A.append({'Q':u*u,'D':t*t,'U':u*u/(1+b*u*u),'V':u*u/(1+b*b*u*u),'W':g*g,'I':t*u,'H':t,'J':u,'F':f*f,'G':t*t/(1+a*t*t)}[c])
 if extras: A.extend(x[:,[i for i in range(4) if i not in (j,(j+2)%4)]].T)
 return np.column_stack(A)
def fit(x,y,j,m,extras=False):
 def res(k,coef=False):
  A=basis(x,j,k,m,extras); p=np.linalg.lstsq(A,y[:,j],rcond=1e-10)[0]; return p if coef else A@p-y[:,j]
 k=least_squares(res,[1.,.8],bounds=(.005,10),max_nfev=100,ftol=1e-10,gtol=1e-11).x; return k,res(k,True)
rms=lambda z:np.sqrt(np.mean(z*z)); out=[]
for a,b in itertools.product('TRAS','LTRAS'):
 for ext in ('Q','U','V','W','QI','UI','VI','WI','DQ','DU','DV','DW','HQU','HQ','HU','HW','HQI','HUI','HWI','QJ','UJ','WJ','QG','UG','WG'):
  m=a+b+ext; f=[fit(X,Y,j,m) for j in range(4)]; r=np.column_stack([basis(X,j,k,m,False)@p-Y[:,j] for j,(k,p) in enumerate(f)]); out.append((rms(r),m,f,r))
for v,m,f,r in sorted(out,key=lambda a:a[0])[:14]:
 scores=[]
 for order in (np.arange(len(X)),np.argsort(np.linalg.norm(X,axis=1)),np.random.default_rng(12).permutation(len(X))):
  for te in np.array_split(order,5):
   tr=np.setdiff1d(np.arange(len(X)),te)
   rr=[]
   for j in range(4):
    k,p=fit(X[tr],Y[tr],j,m); rr.append(basis(X[te],j,k,m,False)@p-Y[te,j])
   scores.append(rms(np.array(rr)))
 print(m,'train/heldout',v,max(scores));
 if m in [a[1] for a in sorted(out,key=lambda a:a[0])[:3]]: print(f); print('axis and mixed residual',r[:17]);print('residue correlations',np.corrcoef(np.c_[X,r].T)[:4,4:])
