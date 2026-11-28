import pickle,numpy as np
from scipy.optimize import least_squares
from scipy.special import erf
s=pickle.load(open('public_batch.pkl','rb'))[0]['study']['sensor_calibration_sweeps']; X,Y=s['ref_rosette'],s['raw_rosette']; keep=np.max(abs(Y),axis=1)<4; X,Y=X[keep],Y[keep]; F=np.random.default_rng(193).permutation(len(X))%5
laws={'exp':lambda v,k:v*np.exp(-k*v*v),'erf':lambda v,k:erf(k*v),'sin':lambda v,k:np.sin(k*v),'tanh':lambda v,k:np.tanh(k*v),'rational':lambda v,k:v/(1+k*v*v),'sinexp':lambda v,k:np.sin(k[0]*v)*np.exp(-k[1]*v*v),'rat4':lambda v,k:v/(1+k[0]*v*v+k[1]*v**4),'poly9':lambda v,k:v**3}
for label,fn in laws.items():
 for extra in (False,True):
  err=np.empty_like(Y); summary=[]
  for i,j in enumerate([2,3,0,1]):
   x,z=X[:,i],X[:,j]; y=Y[:,i]; anchor=(x==0)&(z==0)
   def A(k):
    p=k[1:] if label in ('sinexp','rat4') else k[1]; a=[x,fn(x,p),z,z*z/(1+k[0]*z*z),z**3/(1+k[0]*z*z)]
    if extra:a.extend([x**3,x*x])
    if label=='poly9':a.extend([x**5,x**7,x**9]);a=[a[u] for u in range(len(a)) if not(extra and u==1)]
    return np.column_stack(a)
   def fit(m):
    b=np.median(y[m&anchor]) if np.any(m&anchor) else np.mean(y[m]-np.mean(y[m]));
    def solve(k):
     a=A(k)[m]; c=np.linalg.lstsq(a,y[m]-b,rcond=1e-10)[0]; return a@c+b-y[m],c
    starts=[(.6,h,.25) if label in ('sinexp','rat4') else (.6,h) for h in (.6,1.5,2.5)]
    o=min((least_squares(lambda k:solve(k)[0],start,bounds=(.00001,6),max_nfev=100) for start in starts),key=lambda o:o.cost); return o.x,solve(o.x)[1],b
   for f in range(5):
    a=F==f; k,c,b=fit(~a); err[a,i]=A(k)[a]@c+b-y[a]
   k,c,b=fit(F>=0); summary.append((np.round(k,4).tolist(),np.round(c,4).tolist(),round(np.sqrt(np.mean((A(k)@c+b-y)**2)),7)))
  e=np.linalg.norm(err,axis=1); print(label,extra,'OOF p90/max',np.round(np.quantile(e,[.9,1]),7),summary)
