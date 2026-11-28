import numpy as np,pickle
from scipy.optimize import least_squares
b=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']['sensor_calibration_sweeps']; x,y=b['ref_rosette'],b['raw_rosette']; good=np.max(abs(y),axis=1)<5; x,y=x[good],y[good]; bank=b['single_bank_sweeps']; pair=[2,3,0,1]
def basis(name,t,k):
 if name=='tanh': return np.tanh(k*t)
 if name=='atan': return np.arctan(k*t)
 if name=='rat': return t/(1+k*t*t)
 if name=='sqrt': return t/np.sqrt(1+k*t*t)
 if name=='erf':
  from scipy.special import erf
  return erf(k*t)
 if name=='exp': return t*np.exp(-k*t*t)
 if name=='pole': return t/(1+k*t)
 if name=='asinh': return np.arcsinh(k*t)
 if name=='expasym': return np.expm1(k*t)
 if name=='sat2': return t*t/(1+k*t*t)
 if name=='gauss2': return 1-np.exp(-k*t*t)
 if name=='tanh2': return np.tanh(k*t)**2
 if name=='sig2': return np.tanh(k*t*t)

def variable(f,y,starts,lower,upper):
 def calc(k):
  A=f(k); return A@np.linalg.lstsq(A,y,rcond=None)[0]-y
 opts=[]
 for start in starts:
  r=least_squares(calc,start,bounds=(lower,upper),max_nfev=120,gtol=1e-11); opts.append((np.mean(r.fun**2)**.5,r.x,np.max(abs(r.fun))))
 return min(opts,key=lambda a:a[0])
for j in [0,1]:
 a=(bank['bank_id']==j)&(abs(bank['raw_readout'])<5); t,z=bank['ref_signal'][a],bank['raw_readout'][a]; ranking=[]
 for d in ['tanh','atan','rat','sqrt','erf','exp','asinh']:
  for p in [[],[1],[3],[1,3],[3,5]]:
   f=lambda k: np.column_stack([np.ones(len(t)),basis(d,t,k[0])]+[t**n for n in p]); ranking.append((variable(f,z,[[.5],[1.5]],[.01],[6]),d,p))
 print('bank',j,sorted(ranking,key=lambda a:a[0][0])[:9])
for j in range(4):
 t,u=x[:,j],x[:,pair[j]]; z=y[:,j]; ranking=[]
 for d in ['tanh','rat','atan','sqrt','erf','asinh']:
  for p in [[3],[1],[1,3],[]]:
   for c in ['poly','pole','expasym','asinh','tanh','rat','sat2','gauss2','tanh2','sig2']:
    def f(k):
     C=[u,u*u,u**3,u**4,u**5,u**6] if c=='poly' else [u,basis(c,u,k[1])]+([u*u] if c not in ['pole','sat2','gauss2','tanh2','sig2'] else [])
     return np.column_stack([np.ones(len(t)),basis(d,t,k[0])]+[t**n for n in p]+C)
    upper=.95/max(abs(u)) if c=='pole' else 6.; starts=[[1,.15],[1,.4 if c=='pole' else 1]]
    ranking.append((variable(f,z,starts,[.02,.005],[6.,upper]),d,p,c))
 print('rosette',j,sorted(ranking,key=lambda a:a[0][0])[:8])
