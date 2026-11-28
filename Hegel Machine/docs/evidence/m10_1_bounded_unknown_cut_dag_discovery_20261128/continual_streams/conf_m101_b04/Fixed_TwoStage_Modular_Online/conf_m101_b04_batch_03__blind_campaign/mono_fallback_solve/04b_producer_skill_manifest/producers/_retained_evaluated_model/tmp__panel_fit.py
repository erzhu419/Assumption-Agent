import pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; p=s['panel_b']; u,v=p['u'],p['v']; rng=np.random.default_rng(41)
np.set_printoptions(precision=5,suppress=True)
def rms(e):return np.mean(e**2)**.5

def design(x,q,fun,mode):
 w=q[:4] if mode in ('separate','shifted') else q[0]
 b=q[4:] if mode=='shifted' else 0
 return np.c_[np.ones(len(x)),x,fun(x*w+b)]

def fit(x,y,fun,mode,init):
 def profile(q):
  a=design(x,q,fun,mode); c=np.linalg.lstsq(a,y,rcond=None)[0]; return a,c
 def err(q):
  a,c=profile(q); return (a@c-y).ravel()
 n={'shared':1,'separate':4,'shifted':8}[mode]; lo=[.25]*min(n,4)+[-1]*max(0,n-4); hi=[4.]*min(n,4)+[1]*max(0,n-4)
 opt=least_squares(err,init,bounds=(lo,hi),max_nfev=125); return opt.x,profile(opt.x)[1],rms(opt.fun)
for name,f,start in [('tanh',np.tanh,1.45),('sin',np.sin,2.85),('atan',np.arctan,1.45),('rational',lambda x:x/(1+x*x),1.0),('softsign',lambda x:x/np.sqrt(1+x*x),1.6),('cubic',lambda x:x**3,1.)]:
 for mode,init in [('shared',[start]),('separate',[start]*4),('shifted',[start]*4+[0.]*4)]:
  if name=='cubic' and mode!='shared':continue
  q,c,e=fit(u,v,f,mode,init); out=[]
  for val in np.array_split(rng.permutation(len(u)),6):
   i=np.setdiff1d(np.arange(len(u)),val); qq,cc,_=fit(u[i],v[i],f,mode,init); out.extend(design(u[val],qq,f,mode)@cc-v[val])
  print(name,mode,'rms',e,'cv',rms(np.asarray(out)),'q',q,'nonlinear coefficients',c[5:].round(3).tolist())
for j in range(4):
 for order in (3,5):
  d=np.c_[np.ones(len(u)),u,*[u**k for k in range(3,order+1,2)]]
  c=np.linalg.lstsq(d,v[:,j],rcond=None)[0]; print('odd poly',j,order,rms(d@c-v[:,j]),c[5:])
