exec(open('/tmp/composite11.py').read().split('D={')[0])
from scipy.optimize import least_squares
for k in [1.25,2.15]:
 f=np.c_[aff(u),u*u,1-np.cos(k*u)]; w=ls(f,v); print('panel decomposition',k,np.std(f@w-v),np.round(w,5))
def fitmodel(a,b,f,kind,variant):
 X=z[a]; Y=y[a]; F=f[a]; Q=f[b]; d=X.shape[1]; pp=np.r_[np.zeros(F.shape[1]),.02 if variant else 0.]; scale=np.ones(d); scale[2]=0
 def op(x,p):
  h=x@p[:d]+p[d]; k=p[d+1];
  if kind=='rat': g=1/(1+k*h*h)-1
  elif kind=='gauss':g=np.expm1(-k*h*h)
  elif kind=='sqrt':g=1/np.sqrt(1+k*h*h)-1
  elif kind=='cos':g=np.cos(k*h)-1
  elif kind=='tanh':g=np.tanh(k*h)
  elif kind=='sin':g=np.sin(k*h)
  elif kind=='soft':g=np.sqrt(1+k*h*h)-1
  else:g=h*np.exp(-k*h*h)
  return aff(x)*g[:,None] if variant else g[:,None]
 def lin(p):
  g=op(X,p); dsg=np.c_[F,g]; reg=np.r_[np.zeros(F.shape[1]),np.full(g.shape[1],.012 if variant else .002)]; return np.r_[dsg,np.diag(reg)],np.r_[Y,np.zeros(len(reg))]
 def loss(p):
  M,t=lin(p); return M@ls(M,t)-t
 lim=np.array([3,3,1.00001,3,2,12.]); lo=-lim; lo[2]=.99999; lo[-1]=.035
 init=[[-.5,0,1,0,0,1],[.5,0,1,0,0,1],[0,1,1,-1,0,1],[0,-1,1,1,0,1],[0,0,1,0,.7,1],[0,0,1,0,-.7,1],[-1,0,1,0,0,3]]
 opt=min((least_squares(loss,p,bounds=(lo,lim),max_nfev=75) for p in init), key=lambda o:o.cost)
 M,t=lin(opt.x); pred=np.c_[Q,op(z[b],opt.x)]@ls(M,t); return np.mean(abs(pred-y[b])),np.sqrt(2*opt.cost/len(a)),opt.x
bases={'aff':aff(z),'quad':quad(z),'paneleven':np.c_[aff(z),C],'zA':np.c_[aff(z),A]}
for name,f in bases.items():
 for kind in ['rat','gauss','sqrt','cos','tanh','sin','soft','bump']:
  for var in [0,1]:
   r=[fitmodel(k[:52],k[52:],f,kind,var) for k in splits[:4]]; print(name,kind,var, 'mae',np.round([t[0] for t in r],5),'train',round(r[0][1],5),'p',r[0][2].round(3),flush=True)
