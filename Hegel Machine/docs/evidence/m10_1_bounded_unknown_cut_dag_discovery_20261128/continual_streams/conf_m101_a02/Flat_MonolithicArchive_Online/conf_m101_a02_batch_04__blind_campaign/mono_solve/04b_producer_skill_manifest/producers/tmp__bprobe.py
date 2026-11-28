from calprobe import *
from scipy.optimize import minimize_scalar

z=calibration(s['panel_a'])[2](x)
p=s['panel_b']; u,v=p['u'],p['v'];

def evaluate(d,f):
 c=fit(d,v);e=v-d@c;q=np.linalg.qr(d,mode='reduced')[0]; loo=np.mean((e/(1-np.sum(q*q,axis=1)[:,None]))**2)**.5
 b=f@c; a,g,h,j=b.T; pair=a*h+g*j; con=a-g+h-j
 Ds=[aff(b),np.c_[aff(b),pair],np.c_[aff(b),a*h,g*j],poly(b,2),np.c_[aff(b),pair,np.sin(.2*con),np.tanh(con),pair/(1+con**2)]]
 scores=[]
 for D in Ds:
  pred=D[52:]@fit(D[:52],Y[:52]); scores.append(np.mean(abs(pred[:,0]-Y[52:,0])))
 e2=Ds[0][52:]@fit(Ds[0][:52],Y[:52,1:])-Y[52:,1:];return np.array([np.mean(e*e)**.5,loo,np.mean(abs(e2)),*scores])

families={
 'sin only':(lambda a,t: aff(np.sin(t*a)),(.2,5)),
 'linear sin':(lambda a,t: np.c_[aff(a),np.sin(t*a)],(.2,6)),
 'linear sincos':(lambda a,t: np.c_[aff(a),np.sin(t*a),np.cos(t*a)],(.2,6)),
 'linear tanh':(lambda a,t: np.c_[aff(a),np.tanh(t*a)],(.15,6)),
 'linear exp odd':(lambda a,t: np.c_[aff(a),a*np.exp(-t*a*a)],(.02,8)),
 'linear rational odd':(lambda a,t: np.c_[aff(a),a/(1+t*a*a)],(.02,8)),
 'linear sin x2':(lambda a,t: np.c_[aff(a),a*a,np.sin(t*a)],(.2,5)),
 'diag2 sincos':(lambda a,t: np.c_[aff(a),a*a,np.sin(t*a),np.cos(t*a)],(.2,5)),
 'diag3':(lambda a,t: np.c_[aff(a),a*a,a**3],(.2,5)),
}
print('header: family param panel RMS, panel LOO, state MAE; y MAE affine, paired-sum, paired-separate, degree 2, archived contrast')
for name,(fn,(a,b)) in families.items():
 def obj(t):
  d=fn(u,t);return np.mean((d@fit(d,v)-v)**2)
 grid=np.linspace(a,b,120); idx=np.argmin([obj(t) for t in grid]); t=minimize_scalar(obj,bounds=(grid[max(0,idx-1)],grid[min(len(grid)-1,idx+1)]),method='bounded').x
 print(name,round(t,5),evaluate(fn(u,t),fn(z,t)))
 if name in ('linear sin','linear sin x2','linear sincos'):
  print('coef',fit(fn(u,t),v).T)
# A rank-one nonlinear term per output can differ from nonlinearity before mixing.
for kind in ('sin','tanh','cubic','exp','rational'):
 def h(hh):
  if kind=='sin':return np.sin(hh)
  if kind=='tanh':return np.tanh(hh)
  if kind=='cubic':return hh**3
  if kind=='exp':return hh*np.exp(-hh**2)
  return hh/(1+hh**2)
 rng=np.random.default_rng(20); tr=[]; qz=[]; fits=[]
 for k in range(4):
  base=aff(u); bk=v[:,k]; lin=fit(base,bk)
  def fun(A,ds,xx):return np.c_[ds,h(aff(xx)@A)]
  def err(A):
   d=fun(A,base,u);return d@fit(d,bk)-bk
  sol=min((least_squares(err,A,max_nfev=180) for A in [*[np.r_[0,np.eye(4)[i]*f] for i in range(4) for f in (1.,2.5)],lin,*rng.normal(0,1,(6,5))]),key=lambda r:np.mean(r.fun**2));d=fun(sol.x,base,u);w=fit(d,bk);tr.append(d@w);qz.append(fun(sol.x,aff(z),z)@w);fits.append(sol.x)
 b=np.array(qz).T;D=aff(b); ds=[D,np.c_[D,b[:,0]*b[:,2]+b[:,1]*b[:,3]],poly(b,2)]
 print('projection',kind,'panel rms',np.mean((np.array(tr).T-v)**2)**.5,'52/20 state',np.mean(abs(D[52:]@fit(D[:52],Y[:52,1:])-Y[52:,1:])),'y',[np.mean(abs(d[52:]@fit(d[:52],Y[:52,0])-Y[52:,0])) for d in ds], 'params',np.round(fits,3))
