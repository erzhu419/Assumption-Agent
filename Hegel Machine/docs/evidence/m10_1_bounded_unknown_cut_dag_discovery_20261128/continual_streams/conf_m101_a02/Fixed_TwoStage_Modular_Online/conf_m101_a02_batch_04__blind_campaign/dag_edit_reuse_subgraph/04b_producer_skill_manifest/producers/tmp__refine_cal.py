import numpy as np,pickle,itertools
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; p=s['panel_a']; x=p['u']; y=p['v']; n=len(x); one=np.ones(n); base=np.c_[one,x,x*np.roll(x,-1,axis=1)]
fit=lambda a,b: np.linalg.lstsq(a,b,rcond=None)[0]
def loo(a,b):
 q=np.linalg.qr(a)[0]; z=b-q@(q.T@b); h=1-(q*q).sum(1); return z/h.reshape((-1,)+(1,)*(z.ndim-1))
def rm(z): return np.mean(z*z)**.5
w=np.ones(n)
for _ in range(30):
 e=np.linalg.norm(y-base@fit(base*w[:,None],y*w[:,None]),axis=1);w=1/np.hypot(1,e/max(np.median(e),1e-8))
g=np.argsort(e)[:-3]; print('A good baseline',rm(loo(base[g],y[g])), 'coeffs',np.round(fit(base[g],y[g]),3))
lib={}
for d in [2,3,4]:
 for ix in itertools.combinations_with_replacement(range(4),d): lib[('poly',ix)]=np.prod(x[:,ix],axis=1)
for j in range(4):
 a=x[:,j]; b=x[:,(j+1)%4]; c=x[:,(j+2)%4]
 for k in [.4,.7,1,1.4,2,3]:
  for name,z in [('sin',np.sin(k*a)),('tanh',np.tanh(k*a)),('sinpair',np.sin(k*a*b)),('sinplus',np.sin(k*(a+b))),('sinminus',np.sin(k*(a-b))),('sinopp',np.sin(k*(a-c))),('cospair',np.cos(k*a*b)),('prodcos',a*np.cos(k*b)),('ratprod',a*b/(1+k*(a*b)**2)),('ratx',a/(1+k*a*a))]: lib[(name,j,k)]=z
for j in range(4):
 a,b=x[:,j],x[:,(j+1)%4]; d=np.c_[one,a,b,a*b]; scores=[]
 for k,z in lib.items():
  h=np.c_[d,z][g];
  if np.linalg.matrix_rank(h)==5:scores.append((rm(loo(h,y[g,j])),k,float(fit(h,y[g,j])[-1])))
 print('A output',j,'sparse baseline',rm(loo(d[g],y[g,j])),'single additions',sorted(scores)[:8]);
 print('A row correlations',np.round((y[g,j]-d[g]@fit(d[g],y[g,j]))@x[g]**3,3))
print('A grouped additions')
for name in ['sin','tanh','sinpair','sinplus','sinminus','sinopp','cospair','prodcos','ratprod','ratx']:
 print(name,min((rm(loo(np.c_[base,*[lib[(name,j,k)] for j in range(4)]][g],y[g])),k) for k in [.4,.7,1,1.4,2,3]))
for d in [3,4]:
 print('power',d,rm(loo(np.c_[base,x**d][g],y[g])))
p=s['panel_b']; x=p['u']; y=p['v']; one=np.ones(len(x));
forms={'ratcube':lambda k:x**3/(1+k*x*x),'expcube':lambda k:x*(-np.expm1(-k*x*x)),'tanhcube':lambda k:x*np.tanh(k*x*x),'sinlin':lambda k:x-np.sin(k*x)/k,'tanhlin':lambda k:x-np.tanh(k*x)/k,'softcube':lambda k:x**3/(1+k*abs(x)),'sincube':lambda k:x*np.sin(k*x*x),'logcube':lambda k:x*np.log1p(k*x*x),'sqrtcube':lambda k:x*(1-1/np.sqrt(1+k*x*x))}
for name,form in forms.items():
 for mode in ('aff','tie','single'):
  def design(k):
   z=form(k); return np.c_[one,x,z] if mode=='aff' else np.c_[one,z+(k*0 if mode=='single' else .05)*x]
  op=minimize_scalar(lambda k:rm(loo(design(k),y)),bounds=(.015,4),method='bounded'); print('B',name,mode,round(op.x,4),round(op.fun,6))
print('B best common linear ratio models')
from scipy.optimize import least_squares
for name,form in forms.items():
 def ds(params):k,c=params;return np.c_[one,form(k)+c*x]
 op=least_squares(lambda t:loo(ds(t),y).ravel(),[.7,.05],bounds=([.015,-.5],[5,1.])); print(name,op.x,rm(op.fun))
