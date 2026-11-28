exec(open('/tmp/cubic.py').read().split('def cross')[0])
from scipy.optimize import minimize_scalar
from sklearn.model_selection import KFold
f=B(Ai(x)); p=s['panel_c']; u=p['u']
def odd(z,a):return -np.sign(z)*np.expm1(-a*abs(z))
def even(z,a):return z*z/np.sqrt(1+a*z*z)
def op(v,fun,linear,approx):
 def g(z,a):return design(*([z] if linear else []),fun(z,a))
 def loss(a):
  h=g(u,a);return np.linalg.norm(h@ls(h,v)-v)
 o=minimize_scalar(loss,bounds=(approx*.5,approx*1.5),method='bounded'); a=o.x; w=ls(g(u,a),v);print('panel',a,o.fun,'coef',np.round(w,4));return g(f,a)@w,fun(f,a)
c,o=op(p['v1'],odd,False,1.8); d,e=op(p['v2'],even,True,.389)
def interaction(a,b): return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
def quad(a): i,j=np.triu_indices(a.shape[1]);return a[:,i]*a[:,j]
tr=np.arange(52);te=np.arange(52,72); folds=[(tr,te),*KFold(4,shuffle=True,random_state=123).split(x)]
base=design(f,o,e); feat={'base':base,'all':design(f,o,e,quad(f),quad(o),quad(e),interaction(f,o),interaction(f,e),interaction(o,e)),'samedim':design(f,o,e,f*f,o*o,e*e,f*o,f*e,o*e),'dictionary':design(f,o,e,quad(f),quad(c),quad(d),interaction(f,o),interaction(f,e),interaction(o,e),np.sin(f),np.cos(f),np.tanh(f),np.sin(c),np.cos(c),np.sin(d),np.cos(d))}
for k in (-1,1):
 for a in (.4,.8,1.2,1.6):
  q=np.roll(f,k,axis=1)*a; feat[f'pair{k},{a}']=np.c_[base,odd(f+q,1.8),odd(f-q,1.8),even(f+q,.389),even(f-q,.389),f*q]
res=[]
for name,h in feat.items():
 for mode in ('raw','atanh','asinh'):
  for scale in ([1] if mode=='raw' else ([1.25,1.5,2] if mode=='atanh' else [.4,.8,1.5])):
   Y=T.copy(); pred=lambda z:z
   if mode=='atanh':Y[:,0]=np.arctanh(Y[:,0]/scale);pred=lambda z:scale*np.tanh(z)
   if mode=='asinh':Y[:,0]=np.arcsinh(Y[:,0]/scale);pred=lambda z:scale*np.sinh(np.clip(z,-6,6))
   for l in (1e-7,1e-5,1e-4,.001,.01):
    errors=[]
    for i,j in folds:
     a=h[i]; norms=np.maximum(np.linalg.norm(a,axis=0),1e-12);a=a/norms;b=h[j]/norms;w=np.linalg.solve(a.T@a+l*np.diag(np.r_[0,np.ones(a.shape[1]-1)]),a.T@Y[i]); val=b@w;val[:,0]=pred(val[:,0]);delta=abs(val-T[j]);errors.append([delta[:,0].mean(),delta[:,1:].mean()])
    err=np.array(errors);res.append((name,mode,scale,l,*err[0],*err[1:].mean(0),*err[1:].max(0)))
for m in (4,5,6,7):
 print('LEADER',m); print(*sorted(res,key=lambda z:z[m])[:9],sep='\n')
# An eigenvalue fit separates shared temporal profiles from input nonlinearities.
tt=s['eval_times']; from scipy.optimize import least_squares
for name,ff in {'twoexp': lambda a:np.array([-np.expm1(-tt*a[0]),-np.expm1(-tt*a[1])]),'exp_linear':lambda a:np.array([-np.expm1(-tt*a[0]),tt]),'exp_t_exp':lambda a:np.array([-np.expm1(-tt*a[0]),tt*np.exp(-tt*a[1])]),'exp_log':lambda a:np.array([-np.expm1(-tt*a[0]),np.log1p(tt*a[1])]),'log_linear':lambda a:np.array([np.log1p(tt*a[0]),tt])}.items():
 def err(a):h=ff(a);return (ls(h.T,t.T).T@h-t).ravel()
 opt=least_squares(err,[1.,.3],bounds=(.001,15));print('time',name,opt.x,np.mean(abs(opt.fun)))
np.savez('/tmp/latent.npz',f=f,c=c,d=d,o=o,e=e,x=x,T=T,times=tt)
