import numpy as np,pickle,itertools
from scipy.optimize import least_squares, minimize_scalar
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def fit(a,b): return np.linalg.lstsq(a,b,rcond=1e-10)[0]
def aff(x): return np.c_[np.ones(len(x)),x]
def poly(x,d): return np.column_stack([np.prod(x[:,i],axis=1) for j in range(d+1) for i in itertools.combinations_with_replacement(range(x.shape[1]),j)])
def metric(a,b):
 e=b-a@fit(a,b); h=(a*np.linalg.pinv(a).T).sum(1); return np.sqrt(np.mean(e**2)),np.sqrt(np.mean((e/(1-h[:,None]))**2))
F={'tanh':np.tanh,'atan':np.arctan,'sin':np.sin,'softsign':lambda x:x/np.sqrt(1+x*x),'erf':__import__('scipy').special.erf,'asinh':np.arcsinh}
def cal(x):
 p=S['panel_a']; a=aff(p['u']); v=p['v']; j=np.arange(len(a));
 for _ in range(5):
  w=fit(a[j],v[j]); j=np.argsort(np.linalg.norm(v-a@w,axis=1))[:45]
 return (x-w[0])/np.diag(w[1:])
if __name__=='__main__':
 np.set_printoptions(precision=5,suppress=True)
 for pn,vn in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
  p=S[pn]; u=p['u']; v=p[vn]; print(pn,vn)
  for name,f in F.items():
   if vn=='v2': continue
   for lin in [False,True]:
    def design(x,k,b=0):return np.c_[aff(x) if lin else np.ones(len(x)),f(x*k+b)]
    res=minimize_scalar(lambda k:metric(design(u,k),v)[1],bounds=(.2,3.5),method='bounded'); k=res.x
    d=design(u,k); print(name,lin,'shared',round(k,4),np.round(metric(d,v),6))
    if name in ['tanh','atan','softsign','erf']:
     def res(p):
      d=design(u,np.exp(p[:4]),p[4:]);return (d@fit(d,v)-v).ravel()
     z=least_squares(res,np.r_[np.full(4,np.log(k)),np.zeros(4)],bounds=([-2]*4+[-1.5]*4,[2]*4+[1.5]*4),max_nfev=130); d=design(u,np.exp(z.x[:4]),z.x[4:]); print(' free',np.exp(z.x[:4]),z.x[4:],np.round(metric(d,v),6))
     if metric(d,v)[0]<.002: print(fit(d,v))
  if vn=='v2':
   for d in [np.c_[aff(u),u*u],poly(u,2)]:
    print('quad',metric(d,v),'coef',fit(d,v))
    for link in ['tanh','sin','softsign']:
     fn=F[link]; w=fit(d,v); z=least_squares(lambda p:(fn(d@p[:-1].reshape(w.shape))*np.exp(p[-1])-v).ravel(),np.r_[w.ravel()/2,np.log(2.)],max_nfev=150); print(link,np.exp(z.x[-1]),np.std(z.fun))
