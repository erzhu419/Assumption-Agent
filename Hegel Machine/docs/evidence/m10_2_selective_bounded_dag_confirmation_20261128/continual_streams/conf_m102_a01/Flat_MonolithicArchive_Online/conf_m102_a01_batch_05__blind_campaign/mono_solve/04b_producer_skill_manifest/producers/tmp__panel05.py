exec(open('/tmp/probe05.py').read().split("print('b singular values'")[0]+"\n")
from scipy.optimize import minimize_scalar
p=s['panel_a']; print('A gain/bias',np.linalg.lstsq(A(z),x,rcond=None)[0]); r=p['v']-A(p['u'])@np.linalg.lstsq(A(z),x,rcond=None)[0]; print('A nonoutlier residual spread',np.sort(np.max(abs(r),axis=1))[:40]);
p=s['panel_c']; u=p['u']; print('B sv',np.linalg.svd(b[1:],compute_uv=False))
funcs={'softsign':lambda w,a:w/(1+a*abs(w)),'softabs':lambda w,a:np.sqrt(1+a*w*w),'tanh':lambda w,a:np.tanh(a*w),'sin':lambda w,a:np.sin(a*w),'cos':lambda w,a:np.cos(a*w),'atan':lambda w,a:np.arctan(a*w),'asinh':lambda w,a:np.arcsinh(a*w),'exp':lambda w,a:np.exp(a*w),'quad':lambda w,a:w*w,'sat':lambda w,a:w/np.sqrt(1+a*w*w),'logcosh':lambda w,a:np.log(np.cosh(a*w))}
for key in ['v1','v2']:
 v=p[key]; res=[]
 for name,f in funcs.items():
  for mode in ('pure','resid'):
   def err(a,ret=False):
    F=np.c_[np.ones(len(u)),f(u,a)] if mode=='pure' else np.c_[A(u),f(u,a)]; c=np.linalg.lstsq(F,v,rcond=None)[0]; return c if ret else np.std(v-F@c)
   o=minimize_scalar(err,bounds=(.04,4),method='bounded'); res.append((o.fun,name,mode,o.x,err(o.x,True)))
 for e,k,m,a,c in sorted(res,key=lambda q:q[0])[:6]:print(key,e,k,m,a,np.round(c,3).tolist())
