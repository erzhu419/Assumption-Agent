import contextlib,io
with contextlib.redirect_stdout(io.StringIO()): exec(open('/tmp/dag_sink_probe.py').read())
from scipy.optimize import least_squares
np.set_printoptions(precision=5,suppress=True,linewidth=150)
L=stack(one(z),z); ZQ=stack(L,Q)
print('linear/pure quadratic fit coefficients by time (and y)',ls(ZQ,Y))
print('y vs final residual and observed',np.corrcoef((Y-ZQ@ls(ZQ,Y)).T)[0]); print('PCA residual singular values',np.linalg.svd(Y-ZQ@ls(ZQ,Y),compute_uv=False)); print('time fits',V[0]);
for name,fn in [('log',lambda k:np.log1p(k*t)),('exp',lambda k:-np.expm1(-k*t)),('pow',lambda k:t**k),('sqrt',lambda k:np.sqrt(1+k*t)-1),('asinh',lambda k:np.arcsinh(k*t))]:
 k=minimize_scalar(lambda k:fit(fn(k)[:,None],V[0]),bounds=(.02,9),method='bounded').x; print(name,k,fit(fn(k)[:,None],V[0]),fit(stack(fn(k),np.ones_like(t)),V[0]))
q0=(z*z)@np.array([1,1,-1,-1]); lin0=z@np.array([-1,-1,1,1]);
for yy in [y,y-tr[:,-1],tr[:,-1]]:
 print('baseline residual',np.std(yy-ZQ@ls(ZQ,yy)),ls(ZQ,yy))
for nm,base in [('Q',ZQ),('ABQ',stack(ZQ,A)),('Q+A*B',stack(ZQ,A*B)),('Q+A2',stack(ZQ,A*A)),('Q+B2',stack(ZQ,B*B)),('Q+A2+B2',stack(ZQ,A*A,B*B)),('Q+Az',stack(ZQ,A*z)),('Q+Bz',stack(ZQ,B*z)),('Q+tanh',stack(ZQ,np.tanh(1.7*z))),('Q+sin',stack(ZQ,np.sin(3*z))),('Q+A*B+Az+Bz',stack(ZQ,A*B,A*z,B*z))]:
 for alpha in [0,0.003]:
  sc=np.maximum(base[:52].std(0),.01);sc[0]=1; D=base/sc; G=D[:52].T@D[:52]; P=np.diag(np.r_[0,np.repeat(alpha*52,D.shape[1]-1)]); C=np.linalg.lstsq(G+P,D[:52].T@Y[:52],rcond=1e-11)[0]; e=np.abs(D[52:]@C-Y[52:]); print(nm,alpha,e[:,0].mean(),e[:,1:].mean())
print('Single-index nonlinear corrections: quadratic plus smooth or switch ridge in z/A/B/AB')
from scipy.linalg import qr
rng=np.random.default_rng(193)
for regime in ['tanh','sine','relu','rational','softplus']:
 f={'tanh':np.tanh,'sine':np.sin,'relu':lambda x:np.maximum(x,0),'rational':lambda x:1/(1+x*x),'softplus':lambda x:np.logaddexp(0,x)}[regime]
 for name,coords in [('z',z),('A',A),('B',B),('AB',stack(A,B))]:
  ds=ZQ[:52]; qq=np.linalg.qr(ds)[0]; targets=Y[:52,[0,-1]]; r=targets-qq@(qq.T@targets); best=None
  for j in range(20):
   def fun(p):
    h=f(p[0]+coords[:52]@p[1:]); h-=qq@(qq.T@h); return (h[:,None]*((h@r)/max(h@h,1e-19))-r).ravel()
   o=least_squares(fun,rng.normal(0,1,coords.shape[1]+1),max_nfev=90,bounds=(-15,15)); score=np.mean(o.fun**2)
   if best is None or score<best[0]:best=score,o.x
  d=stack(ZQ,f(best[1][0]+coords@best[1][1:])); e=np.abs(d[52:]@ls(d[:52],Y[:52])-Y[52:]); print(regime,name,round(best[0],6),round(e[:,0].mean(),5),round(e[:,1:].mean(),5),best[1])
