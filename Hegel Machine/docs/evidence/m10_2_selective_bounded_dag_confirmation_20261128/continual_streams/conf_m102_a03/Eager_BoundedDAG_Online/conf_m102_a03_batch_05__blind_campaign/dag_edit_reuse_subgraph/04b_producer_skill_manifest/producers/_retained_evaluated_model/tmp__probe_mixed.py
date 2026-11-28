exec(open('/tmp/probe_dag.py').read().split('for d in (1,2,3,4):')[0])
from scipy.optimize import least_squares, minimize_scalar
rng=np.random.default_rng(5); folds=[(np.arange(52),np.arange(52,72))]; folds += [(np.setdiff1d(np.arange(72),j),j) for j in np.array_split(rng.permutation(72),4)]
cr=list(it.combinations(range(4),2)); dg=[(j,j) for j in range(4)];
def cal(k):return n._fit(np.c_[n._affine(u),np.sin(k*u)],v)
k=minimize_scalar(lambda k: np.mean((np.c_[n._affine(u),np.sin(k*u)]@cal(k)-v)**2),bounds=(2,4),method='bounded').x; c=cal(k); z=np.c_[n._affine(x),np.sin(k*x)]@c; print('B sinusoid',k,np.round(c,4),np.sqrt(np.mean((np.c_[n._affine(u),np.sin(k*u)]@c-v)**2)))
warps={'sin':lambda x,k:np.sin(k*x),'tanh3':lambda x,k:np.tanh(k*x**3),'rat4':lambda x,k:x**3/(1+k*x**4),'rat2':lambda x,k:x/(1+k*x**2),'tanh':lambda x,k:np.tanh(k*x),'atan3':lambda x,k:np.arctan(k*x**3),'cos':lambda x,k:np.cos(k*x)}
results=[]
for name,h in warps.items():
 for cross in ('none','xx','zz','ww','xw','xxww','xz','xxx','xxsq','xwdiag'):
  for outer in ('lin','tanh','sin'):
   for share in (True,False):
    if not share and outer!='lin':continue
    def design(k):
     w=h(x,k); pair=lambda a,b:np.column_stack([a[:,i]*b[:,j] for i,j in cr]); d=[n._affine(x),w];
     if cross in ('xx','xxww','xxx','xxsq'):d.append(pair(x,x))
     if cross=='xxx':d.append(mat(x,list(it.combinations(range(4),3))))
     if cross=='xxsq':d.append(x*x)
     if cross in ('ww','xxww'):d.append(pair(w,w))
     if cross=='zz':d.append(pair(z,z))
     if cross in ('xw','xz'):
      b=w if cross=='xw' else z; d += [pair(x,b),pair(b,x)]
     if cross=='xwdiag':d += [pair(x,x),x*w]
     return np.column_stack(d)
    err=[]; tr=[]; params=[]
    for i,j in folds:
     def calc(g,ret=False):
      k=g[0] if share else g[:4]; A=design(k); yp=y[i]; f=lambda v:v
      if outer!='lin':
       cap=g[-1]*np.max(abs(yp)); inv,f0=(np.arctanh,np.tanh) if outer=='tanh' else (np.arcsin,np.sin); yp=inv(yp/cap); f=lambda v:cap*f0(v)
      coef=n._fit(A[i],yp); pred=f(A@coef); return (pred,coef) if ret else pred[i]-y[i]
     start={'sin':2.8,'tanh3':1.,'rat4':.3,'rat2':.7,'tanh':1.4,'atan3':.8,'cos':2.8}[name]; kk=min([least_squares(calc,[q]*(1 if share else 4)+([1.25] if outer!='lin' else []),bounds=([.03]*(1 if share else 4)+([1.001] if outer!='lin' else []),[7.]*(1 if share else 4)+([10.] if outer!='lin' else [])),max_nfev=60) for q in (start,1.6*start)],key=lambda o:o.cost); pred,coef=calc(kk.x,True); err.append(np.mean(abs(pred[j]-y[j]))); tr.append(np.mean(abs(pred[i]-y[i]))); params.append(kk.x)
    results.append((max(err),np.mean(err),name,cross,outer,share,np.round(err,4),np.round(tr,4),np.round(params[0],3)))
for r in sorted(results,key=lambda r:r[0])[:26]:print(r)
for r in sorted(results,key=lambda r:r[6][0])[:10]:print('public',r)
