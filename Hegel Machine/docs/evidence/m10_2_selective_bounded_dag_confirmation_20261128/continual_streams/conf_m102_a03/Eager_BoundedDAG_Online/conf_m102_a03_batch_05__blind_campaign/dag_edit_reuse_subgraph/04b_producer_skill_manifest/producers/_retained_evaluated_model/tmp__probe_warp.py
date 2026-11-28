exec(open('/tmp/probe_dag.py').read().split('for d in (1,2,3,4):')[0])
from scipy.optimize import minimize_scalar
cr=list(it.combinations(range(4),2)); sq=[(j,j) for j in range(4)]; ix=np.arange(52); jj=np.arange(52,72)
warp={'cube':lambda x,k:x**3,'sin_cube':lambda x,k:np.sin(k*x**3),'tanh_cube':lambda x,k:np.tanh(k*x**3),'atan_cube':lambda x,k:np.arctan(k*x**3),'rat2':lambda x,k:x**3/(1+k*x**2),'rat4':lambda x,k:x**3/(1+k*x**4),'rat6':lambda x,k:x**3/(1+k*x**6),'pow':lambda x,k:np.sign(x)*abs(x)**k,'sin':lambda x,k:np.sin(k*x)}
def probe(X,Y,i,j):
 out=[]
 for key,fn in warp.items():
  for mode in ('add','cross','wcross','both','sqcross','no_raw'):
   def feat(k):
    w=fn(X,k); fs=[np.ones((len(X),1)),w];
    if mode!='no_raw':fs.append(X)
    if mode in ('cross','sqcross','both'):fs.append(mat(X,cr+(sq if mode=='sqcross' else [])))
    if mode in ('wcross','both'):fs.append(mat(w,cr))
    return np.column_stack(fs)
   def fit(k, ret=False):
    A=feat(k); c=n._fit(A[i],Y[i]); r=A@c-Y
    return (np.mean(abs(r[j])),np.sqrt(np.mean(r[i]**2)),k,c) if ret else np.mean(r[i]**2)
   grid=np.geomspace(.02,6,36); m=np.argmin([fit(k) for k in grid]); k=grid[m]
   if key!='cube' and 0<m<len(grid)-1:k=minimize_scalar(fit,bounds=(grid[m-1],grid[m+1]),method='bounded').x
   e,train,k,c=fit(k,True);out.append((e,train,key,mode,k,c))
 for e,train,key,mode,k,c in sorted(out,key=lambda r:r[0])[:12]:
  print('val/fit %.5f %.5f %s %s %.4f'%(e,train,key,mode,k), np.round(c,3) if c.ndim==1 else '')
 return min(out,key=lambda r:r[1])
print('B'); probe(u,v,np.arange(18),np.arange(18,24)); print('scalar'); probe(x,y,ix,jj); print('amp'); probe(x,yy,ix,jj)
print('Outer links & panel prediction');
for h in ('raw','tanh','sin','asinh'):
 for k in [1.01,1.15,1.5,2,3]:
  if h=='raw' and k!=1.01:continue
  ft,inv={'raw':(lambda x:x,lambda x:x), 'tanh':(lambda x:np.arctanh(x/k),lambda x:k*np.tanh(x)), 'sin':(lambda x:np.arcsin(x/k),lambda x:k*np.sin(x)), 'asinh':(lambda x:np.arcsinh(x/k),lambda x:k*np.sinh(x))}[h]; k*=np.max(abs(y[ix]));
  for bname,bf in [('cube',lambda x:np.c_[n._affine(x),x**3]),('satcube',lambda x:np.c_[n._affine(x),np.tanh(x**3)]),('sin',lambda x:np.c_[n._affine(x),np.sin(3*x)])]:
   A=np.c_[bf(x),mat(x,cr)]; c=n._fit(A[ix],ft(y[ix])); r=inv(A@c)-y; print(h, round(k,2),bname,np.round([np.mean(abs(r[ix])),np.mean(abs(r[jj]))],5))
