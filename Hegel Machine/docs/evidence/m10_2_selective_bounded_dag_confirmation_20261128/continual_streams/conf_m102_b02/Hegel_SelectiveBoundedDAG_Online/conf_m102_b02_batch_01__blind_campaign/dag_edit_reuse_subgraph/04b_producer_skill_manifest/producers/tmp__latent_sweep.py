import contextlib,io
with contextlib.redirect_stdout(io.StringIO()): exec(open('/tmp/compare_maps.py').read())
from scipy.optimize import minimize_scalar
U,V=b['u'],b['v']; u=base['deA'];
opts={'tanh':np.tanh,'sin':np.sin,'atan':np.arctan,'asinh':np.arcsinh,'rational':lambda u:u/(1+u*u)}
res=[]
for name,f in opts.items():
 def fit(k):
  F=np.c_[np.ones(len(U)),U,f(k*U)]; cf=np.linalg.lstsq(F,V,rcond=None)[0];return np.mean((F@cf-V)**2),cf
 ks=np.linspace(.1,5,60); j=np.argmin([fit(k)[0] for k in ks]); k=minimize_scalar(lambda k:fit(k)[0],bounds=(ks[max(0,j-1)],ks[min(j+1,len(ks)-1)]),method='bounded').x
 _,cf=fit(k); A=cf[1:5]; B=cf[5:]; print(name,k,'ratio',np.sum(A*B,axis=1)/np.sum(B*B,axis=1),'svA',np.linalg.svd(A,compute_uv=False), 'bias',cf[0]);
 for desc,M in [('A',A),('B',B),('OLS',make(U,V)[1][1:]),('invA',np.linalg.inv(A)),('invB',np.linalg.inv(B))]:
  M=M/np.linalg.norm(M,axis=0); P=np.linalg.svd(M); O=P[0]@P[2]
  for matname,L in [('raw',M),('ortho',O)]:
   for t in [-2,-1,-.5,0,.5,1,2]:
    for bias in [0,1]:
     for sc in [.15,.3,.55,.85,1.25,1.8]:
      z=(u@(L)+t*f(k*u)@L+bias*cf[0])*sc; e=score(z); res.append((max(e),e.tolist(),name,desc,matname,t,bias,sc))
for axis in [('raw',u),('tanh',np.tanh(1.25*u)),('sinh',np.sinh(1.25*u)),('cubic',u**3)]:
 for sc in [.1,.2,.4,.7,1.0,1.4]:
  e=score(axis[1]*sc);res.append((max(e),e.tolist(),axis[0],sc))
print('best',*sorted(res)[:20],sep='\n')
