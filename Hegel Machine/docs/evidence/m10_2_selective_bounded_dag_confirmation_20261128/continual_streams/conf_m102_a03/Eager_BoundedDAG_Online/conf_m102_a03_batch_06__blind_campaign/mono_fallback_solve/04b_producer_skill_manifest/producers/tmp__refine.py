exec(open('/tmp/latent.py').read().split("print('cal'")[0])
from scipy.optimize import minimize_scalar
from sklearn.linear_model import Ridge
from scipy.special import expit
from itertools import product

def LOO(A,Y):
 Q=np.linalg.qr(A)[0];return np.mean(((Y-Q@(Q.T@Y))/(1-(Q*Q).sum(1))[:,None])**2)**.5

def activate(name,u,k):
 if name=='rat':return u/(1+k*u*u)
 if name=='ratio':return u/np.sqrt(1+k*u*u)
 if name=='soft':return np.sqrt(1+k*u*u)
 if name=='asinh':return np.arcsinh(k*u)
 if name=='atan':return np.arctan(k*u)
 if name=='tanh':return np.tanh(k*u)
 if name=='sin':return np.sin(k*u)
 if name=='cos':return np.cos(k*u)
 if name=='exp':return np.exp(-k*u*u)

def op(p,key):
 U=p['u'];V=p[key];out=[]
 for kind,lin in product(['rat','ratio','soft','asinh','atan','tanh','sin','cos','exp'],[False,True]):
  def design(u,k):return np.c_[aff(u),activate(kind,u,k)] if lin else aff(activate(kind,u,k))
  def get(k):return LOO(design(U,k),V)
  opt=minimize_scalar(get,bounds=(.2,3.5),method='bounded');out.append((opt.fun,kind,lin,opt.x))
 print(key,sorted(out)[:9]);_,kind,lin,k=min(out);fn=lambda u:np.c_[aff(u),activate(kind,u,k)] if lin else aff(activate(kind,u,k));coef=ls(fn(U),V);return lambda u:fn(u)@coef
ops={n:op(s[p],k) for n,p,k in [('b','panel_b','v'),('p','panel_c','v1'),('q','panel_c','v2')]}
# Check cyclic calibration residual for a shared extra term and geometry.
A=np.stack([np.ones_like(u),u,np.roll(u,-1,1),u*np.roll(u,-1,1)],axis=-1);E=v[g]-np.einsum('ijk,kj->ij',A[g],c)
extra=[]
for k,l in product(range(4),repeat=2):
 a0,b0=np.roll(u,-k,1),np.roll(u,-l,1)
 for nm,a in [('single',a0),('pair',a0*b0),('sum',a0+b0),('diff',a0-b0),('cubic',a0*a0*b0)]:
  for k0 in [.5,1,1.25,2,3]:
   for f0 in ['rat','ratio','soft','atan','tanh','sin','cos','exp']:
    D=activate(f0,a,k0)
    vv=np.array([LOO(np.c_[A[g,j],D[g,j]],v[g,j:j+1]) for j in range(4)])
    extra.append((np.mean(vv**2)**.5,k,l,nm,k0,f0))
print('A extra',sorted(extra)[:12]);print('A residual SV',np.linalg.svd(E,compute_uv=False))
I=np.arange(52);J=np.arange(52,72);_,_,V=np.linalg.svd(t[I],full_matrices=False);Y=np.c_[y,t@V[0]]; Z={'z':z,'x':x}
for n,f in ops.items():
 for k in ['z','x']: Z[n if k=='z' else n+k]=f(Z[k])
for n,f in ops.items():
 for k in ['b','p','q']:Z[n+k]=f(Z[k])
print('tomography')
def score(A,Y,I,J,alpha):
 m=A[I].mean(0);d=A[I].std(0)+1e-9;A=(A-m)/d; M=A[I]; yy=Y[I];B=np.linalg.solve(M@M.T+alpha*np.eye(len(I)),yy-yy.mean(0));return np.mean(abs(A[J]@M.T@B+yy.mean(0)-Y[J]),0)*[1,.325]
best=[[],[]]
for names in [(k,) for k in Z]+list(itertools.combinations(Z,2))+[('z','b','p'),('z','p','q'),('z','b','q'),('b','p','q'),('z','b','p','q')]:
 U=np.concatenate([Z[k] for k in names],1)
 for typ in ['linear','quadratic','separable','mixed']:
  if typ=='linear': D=U
  if typ=='quadratic': D=PolynomialFeatures(2,include_bias=False).fit_transform(U)
  if typ=='separable': D=np.c_[U,np.tanh(1.25*U),np.sqrt(1+U*U)]
  if typ=='mixed': D=PolynomialFeatures(2,include_bias=False).fit_transform(np.c_[U,np.tanh(1.25*U),np.sqrt(1+U*U)])
  for a in [1e-5,.0003,.003,.03,.3,3]:
   res=score(D,Y,I,J,a)
   for j in range(2):best[j].append((res[j],names,typ,a))
for zbest in best: print(*sorted(zbest)[:15],sep='\n')
