exec(open('/tmp/latent.py').read().split("print('cal'")[0])
from scipy.optimize import least_squares
from scipy.special import expit
I=np.arange(52); J=np.arange(52,72)
func={'linear':lambda x:x,'tanh':np.tanh,'sin':np.sin,'rat':lambda x:x/(1+x*x),'cube':lambda x:x**3,'abs':np.abs,'relu':lambda x:np.maximum(x,0),'atan':np.arctan}
def loo(A,Y):
 Q=np.linalg.qr(A)[0]; return np.mean(((Y-Q@(Q.T@Y))/np.maximum(1e-6,1-(Q**2).sum(1))[:,None])**2)**.5
ops={}
for pn,key in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 p=s[pn];u=p['u'];Y=p[key]; opts=[]
 for name,fn in func.items():
  for k in [.5,1,1.25,1.5,2,2.5,3.14]:
   for form in ['add','pure','post']:
    if form=='post':
     if name in ('abs','relu','cube','linear'):continue
     if np.max(abs(Y))>=k and name in ('sin','tanh'):continue
     invf={'sin':np.arcsin,'tanh':np.arctanh,'atan':np.tan}.get(name)
     if invf is None:continue
     if name=='atan' and np.max(abs(Y/k))>=1.4:continue
     A=aff(u);C=ls(A,invf(Y/k)); R=k*fn(A@C)-Y;err=np.mean((R/(1-np.sum(np.linalg.qr(A)[0]**2,1))[:,None])**2)**.5;Z=k*fn(aff(z)@C)
    else:
     D=fn(k*u);A=aff(D) if form=='pure' else np.c_[aff(u),D];
     if np.linalg.matrix_rank(A)<A.shape[1]:continue
     err=loo(A,Y); C=ls(A,Y); Z=(aff(fn(k*z)) if form=='pure' else np.c_[aff(z),fn(k*z)])@C
    opts.append((err,name,k,form,Z))
 # affine + nonlinear per input, choose by multioutput LOO
 print(pn,key,'aff',loo(aff(u),Y),'best',[(round(a,5),b,c,d) for a,b,c,d,_ in sorted(opts,key=lambda p:p[0])[:10]])
 ops[key+pn[-1]]=sorted(opts,key=lambda p:p[0])[:3]
 for i in range(4):
  Oo=[]
  for name,fn in func.items():
   for k in [.75,1.25,2,3.14]:
    for j in range(4):
     a=np.c_[aff(u),fn(k*u[:,j])];
     if np.linalg.matrix_rank(a)==a.shape[1]:Oo.append((loo(a,Y[:,i:i+1]),name,k,j))
  print('column',i,sorted(Oo)[:4])
T=t@np.linalg.svd(t[:52],full_matrices=False)[2][0]; T/=np.std(T[:52]);rng=np.random.default_rng(456)
def test(X,Y,kind,deg,H,ntry=16):
 A=PolynomialFeatures(deg).fit_transform(X); B=aff(X); A0=A[I];B0=B[I];Y0=Y[I]; opts=[]
 def F(B,W):
  v=B@W; return np.abs(v) if kind=='abs' else np.tanh(v) if kind=='tanh' else expit(v)
 for _ in range(ntry):
  def calc(p,return_pred=False):
   W=p.reshape(B.shape[1],H); G=F(B0,W);
   D=np.c_[A0,G] if kind!='switch' else np.c_[A0,A0[:,1:]*G,G]; c0=ls(D,Y0); R=D@c0-Y0; 
   if return_pred:
    G=F(B,W); return (np.c_[A,G] if kind!='switch' else np.c_[A,A[:,1:]*G,G])@c0
   return np.r_[R,.002*p]
  p=least_squares(calc,rng.normal(size=B.shape[1]*H),max_nfev=180).x; pred=calc(p,True); opts.append((np.mean(abs(pred[I]-Y[I])),np.mean(abs(pred[J]-Y[J])),p))
 a,b,p=min(opts,key=lambda x:x[0]);return np.round([a,b],4),np.round(p,2)
for Xname,X in [('z',z),('x',x)]:
 for Yname,Y in [('y',y),('T',T)]:
  for kind,deg,H in [('abs',1,2),('abs',2,1),('abs',1,3),('tanh',1,2),('tanh',2,1),('switch',1,1),('switch',2,1)]:
   print(Xname,Yname,kind,deg,H,*test(X,Y,kind,deg,H),flush=True)
