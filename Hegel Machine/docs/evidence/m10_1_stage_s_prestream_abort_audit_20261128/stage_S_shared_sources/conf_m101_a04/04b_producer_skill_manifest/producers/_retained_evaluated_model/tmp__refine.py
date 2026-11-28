import contextlib,io
with contextlib.redirect_stdout(io.StringIO()): exec(open('/tmp/links.py').read())
from itertools import product
from scipy.special import erf
from sklearn.linear_model import Ridge
from scipy.linalg import orthogonal_procrustes

def af(X):return np.c_[np.ones(len(X)),X]
base={'tanh':np.tanh,'atan':np.arctan,'erf':erf,'asinh':np.arcsinh,'softsign':lambda x:x/np.sqrt(1+x*x),'sin':np.sin,'rational':lambda x:x/(1+x*x),'algebraic':lambda x:x/np.cbrt(1+abs(x)**3),'log':lambda x:np.sign(x)*np.log1p(abs(x))}
results=[]
for nm,f in base.items():
 for mode in ('pure','lin','cub','warp','mixlin','mixsin'):
  def F(W,p,nm=nm,f=f,mode=mode):
   k=p[0]; c=(p[1] if len(p)>1 else 0); return f(k*W+c*W**3) if mode=='warp' else f(k*W)+({'pure':0,'lin':c*W,'cub':c*W**3,'mixlin':c*W+p[2]*W**3,'mixsin':c*np.sin(p[2]*W)}[mode] if mode.startswith('mix') else {'pure':0,'lin':c*W,'cub':c*W**3}[mode])
  def opt(idx):
   def res(p):
    E=af(F(U[idx],p));return (E@fit(E,R[idx])-R[idx]).ravel()
   n=1 if mode=='pure' else 3 if mode.startswith('mix') else 2
   o=min((least_squares(res,[k,0,1.8][:n],bounds=([.15,-12,.15][:n],[9,12,9][:n]),max_nfev=160) for k in (1,2,3.5)),key=lambda a:np.linalg.norm(a.fun));return o.x,fit(af(F(U[idx],o.x)),R[idx])
  pp=np.empty_like(R)
  for i,j in KFold(4,shuffle=True,random_state=61).split(U):
   p,w=opt(i); pp[j]=af(F(U[j],p))@w
  p,w=opt(np.arange(len(U))); results.append((np.mean(abs(pp-R)),np.std(af(F(U,p))@w-R),nm,mode,p,w,F))
J=af(np.logaddexp(0,2.2*z))@fit(af(np.logaddexp(0,2.2*U)),S)
for a in sorted(results,key=lambda a:a[0])[:12]:print('C1',a[:5])
models=[(str((a[2],a[3])), af(a[-1](z,a[4]))@a[-2]) for a in sorted(results,key=lambda a:a[0])[:6]]+[('asinh coord',af(np.c_[z,np.arcsinh(z*ops2[0][2][:4]+ops2[0][2][4:])])@ops2[0][-1])]
for name,H in models:
 cross=np.einsum('ni,nj->nij',H,J).reshape(len(z),-1); W=np.c_[H,J]; full=poly(W,2)[:,1:]
 for mode,E in [('cross',np.c_[W,cross]),('full',full),('cross+J2',np.c_[H,poly(J,2)[:,1:],cross]),('cross+H2',np.c_[J,poly(H,2)[:,1:],cross])]:
  p=np.empty_like(Y)
  for a,c in folds:
   scale=E[a].std(0); p[c]=Ridge(alpha=.001).fit(E[a]/scale,Y[a]).predict(E[c]/scale)
  print('CV',name,mode,np.mean(abs(p[:,2]-y)),np.mean(abs(p[:,:2]@D.T-T),0).max())
 if name==models[0][0]:
  E=af(np.c_[W,cross]); coef=fit(E,Y); print('cross coefficients',coef,'train residual',np.std(E@coef-Y,0));print('C1 full mixing',sorted(results,key=lambda a:a[0])[0][-2]); print('C2 coeff',fit(af(np.logaddexp(0,2.2*U)),S))
