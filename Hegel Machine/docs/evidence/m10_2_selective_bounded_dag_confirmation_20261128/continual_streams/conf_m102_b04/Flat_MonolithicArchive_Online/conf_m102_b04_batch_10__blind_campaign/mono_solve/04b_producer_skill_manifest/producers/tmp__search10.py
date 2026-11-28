import numpy as np,pickle
from scipy.optimize import minimize_scalar
from itertools import combinations_with_replacement,product
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; X=S['train']['x']; Y=np.c_[S['train']['y'],S['train']['traj']]; n=52
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
def a(*x):return np.column_stack((np.ones(len(x[0])),*x))
forms={'sin':lambda x,k:np.sin(k*x),'tanh':lambda x,k:np.tanh(k*x),'rat':lambda x,k:x/(1+k*x*x),'rat3':lambda x,k:x**3/(1+k*x*x),'atan':lambda x,k:np.arctan(k*x),'asinh':lambda x,k:np.arcsinh(k*x),'sq':lambda x,k:x*x}
def panel(P,key,forms_ ,rob=False):
 u,v=P['u'],P[key]; m=np.linalg.norm(v-u,axis=1)<4*np.median(np.linalg.norm(v-u,axis=1)) if rob else np.ones(len(u),bool); u,v=u[m],v[m]; opts=[]
 for name in forms_:
  for sq in ([False] if name=='sq' else [False,True]):
   def design(x,k,name=name,sq=sq):return a(x,forms[name](x,k),*([x*x] if sq else []))
   def f(k,ret=False):
    d=design(u,k); b=ls(d,v); e=d@b-v; lev=(np.linalg.qr(d,mode='reduced')[0]**2).sum(1); return b if ret else np.mean((e/(1-lev[:,None]))**2)
   r=minimize_scalar(f,bounds=(.05,4),method='bounded'); k=r.x; b=f(k,True); opts.append((r.fun,name,sq,k,(lambda x,k=k,b=b,d=design:d(x,k)@b)))
 opts.sort(key=lambda x:x[0]); print(key,len(u),[(round(np.sqrt(z[0]),5),*z[1:4]) for z in opts[:7]]); return opts
A=panel(S['panel_a'],'v',forms,True); B=panel(S['panel_b'],'v',forms); C=panel(S['panel_c'],'v1',forms); D=panel(S['panel_c'],'v2',forms)
from scipy.optimize import root
F=A[0][-1]; Z=np.array([root(lambda p:F(p[None])[0]-x,x,tol=1e-10).x for x in X]); print('inverse',np.max(abs(F(Z)-X)));print('times',S['eval_times'],'sv',np.linalg.svd(Y[:,1:],compute_uv=False));
ops={'z':lambda x:x,'b':B[0][-1],'c':C[0][-1],'d':D[0][-1]}; feats={'x':X,'z':Z}
for l in range(1,4):
 for k in product('bcd',repeat=l):
  z=Z
  for j in k:z=ops[j](z)
  feats[''.join(k)]=z

def p2(z):return np.column_stack([z[:,i]*z[:,j] for i,j in combinations_with_replacement(range(4),2)])
def cv(f):
 e=abs(f[n:]@ls(f[:n],Y[:n])-Y[n:]);return [e[:,0].mean(),e[:,1:].mean(),e[:,-1].mean()]
res=[]
for k,z in feats.items():
 for k0 in ['', 'z', 'b','c','d']:
  for deg in (1,2):
   for cross in (False,True) if k0 and deg==2 else (False,):
    f=a(z,*([feats[k0]] if k0 else []),*([p2(z)] if deg==2 else []),*([(z[:,:,None]*feats[k0][:,None,:]).reshape(-1,16)] if cross else [])); res.append((cv(f),k,k0,deg,cross))
for j in range(2): print('leader',j,*sorted(res,key=lambda r:r[0][j])[:15],sep='\n')
np.savez('/tmp/features10.npz',**feats)
