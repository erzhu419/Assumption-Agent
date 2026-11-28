import numpy as np,pickle
from scipy.optimize import minimize_scalar,least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; ls=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-10)[0]
fun={'sin':np.sin,'cos':np.cos,'tanh':np.tanh,'atan':np.arctan,'rat':lambda z:z/(1+z*z),'gauss':lambda z:np.exp(-z*z),'soft':lambda z:z/np.sqrt(1+z*z),'absrat':lambda z:z/(1+abs(z))}
u,v=[s['panel_a'][k] for k in ('u','v')]; keep=np.linalg.norm(v,axis=1)<5; u,v=u[keep],v[keep]; f=u; g=np.roll(u,2,axis=1)
for base in [lambda f,g:[f],lambda f,g:[f,f**3],lambda f,g:[f,f**2,f**3,f*g],lambda f,g:[f,f**2,f**3,f*g,g**2],lambda f,g:[f,f**3,f*g,np.roll(f,1)*np.roll(f,-1)]]:
 result=[]
 for k,h in fun.items():
  def cal(p,ret=False):
   Bs=[np.c_[np.ones(len(f)),*base(f[:,j],g[:,j]),h(p*g[:,j])] for j in range(4)]; cs=[ls(b,v[:,j]) for j,b in enumerate(Bs)]; r=np.column_stack([b@c-v[:,j] for j,(b,c) in enumerate(zip(Bs,cs))]);return (r,cs) if ret else np.mean(r*r)
  fit=minimize_scalar(cal,bounds=(.05,3.7),method='bounded');result.append((np.sqrt(fit.fun),k,fit.x,cal(fit.x,True)[1][0]))
 print('A',sorted(result,key=lambda a:a[0])[:3])
u,v=[s['panel_c'][k] for k in ('u','v1')]; one=np.ones(len(u))
# Fit single and shared-mixing paired families, including cyclic permutation and additive input mixtures.
results=[]
for i,(name,h) in enumerate(fun.items()):
 for shift in range(4):
  for inp in ('u','sum','diff'):
   U=u if inp=='u' else u+(1 if inp=='sum' else -1)*np.roll(u,shift,axis=1)
   if not U.any():continue
   def proj(p):
    b=np.c_[one,h(p*U)];return np.mean((b@ls(b,v)-v)**2)
   fit=minimize_scalar(proj,bounds=(.1,4.5),method='bounded');results.append((np.sqrt(fit.fun),name,inp,shift,fit.x))
print('C singles',sorted(results)[:12]);results=[]
for name,h in fun.items():
 for name2,h2 in fun.items():
  for shift in range(4):
   U=np.roll(u,shift,axis=1)
   def proj(p):
    b=np.c_[one,h(p[0]*u)+p[2]*h2(p[1]*U)];return (b@ls(b,v)-v).ravel()
   fit=least_squares(proj,[1.55,2.2,.25],bounds=([.12,.12,-2],[4.5,4.5,2]),max_nfev=55); results.append((np.std(fit.fun),name,name2,shift,fit.x))
print('C paired',sorted(results)[:15]); print('counts',len(s['train']['x']),len(s['test']['x']))
