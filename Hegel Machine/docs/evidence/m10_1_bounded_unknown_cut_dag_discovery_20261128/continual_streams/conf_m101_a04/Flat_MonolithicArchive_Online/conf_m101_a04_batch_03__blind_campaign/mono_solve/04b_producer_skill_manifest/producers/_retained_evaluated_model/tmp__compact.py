import numpy as np
from scipy.optimize import least_squares
from graph_probe import s,fit
z=np.load('/tmp/refined_sin+soft.npz'); x,c,d,Y=(z[k] for k in ('x','c','d','Y')); n=len(Y)

def base(x):return np.c_[np.ones(len(x)),x]
def report(name,F,Y):
 errors=[]
 for i in range(10):
  p=np.random.default_rng(i).permutation(n) if i>1 else np.arange(n); a,b=p[:52 if i==0 else 56],p[52 if i==0 else 56:]; E=np.abs(F[b]@fit(F[a],Y[a])-Y[b]);errors.append([E.mean(),E[:,-1].mean() if E.ndim==2 else E.max()]);
 print(name,np.round(errors[0],6),np.round(np.max(errors,axis=0),6),np.round(np.mean(errors,axis=0),6))

h=(c*d).sum(1); q=c[:,0]*d[:,1]; cross=(c[:,:,None]*d[:,None,:]).reshape(-1,16)
for name,F in [('sum',np.c_[c,d,h]),('sum const',np.c_[base(c),d,h]),('sum cross',np.c_[c,d,h,q]),('sum cross const',np.c_[base(c),d,h,q]),('diag cross const',np.c_[base(c),d,c*d,q]),('full cross',np.c_[base(c),d,cross])]:report('scalar '+name,F,Y[:,0])
for name,F in [('sum',np.c_[c.sum(1),d,h,q]),('sum const',np.c_[np.ones(len(c)),c.sum(1),d,h,q]),('full c',np.c_[base(c),d,h,q]),('full diag',np.c_[base(c),d,c*d,q]),('full cross',np.c_[base(c),d,cross])]:
 report('traj '+name,F,Y[:,1:]);
 if name in ('sum','full c'):
  C=fit(F[:52],Y[:52,1:]); print(np.round(C,4)); print('subspace sv',np.linalg.svd(C,compute_uv=False))
# Fit candidate operator libraries to panel measurements only.
p=s['panel_c']; u=p['u']
for key, models in [('v1',[
 ('soft',lambda a,f:np.c_[base(a),a/(1+f[0]*abs(a))],[.8],([.05],[5])),
 ('soft+sine',lambda a,f:np.c_[base(a),a/(1+f[0]*abs(a)),np.sin(f[1]*a)],[.8,4.5],([.05,.1],[5,9])),
 ('soft+cube',lambda a,f:np.c_[base(a),a/(1+f[0]*abs(a)),a**3],[.8],([.05],[5])),
 ('soft+soft',lambda a,f:np.c_[base(a),a/(1+f[0]*abs(a)),a/(1+f[1]*a*a)],[.8,1.],([.05,.05],[5,5])),
 ('soft+tanh',lambda a,f:np.c_[base(a),a/(1+f[0]*abs(a)),np.tanh(f[1]*a)],[.8,1.5],([.05,.05],[5,8]))]),('v2',[
 ('cos+square',lambda a,f:np.c_[base(a),np.cos(f[0]*a),a*a],[2.49],([.5],[5])),
 ('cos+cos',lambda a,f:np.c_[base(a),np.cos(f[0]*a),np.cos(f[1]*a)],[1.1,3.9],([.3,1.5],[2,8]))])]:
 v=p[key]
 for name,D,f,bounds in models:
  def model(a,b):
   def residual(f):
    A=D(a,f);return (A@fit(A,b)-b).ravel()
   opt=least_squares(residual,f,bounds=bounds,gtol=1e-12,ftol=1e-12,xtol=1e-12); A=D(a,opt.x);return opt.x,fit(A,b),np.std(opt.fun)
  w,B,r=model(u,v); E=[]
  for i in range(len(u)):
   m=np.arange(len(u))!=i; w1,B1,_=model(u[m],v[m]);E.extend((D(u[i:i+1],w1)@B1-v[i]).ravel())
  print(key,name,np.round(w,5),round(r,7),round(np.std(E),7));print(np.round(B,4))
