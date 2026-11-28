import pickle,numpy as np,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; p=s['panel_c'];u,v=p['u'],p['v2']; n,d=u.shape; base=np.c_[np.ones(n),u]; idx=np.random.default_rng(53).permutation(n); a,b=idx[:18],idx[18:];
def ls(d,y):return np.linalg.lstsq(d,y,rcond=None)[0]
def rm(x):return np.sqrt(np.mean(x*x))
def ortho(x):return x-base@ls(base,x)
w=ortho(v); print('residual singulars',np.linalg.svd(w,compute_uv=False)); inds=list(itertools.combinations_with_replacement(range(4),2)); q=np.stack([u[:,i]*u[:,j] for i,j in inds],1); print('quad coefs',inds,ls(ortho(q),w).round(4));print('quad remainder',np.linalg.svd(w-ortho(q)@ls(ortho(q),w),compute_uv=False));
funcs={'id':lambda t:t,'soft':lambda t:t/(1+abs(t)),'tanh':np.tanh,'cos':np.cos,'sin':np.sin,'rat':lambda t:1/(1+t*t),'odd_rat':lambda t:t/(1+t*t),'exp':lambda t:np.exp(-t*t)}
cases=[]
for off in [1,2,3]:
 x,y=u,np.roll(u,-off,axis=1); xy=x*y
 for label,z in [('prod',xy),('sum',x+y),('diff',x-y)]:
  for f,fn in funcs.items():
   for linear in [False,True]:
    def des(t,z=z,fn=fn,xy=xy,linear=linear):return np.c_[base,*([xy] if linear else []),fn(t[0]*z+t[1])]
    cases.append(((off,label,f,linear),des,2))
 for what in range(9):
  for f,fn in funcs.items():
   def des(t,x=x,y=y,xy=xy,fn=fn,what=what):
    k,h=t
    z=[x*fn(k*y+h),fn(k*x+h)*fn(k*y+h),xy/(1+k*x*x+h*h*y*y),xy/(1+k*(x+y)**2+h*h*(x-y)**2),xy/(1+k*xy*xy+h*h*x*x),xy/(1+k*x*x),x*y+k*x*x*y+h*x*y*y,xy+k*x*x*y*y+h*(xy**3),xy/(1+k*(x-y)**2)][what]
    return np.c_[base,z]
   cases.append(((off,'mixed',what,f),des,2))
res=[]
for name,d,k in cases:
 def err(t):
  m=d(t)[a];return (m@ls(m,v[a])-v[a]).ravel()
 fits=[least_squares(err,[c,h],bounds=([.0001,-5],[9,5]),max_nfev=70) for c,h in [(.4,.15),(1.25,.6),(2.15,-.5)]]; fit=min(fits,key=lambda f:f.cost);m=d(fit.x);res.append((rm(m[b]@ls(m[a],v[a])-v[b]),rm(fit.fun),name,fit.x.round(5)))
for r in sorted(res,key=lambda t:t[0])[:28]: print(r)
for r in sorted(res,key=lambda t:t[1])[:12]: print('in',r)
