exec(open('/tmp/spatial_probe.py').read().split('for mode')[0])
from scipy.optimize import minimize_scalar
ls=lambda f,y: np.linalg.lstsq(f,y,rcond=1e-10)[0]
aff=lambda x:np.c_[np.ones(len(x)),x]
z=np.roll(u,-1,axis=1); F=np.stack([np.ones_like(u),u,z,u*z],axis=-1)
mask=np.linalg.norm(v,axis=1)<5
print('panel A rows',len(u),mask.sum(),np.sort(np.linalg.norm(v,axis=1)))
r=v[mask]-np.stack([F[mask,i]@ls(F[mask,i],v[mask,i]) for i in range(4)],axis=1)
for i in range(4):
 b=F[mask,i]; y=v[mask,i]; d=u[mask]; j=(i+1)%4; extra={}
 for k in range(4):
  for o in range(2,6):extra[f'{k}^{o}']=d[:,k]**o
  for l in range(k+1,4):
   for m,n in [(1,1),(1,2),(2,1),(2,2),(1,3),(3,1)]:extra[f'{k}^{m}*{l}^{n}']=d[:,k]**m*d[:,l]**n
 for k,l,h in combinations(range(4),3):extra[f'{k}*{l}*{h}']=d[:,k]*d[:,l]*d[:,h]
 def score(f):return np.linalg.norm(y-f@ls(f,y))
 ranked=sorted((score(np.c_[b,e]),k) for k,e in extra.items()); print('A',i,score(b),ranked[:8]);
 for step in range(4):
  err,k=min((score(np.c_[b,e]),k) for k,e in extra.items()); b=np.c_[b,extra.pop(k)]; print(k,round(err,7),np.round(ls(b,y),4));
C=s['panel_c']; x=C['u'];
funcs={'x2':lambda x,k:x*x,'x3':lambda x,k:x**3,'r2':lambda x,k:x*x/(1+k*x*x),'r3':lambda x,k:x**3/(1+k*x*x),'r1abs':lambda x,k:x/(1+k*abs(x)),'r2abs':lambda x,k:x*x/(1+k*abs(x)),'sin':lambda x,k:np.sin(k*x),'cos':lambda x,k:np.cos(k*x),'tanh':lambda x,k:np.tanh(k*x),'abs':lambda x,k:abs(x),'exp':lambda x,k:np.exp(k*x)}
for key in ['v1','v2']:
 y=C[key]; print('C',key,'shape',x.shape,'range',x.min(),x.max());
 for name,h in funcs.items():
  def loss(k):
   f=np.c_[aff(x),h(x,k)];return np.linalg.norm(y-f@ls(f,y));
  grid=np.linspace(.005,5,51); i=np.argmin([loss(k) for k in grid]); lo,hi=grid[max(i-1,0)],grid[min(i+1,50)]; out=minimize_scalar(loss,bounds=(lo,hi),method='bounded'); print(name,round(out.fun,7),round(out.x,6));
 r=y-opts[1](x); print('resid row norm',np.round(np.sort(np.linalg.norm(r,axis=1)),3)); print('affine closest residual quantiles',np.quantile(np.linalg.norm(y-aff(x)@ls(aff(x),y),axis=1),[0,.25,.5,.75,1]));
