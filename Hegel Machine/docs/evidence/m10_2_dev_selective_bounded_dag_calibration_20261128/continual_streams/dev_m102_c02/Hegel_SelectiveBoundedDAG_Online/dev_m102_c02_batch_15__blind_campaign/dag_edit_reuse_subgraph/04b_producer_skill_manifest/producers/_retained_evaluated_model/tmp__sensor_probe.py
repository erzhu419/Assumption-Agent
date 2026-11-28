import pickle,numpy as np
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u,v=(s['panel_a'][k] for k in ['u','v']); n=len(u); affine=np.c_[np.ones(n),u]; e=v-affine@np.linalg.lstsq(affine,v,rcond=None)[0]; keep=np.argsort(np.linalg.norm(e,axis=1))[:-3]; u=u[keep]; v=v[keep]; A=np.c_[np.ones(len(u)),u]
def score(d,y):
 b=np.linalg.lstsq(d,y,rcond=None)[0]; h=np.sum(np.linalg.svd(d,full_matrices=False)[0]**2,axis=1); r=y-d@b; p=r/(1-h).reshape((-1,)+(1,)*(y.ndim-1)); return np.sqrt(np.mean(p*p)),np.sqrt(np.mean(r*r)),b
print('removed',sorted(set(range(n))-set(keep))); print('affine',score(A,v)[:2]); print('affine coef',score(A,v)[2].round(4));
for j in range(4):
 z=u[:,j]; r=v[:,j]-A@score(A,v[:,j])[2]; ii=z.argsort(); print(j,np.c_[z[ii],r[ii]][::2].round(3).T)
features={'quadratic':u*u,'cubic':u**3,'cross':np.column_stack([u[:,i]*u[:,j] for i in range(4) for j in range(i)]),'abs':abs(u),'sign':np.sign(u)}
for k,x in features.items(): print(k,score(np.c_[A,x],v)[:2]);
for typ,f in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('rat',lambda x:x/(1+x*x))]:
 for approach in ['linear_all','linear_self','sumdiff']:
  def calc(w,details=False):
   if approach=='linear_all': return score(np.c_[A,f(w*u)],v)
   if approach=='linear_self':
    a=[score(np.c_[np.ones(len(u)),u[:,j],f(w*u[:,j])],v[:,j]) for j in range(4)]; return (np.mean([b[0] for b in a]),np.mean([b[1] for b in a]),np.array([b[2] for b in a]))
   pairs=np.column_stack([u[:,i]+sg*u[:,j] for i in range(4) for j in range(i) for sg in [-1,1]]); return score(np.c_[A,f(w*pairs)],v)
  grid=np.linspace(.2,13,257); val=np.array([calc(x)[0] for x in grid]); a=val.argmin(); opt=minimize_scalar(lambda p:calc(p)[0],bounds=(grid[max(a-1,0)],grid[min(a+1,len(grid)-1)]),method='bounded'); print(typ,approach,round(opt.x,5),calc(opt.x)[:2]);
