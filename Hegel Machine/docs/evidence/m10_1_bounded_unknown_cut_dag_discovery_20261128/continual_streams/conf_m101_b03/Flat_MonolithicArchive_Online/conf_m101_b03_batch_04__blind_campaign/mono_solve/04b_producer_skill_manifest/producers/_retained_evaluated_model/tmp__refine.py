exec(open('/tmp/graph_probe.py').read().split("print('C1 coef")[0])
from itertools import combinations
from scipy.optimize import minimize_scalar
u=s['panel_c']['u']; g0=s['panel_c']['v2']
edges=((0,1),(0,3),(1,2),(2,3))
ed=lambda x:np.column_stack([x[:,i]*x[:,j] for i,j in edges]); f=A(ed(z))@fit(A(ed(u)),f0)
funcs={'rational':lambda x,k:1/(1+k*x*x),'sqrt':lambda x,k:1/np.sqrt(1+k*x*x),'gauss':lambda x,k:np.exp(-k*x*x),'cos':lambda x,k:np.cos(k*x),'log':lambda x,k:np.log1p(k*x*x),'hypot':lambda x,k:np.sqrt(1+k*x*x),'tanh2':lambda x,k:np.tanh(k*x)**2,'atan2':lambda x,k:np.arctan(k*x)**2}
models=[]
for name,fun in funcs.items():
 def ds(w,k):return A(w,fun(w,k))
 def loss(k):
  a=ds(u,k);return np.mean((a@fit(a,g0)-g0)**2)
 r=minimize_scalar(loss,bounds=(.001,2),method='bounded').x; a=ds(u,r); c=fit(a,g0); e=a@c-g0
 loo=(np.mean((e/(1-np.sum(a*np.linalg.pinv(a).T,axis=1))[:,None])**2))**.5
 models.append((loo,name,ds(z,r)@c));print(name,round(r,5),np.std(e),loo)
g=min(models,key=lambda x:x[0])[2]
y=np.c_[tr['y'],tr['traj']]; n=52; t=s['eval_times']; H=lambda r:-np.expm1(-t[:,None]*r)
pair=lambda x,w:np.einsum('ni,nj->nij',x,w).reshape(len(x),-1)
base=A(z,products(z,2),f*g)
cols={'base':base,'base+fg':A(z,products(z,2),pair(f,g)),'base+zfg':A(z,products(z,2),pair(z,g),pair(f,g)),'base+ff':A(z,products(z,2),f*g,products(f,2)),'base+zf':A(z,products(z,2),f*g,pair(z,f)),'base+g':A(z,products(z,2),f*g,g),'base+diagz':A(z,products(z,2),f*g,z*g,z*f),'base+all2fg':A(z,products(z,2),pair(f,g),products(f,2),products(g,2)),'fg2':A(f,g,products(np.c_[f,g],2))}
r=least_squares(lambda r:(H(r)@fit(H(r),y[:n,1:].T)-y[:n,1:].T).ravel(),[.4,1.8],bounds=([.05,.8],[.8,5])).x; h=H(r); q=np.c_[y[:,0],fit(h,y[:,1:].T).T]
for nm,a in cols.items():
 x=a[:,1:]-a[:n,1:].mean(0);x/=np.maximum(x[:n].std(0),1e-10); U,S,V=svd(x[:n],full_matrices=False); vals=[]
 for lam in [1e-8,1e-5,.0001,.001,.01,.1,1]:
  pred=q[:n].mean(0)+x@(V.T*(S/(S*S+lam)))@U.T@q[:n]; p=np.c_[pred[:,0],pred[:,1:]@h.T]; e=abs(p-y); vals.append((lam,*np.round([e[:n,0].mean(),e[n:,0].mean(),e[:n,1:].mean(),e[n:,1:].mean()],5)))
 print(nm,'best y',min(vals,key=lambda v:v[2]),'best T',min(vals,key=lambda v:v[4]))
coef=fit(base[:n],q[:n]); print('base coefficients',np.round(coef,3)); e=q-base@coef
print('base mode error',np.mean(abs(e[:n]),0),np.mean(abs(e[n:]),0));print('time residual',np.sqrt(np.mean((y[:,1:]-q[:,1:]@h.T)**2,axis=0)))
