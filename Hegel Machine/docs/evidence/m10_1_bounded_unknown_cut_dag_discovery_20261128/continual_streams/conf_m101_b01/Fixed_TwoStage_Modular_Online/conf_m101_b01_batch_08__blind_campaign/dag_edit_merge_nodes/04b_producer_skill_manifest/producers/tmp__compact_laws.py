import pickle,itertools,numpy as np
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=5,suppress=True,linewidth=140)
def fit(X,y):
 return np.linalg.lstsq(X,y,rcond=None)[0]
def err(X,y):
 c=fit(X,y); return np.sqrt(np.mean((X@c-y)**2))
p=s['panel_a']; u=p['u']; z=p['v']; j=[1,2,3,0]; v=u[:,j]; X=np.c_[np.ones(len(u)),u,u*v]; w=np.ones(len(u))
for _ in range(15):
 c=fit(X*w[:,None],z*w[:,None]); r=np.linalg.norm(z-X@c,axis=1); w=1/np.sqrt(1+(r/.12)**2)
g=np.argsort(r)[:-3]; print('A gaps',np.sort(r)[-8:]); u,v,z=u[g],v[g],z[g];
for k in range(4):
 a,b,y=u[:,k],v[:,k],z[:,k]; D=np.c_[np.ones(len(a)),a,b,a*b]; T=np.c_[a*a,b*b,a**3,b**3,a*a*b,a*b*b,a*a*b*b,a**4,b**4,a*a*(a*a+b*b)]
 print('A',k,'pair',fit(D,y),'rmse',err(D,y),'cubic',err(np.c_[D,T[:,:6]],y),'coefs',fit(np.c_[D,T[:,:6]],y)); R=[]
 for label,h in {'a':a,'b':b,'a+b':a+b,'a-b':a-b,'a*b':a*b,'radius':np.sqrt(a*a+b*b)}.items():
  for name,F in [('divide all',lambda t:np.c_[D[:,0],D[:,1:]/(1+t*h[:,None]**2)]),('divide product',lambda t:np.c_[D[:,:3],a*b/(1+t*h*h)]),('divide own',lambda t:np.c_[D[:,0],a/(1+t*h*h),D[:,2:]]),('sin',lambda t:np.c_[D,np.sin(t*h)]),('tanh',lambda t:np.c_[D,np.tanh(t*h)])]:
   f=minimize_scalar(lambda t:err(F(t),y),bounds=(.001,3),method='bounded'); R.append((f.fun,label,name,f.x))
 print('A best',sorted(R)[:5]); print('A sparse', sorted((err(np.c_[D,T[:,q]],y),q) for q in itertools.combinations(range(T.shape[1]),2))[:3])
for pk,vk in [('panel_b','v'),('panel_c','v2')]:
 u=s[pk]['u']; y=s[pk][vk]; one=np.ones(len(u)); print(pk)
 for name,F in [('lin+tanh',lambda t:np.c_[one,u,np.tanh(t*u)]),('tanh',lambda t:np.c_[one,np.tanh(t*u)]),('lin+rat',lambda t:np.c_[one,u,u/(1+t*u*u)]),('lin+cos',lambda t:np.c_[one,u,1-np.cos(t*u)]),('lin+quad+cos',lambda t:np.c_[one,u,u*u,1-np.cos(t*u)]),('tanh+cos',lambda t:np.c_[one,np.tanh(t*u),1-np.cos(t*u)]),('lin+tanh+square',lambda t:np.c_[one,u,np.tanh(t*u),np.tanh(t*u)**2])]:
  f=minimize_scalar(lambda t:err(F(t),y),bounds=(.2,3.6),method='bounded'); c=fit(F(f.x),y); print(name,f.x,f.fun); 
  if f.fun<.015: print(c)
