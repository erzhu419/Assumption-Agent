from probe import *
from scipy.optimize import least_squares

def calibration(p):
 u,v=p['u'],p['v']; j=np.roll(np.arange(4),-1)
 def cols(u,i):
  a,b=u[:,i],u[:,j[i]]
  return np.c_[np.ones(len(u)),a,b,a*b,a**3]
 keep=np.argsort(np.linalg.norm(v-aff(u)@fit(aff(u),v),axis=1))[:-3]
 c=np.column_stack([fit(cols(u[keep],i),v[keep,i]) for i in range(4)])
 def forward(u):return np.column_stack([cols(u,i)@c[:,i] for i in range(4)])
 def invert(x):
  z=(x-c[0])/c[1]; i=np.arange(4)
  for _ in range(20):
   jac=np.zeros((len(z),4,4));jac[:,i,i]=c[1]+c[3]*z[:,j]+3*c[4]*z*z;jac[:,i,j]=c[2]+c[3]*z
   z-=np.linalg.solve(jac,(forward(z)-x)[...,None])[...,0]
  return z
 return c,forward,invert,keep

if __name__=='__main__':
 p=s['panel_a'];c,f,inv,k=calibration(p);z=inv(x);zt=inv(s['test']['x']);print('A coefficients',c,'A residual rms',np.mean((f(p['u'][k])-p['v'][k])**2)**.5,'inverse max residual',max(np.max(abs(f(z)-x)),np.max(abs(f(zt)-s['test']['x']))));print('calibrated ranges',z.min(0),z.max(0),zt.min(0),zt.max(0))
 def cv(d):
  e=d[52:]@fit(d[:52],Y[:52])-Y[52:];return np.mean(abs(e[:,0])),np.mean(abs(e[:,1:]))
 for d in (1,2,3):print('A inverse poly',d,cv(poly(z,d)))
 for name in ('panel_b','panel_c'):
  p=s[name];u=p['u']
  for target,v in p.items():
   if target=='u':continue
   print(name,target)
   for key,g in [('affine',aff),('diag2',lambda u:np.c_[aff(u),u*u]),('diag3',lambda u:np.c_[aff(u),u*u,u**3]),('pair2',lambda u:poly(u,2)),('harmonic1',lambda u:np.c_[aff(u),np.sin(2*u),np.cos(2*u)]),('harmonic2',lambda u:np.c_[aff(u),np.sin(4*u),np.cos(4*u)])]:
    d=g(u);w=fit(d,v);e=v-d@w;lev=np.sum(np.linalg.qr(d,mode='reduced')[0]**2,1)
    print(key,'panel loo rms',np.sqrt(np.mean((e/(1-lev[:,None]))**2,0)),'after A label/state',cv(aff(g(z)@w)))
