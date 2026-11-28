exec(open('/tmp/grammar.py').read().split('for k,z in Z.items():')[0].replace("print('A',A,'B',B)",''))
from scipy.optimize import least_squares
z=Z['cal']; l=aff(z); q=terms(z,p2); P=np.c_[l,q]; i=np.arange(52); j=np.arange(52,72)

def fit(fun,p0,bounds=(-np.inf,np.inf)):
 def calc(p):
  F=fun(p); c=np.linalg.lstsq(F[i],y[i],rcond=None)[0]; return y-F@c,c
 m=least_squares(lambda p:calc(p)[0][i],p0,bounds=bounds,max_nfev=150)
 r,c=calc(m.x); return np.sqrt(np.mean(r[i]**2)),np.mean(abs(r[j])),m.x,c
for deg in [1,2]:
 for free in [0,1,2]:
  for form in ['combined','pure','affine_pure']:
   R=l if deg==1 else P
   def f(p):
    w=z[:,2]-p[1]*z[:,0]+p[2]
    if free==2: w=w+z[:,[1,3]]@p[3:]
    g=1/(1+p[0]*w*w)
    return {'combined':lambda:np.c_[P,R*(g[:,None]-1)],'pure':lambda:R*g[:,None],'affine_pure':lambda:np.c_[l,R*g[:,None]]}[form]()
   p0=[.5,1.,0.]+([0.,0.] if free==2 else [])
   if free==0:
    fn=lambda a:f([a[0],1,0]); bounds=([.005],[4]);p0=p0[:1]
   else:
    fn=f; bounds=([.005,.1,-1]+([-1,-1] if free==2 else []),[4,2,1]+([1,1] if free==2 else []))
   a,b,p,c=fit(fn,p0,bounds); print(deg,free,form,round(a,6),round(b,6),p.round(4),c.round(3))
# Recover the smooth trajectory within the exact affine spatial family.
M=np.linalg.lstsq(l[i],T[i],rcond=None)[0]
print('times',t)
for name,init,fn,bounds in [('exp2',[2.,.35],lambda p:-np.expm1(-t[:,None]*p),([.05,.02],[15,15])),('exp+linear',[2.],lambda p:np.c_[-np.expm1(-t*p[0]),t],([.02],[15])),('exp1',[2.],lambda p:-np.expm1(-t[:,None]*p),([.02],[15]))]:
 def res(p, full=False):
  H=fn(p); V=np.linalg.lstsq(H,M.T,rcond=None)[0].T@H.T
  return V if full else (l[i]@(M-V)).ravel()
 m=least_squares(res,init,bounds=bounds); r=l@res(m.x,True)-T; print(name,m.x,np.mean(abs(r[i])),np.mean(abs(r[j])),np.mean(abs(r[j]),axis=0))
