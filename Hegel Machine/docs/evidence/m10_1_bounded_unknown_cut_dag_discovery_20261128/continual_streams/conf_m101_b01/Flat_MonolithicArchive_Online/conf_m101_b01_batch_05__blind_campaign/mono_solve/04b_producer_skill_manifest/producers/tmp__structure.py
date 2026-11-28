from probe import *
from scipy.special import erfi
u,v=s['panel_b']['u'],s['panel_b']['v']; ui,uj=u[:,l],u[:,r]
F=dict(funcs, gaussodd=lambda x,k:x*np.exp(-k*x*x),expm=lambda x,k:x*(-np.expm1(-k*x*x)), rationalcub=lambda x,k:x**3/(1+k*x*x), softcub=lambda x,k:x**3/(1+k*abs(x))**2)
for name,g in F.items():
 def d(p,x=u):return aff(x+p[0]*g(x,np.exp(p[1])))
 def res(p):
  a=d(p);return (a@fit(a,v)-v).ravel()
 opts=[least_squares(res,[a,k],bounds=([-20,-4],[20,3]),max_nfev=120) for a in [-1,1] for k in [-1,0,1,2]]
 o=min(opts,key=lambda z:z.cost); print(name,np.std(o.fun),o.x[0],np.exp(o.x[1]))
for name,g in F.items():
 def d(k,h):
  z=g(u,k); return np.c_[aff(u),z,h]
 out=[]
 for tag,h in [('even',u*u),('cross',ui*uj),('cyclic',u*np.roll(u,-1,axis=1)),('triple',np.array([np.prod(np.delete(u,j,axis=1),axis=1) for j in range(4)]).T),('h5',u**5),('sumdiff',(u+np.roll(u,-1,axis=1))**3),('diff',(u-np.roll(u,-1,axis=1))**3)]:
  grid=np.geomspace(.04,9,80); e=[err(d(k,h),v) for k in grid]; a=np.argmin(e); o=minimize_scalar(lambda k:err(d(k,h),v),bounds=(grid[max(0,a-1)],grid[min(len(grid)-1,a+1)]),method='bounded'); out.append((o.fun,tag,o.x))
 print('aug',name,sorted(out)[:2]);
 if min(out)[0]<.0005:
  _,tag,k=min(out); print(np.round(fit(d(k,dict(even=u*u,cross=ui*uj,cyclic=u*np.roll(u,-1,axis=1),h5=u**5)[tag]),v),4))
print('B input ranges',u.min(0),u.max(0))
