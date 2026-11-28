from structure import *
from itertools import product
u=s['panel_c']['u']; v=s['panel_c']['v1']; w=s['panel_c']['v2']; i=np.arange(19); j=np.arange(19,24)
def tests(name, fun, y, starts, bounds=(-5,5)):
 def res(p):
  f=fun(p); return (f[i]@np.linalg.lstsq(f[i],y[i],rcond=None)[0]-y[i]).ravel()
 fits=[least_squares(res,p,bounds=bounds,max_nfev=600) for p in starts]; m=min(fits,key=lambda m:np.linalg.norm(m.fun)); f=fun(m.x); coef=np.linalg.lstsq(f[i],y[i],rcond=None)[0]; print(name,np.round(m.x,5), 'fit',np.mean(abs(m.fun)),'CV',np.mean(abs(f[j]@coef-y[j]))); return m.x,coef
if __name__=='__main__':
 print('C v1 limits',v.min(0),v.max(0)); print('C v2 quadratic coeffs',np.linalg.lstsq(poly(u),w,rcond=None)[0].T)
 for label,fn in [('tanh',np.tanh),('sin',np.sin),('rational',lambda q:q/(1+q*q))]:
  tests(label+' shared',lambda p:np.c_[np.ones(len(u)),fn(p[0]*u+p[1])],v,[[1.,0],[2.,0]])
  tests(label+' diagonal',lambda p:np.c_[np.ones(len(u)),fn(u*p[:4]+p[4:])],v,[[1.,1,1,1,0,0,0,0],[2.,2,2,2,0,0,0,0]])
  tests(label+' + affine',lambda p:np.c_[poly(u,1),fn(u*p[:4]+p[4:])],v,[[1.,1,1,1,0,0,0,0],[2.,2,2,2,0,0,0,0]])
 for col in range(4):
  yy=v[:,col]; f=poly(u,1); init=np.linalg.lstsq(f[i],yy[i],rcond=None)[0];
  for name,fn,a0 in [('tanh',np.tanh,1.3),('sin',np.sin,1.3),('rat',lambda q:q/np.sqrt(1+q*q),1.5)]:
   pred=lambda p:p[0]+p[1]*fn(f@p[2:]); p=least_squares(lambda p:pred(p)[i]-yy[i],[0,a0,*init/a0],max_nfev=800).x; print(col,name,np.round(p,4),np.mean(abs(pred(p)[i]-yy[i])),np.mean(abs(pred(p)[j]-yy[j])))
 simple=np.linalg.lstsq(poly(u),w,rcond=None)[0]; print('v2 residual max',np.max(abs(w-poly(u)@simple)))
 f=next(f for e,name,q,f in temporal(z[:56]) if name=='step-wave'); amp=np.linalg.lstsq(f,z.T,rcond=None)[0].T
 for nm,co in [('raw',x),('A',xa),('B',xb),('AB',ab)]:
  for d in (1,2,3):
   F=np.c_[poly(co,d),amp,amp**2,amp.prod(1)]; c=np.linalg.lstsq(F[:56],y[:56],rcond=None)[0]; print('scalar with amplitudes',nm,d,np.mean(abs(F[56:]@c-y[56:])),np.round(c[-5:],3))
