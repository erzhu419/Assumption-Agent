exec(open('/tmp/probe07.py').read().split('for name,x')[0])
from scipy.optimize import minimize_scalar
for label,x,v in [('B',b['u'],b['v']),('Brev',b['v'],b['u']),('C2',c['u'],c['v2'])]:
 print(label,'ranges',x.min(0),x.max(0),v.min(0),v.max(0)); print(np.round(np.c_[x,v],3));
 def err(F,z=v):
  H=F@np.linalg.pinv(F); e=(z-H@z)/(1-np.diag(H)[:,None]);return np.sqrt(np.mean(e*e))
 opts=[]
 for name,fn in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('atan',np.arctan),('sinh',np.sinh),('asinh',np.arcsinh),('sat',lambda z:z/(1+z*z)),('sqrt',lambda z:z/np.sqrt(1+z*z)),('exp',np.exp),('logcosh',lambda z:np.logaddexp(z,-z)),('rational',lambda z:z*z/(1+z*z)),('softplus',lambda z:np.logaddexp(z,0))]:
  for k in np.geomspace(.1,15,65):
   z=fn(k*x)
   for m,F in [('only',aff(z)),('lin',np.c_[aff(x),z]),('sq',np.c_[aff(x),x*x,z]),('quad',np.c_[aff(z),z*z]),('pair',np.c_[aff(z),pair(z)])]:opts.append((err(F),name,k,m))
 print('best',sorted(opts)[:12]);
 for j in range(4):
  opts=[]
  for name,fn in [('sin',np.sin),('tanh',np.tanh),('atan',np.arctan),('asinh',np.arcsinh),('identity',lambda z:z)]:
   G=np.c_[x, np.roll(x,-1,axis=1),np.roll(x,-2,axis=1)];
   for k in np.geomspace(.15,15,44):
    for l in range(4):
     for h in range(l+1,4):
      for r in (-1,0,1):
       z=fn(k*(x[:,l]+r*x[:,h]));F=np.c_[aff(x),z,z*z];opts.append((err(F,v[:,j:j+1]),name,k,l,h,r));
  print('j',j,sorted(opts)[:2]);
