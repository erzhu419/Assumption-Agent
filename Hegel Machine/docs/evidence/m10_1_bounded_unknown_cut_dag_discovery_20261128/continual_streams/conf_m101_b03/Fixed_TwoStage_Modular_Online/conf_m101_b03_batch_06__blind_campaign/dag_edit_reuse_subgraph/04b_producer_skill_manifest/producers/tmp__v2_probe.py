from probe import *
from scipy.optimize import minimize_scalar

def check(name,fun,bounds=None):
 if bounds:
  vals=np.linspace(*bounds,151); f=lambda k: loo(fun(u,k),v2).mean(); i=np.argmin([f(k) for k in vals]); opt=minimize_scalar(f,bounds=(vals[max(0,i-1)],vals[min(len(vals)-1,i+1)]),method='bounded'); k=opt.x
 else:k=1.
 P=fun(u,k); Q=ls(P,v2); g=fun(z,k)@Q; print(name,round(k,6),'loo',loo(P,v2),'official/worst',ev(np.c_[f1,g])[0],ev(np.c_[f1,g]).max(0)); return k
add=lambda x,f:np.c_[np.ones(len(x)),f]
for name,fun,range_ in [ ('cos',lambda x,k:add(x,np.cos(k*x)),(.1,12)),('sin',lambda x,k:add(x,np.sin(k*x)),(.1,12)),('sin_square',lambda x,k:add(x,np.sin(k*x*x)),(.1,12)),('cos_square',lambda x,k:add(x,np.cos(k*x*x)),(.1,12)),('rational',lambda x,k:add(x,1/(1+k*x*x)),(.001,12)),('abs',lambda x,k:add(x,abs(x)),None),('evenpower',lambda x,k:add(x,abs(x)**k),(.1,7)),('quartic',lambda x,k:add(x,np.c_[x*x,x**4]),None),('sextic',lambda x,k:add(x,np.c_[x*x,x**4,x**6]),None),('mixed cos',lambda x,k:add(x,np.c_[np.cos(k*x),x[:,0]*x[:,1],x[:,0]*x[:,3],x[:,1]*x[:,2],x[:,2]*x[:,3]]),(.1,12)),('mixed trig',lambda x,k:add(x,np.c_[np.cos(k*x),np.sin(k*x)]),(.1,12))]:check(name,fun,range_)
