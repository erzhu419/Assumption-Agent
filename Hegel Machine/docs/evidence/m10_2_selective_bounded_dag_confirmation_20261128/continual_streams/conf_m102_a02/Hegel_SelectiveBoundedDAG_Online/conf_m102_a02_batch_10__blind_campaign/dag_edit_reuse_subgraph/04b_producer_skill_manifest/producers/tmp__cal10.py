exec(open('/tmp/cyclic10.py').read().split('for name,Q')[0])
U,V=pa['u'],pa['v']; print('cal range',U.shape,np.max(abs(U),axis=0));
for j in range(4):
 x,z=U[:,j],U[:,(j+2)%4]; yy=V[:,j];
 for label,fn in [('cubic',lambda a:np.c_[np.ones(len(x)),x,z,z**3]),('quintic',lambda a:np.c_[np.ones(len(x)),x,z,z**3,z**5]),('rat',lambda a:np.c_[np.ones(len(x)),x,z/(1+np.exp(a[0])*z*z)]),('tanh',lambda a:np.c_[np.ones(len(x)),x,np.tanh(np.exp(a[0])*z)]),('atan',lambda a:np.c_[np.ones(len(x)),x,np.arctan(np.exp(a[0])*z)]),('sin',lambda a:np.c_[np.ones(len(x)),x,np.sin(np.exp(a[0])*z)])]:
  def fun(theta):return fn(theta[:1])@theta[1:]-yy
  init=ls(fn([0]),yy); fit=least_squares(fun,np.r_[0,init],loss='cauchy',f_scale=.003,max_nfev=600); err=np.abs(fun(fit.x)); print(j,label,fit.x.round(5),np.quantile(err,[.5,.8,.9,.95,1]).round(5));
