import numpy as np
from scipy.optimize import least_squares
exec(open('/tmp/explore02.py').read().split('p=least_squares')[0])
D=np.c_[A(z),z[:,0]*z[:,2]+z[:,1]*z[:,3]]
D2=np.c_[A(z),z[:,0]*z[:,2],z[:,1]*z[:,3]]
for X in (D,D2):
 for name,fn in [('atan',lambda h:np.c_[h*0+1,np.arctan(h)]),('tanh',lambda h:np.c_[h*0+1,np.tanh(h)]),('sin',lambda h:np.c_[h*0+1,np.sin(h)]),('atan+linear',lambda h:np.c_[h*0+1,np.arctan(h),h]),('tanh+linear',lambda h:np.c_[h*0+1,np.tanh(h),h]),('sin+linear',lambda h:np.c_[h*0+1,np.sin(h),h]),('sin+cos+linear',lambda h:np.c_[h*0+1,np.sin(h),np.cos(h),h]),('rational+linear',lambda h:np.c_[h*0+1,h/(1+h*h),h]),('sin+second',lambda h:np.c_[h*0+1,np.sin(h),np.sin(2*h)]),('tanh+atan',lambda h:np.c_[h*0+1,np.tanh(h),np.arctan(h)])]:
  vals=[]
  for vi in [va,np.random.default_rng(12).permutation(len(y))[:16],np.random.default_rng(21).permutation(len(y))[:16]]:
   ti=np.setdiff1d(np.arange(len(y)),vi); init=np.linalg.lstsq(X[ti],y[ti],rcond=None)[0]
   def calc(p,ii):
    B=fn(X@p); return B[ii]@np.linalg.lstsq(B[ti],y[ti],rcond=None)[0]
   opt=min([least_squares(lambda p:calc(p,ti)-y[ti],init*val,max_nfev=180) for val in (.35,.7,1.1,1.6)],key=lambda o:o.cost)
   vals.append((np.abs(calc(opt.x,ti)-y[ti]).mean(),np.abs(calc(opt.x,vi)-y[vi]).mean()))
  print(X.shape[1],name,np.array(vals).round(5).tolist())
