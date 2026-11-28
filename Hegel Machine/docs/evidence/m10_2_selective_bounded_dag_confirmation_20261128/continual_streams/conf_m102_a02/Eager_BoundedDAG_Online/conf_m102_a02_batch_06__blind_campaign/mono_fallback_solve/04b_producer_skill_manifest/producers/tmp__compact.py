exec(open('/tmp/readout.py').read().split("print('range'")[0])
from scipy.optimize import least_squares
from scipy.special import erf
q=(g*np.roll(g,2,axis=1))[:,:2]
for name,A in [('no z',np.c_[D,q]),('with z',np.c_[D1,q]),('with sum',np.c_[D1,q.sum(1)]),('with z cubic',np.c_[D1,q,z**3]),('signed pair',np.c_[D1,q[:,0]-q[:,1]])]:
 print(name,cv(A),fit(A,y), 'trajectory',cv(A,T)); R=y-A@fit(A,y)
 for nm,F0 in [('z cube',z**3),('g cube',g**3),('cyclic v',v),('z q',z*q.sum(1)[:,None]),('z opposite',(z*np.roll(z,2,axis=1))[:,:2])]: print('  ',nm,cv(np.c_[A,F0]))
# fit a general monotone nonlinear distortion before a linear mixing, with a separate small linear correction
for nm,act in [('erf',erf),('tanh',np.tanh),('atan',np.arctan),('rational',lambda a:a/(1+a*a)),('gaussian',lambda a:a*np.exp(-a*a)),('sin',np.sin)]:
 for mode in range(4):
  def mat(p,a):
   w=np.exp(p[0]); G=act(w*a); L=np.c_[aff(a),G];
   if mode==1:L=np.c_[L,a**3]
   if mode==2:L=np.c_[L,a**2]
   if mode==3:L=np.c_[L,a*np.roll(a,1,axis=1)]
   return L
  def fun(p):
   A=mat(p,u); return (A@fit(A,V)-V).ravel()
  res=min([least_squares(fun,[np.log(w0)],bounds=(-4,1.55),max_nfev=60) for w0 in (.6,1.4,2.1)],key=lambda r:sum(r.fun**2))
  A=mat(res.x,u); b=fit(A,V); leverage=(np.linalg.svd(A,full_matrices=False)[0]**2).sum(1); G=mat(res.x,z)@b; vv=z*np.roll(G,1,axis=1); qq=(G*np.roll(G,2,axis=1))[:,:2]; Dnew=aff(np.c_[G,z,vv,qq]); print(nm,mode,'panel',round(float(np.mean(res.fun**2)**.5),6),round(float(np.mean((res.fun.reshape(-1,4)/(1-leverage[:,None]))**2)**.5),6),'w',np.exp(res.x),'y',cv(Dnew),'T',cv(aff(np.c_[G,z,vv]),T),'q coeff',fit(Dnew,y)[-2:])
