from laws12 import *
u,v=s['panel_a']['u'],s['panel_a']['v'];_,_,keep=calfit('aff');u,v=u[keep],v[keep];q=np.roll(u,2,axis=1); eye=np.broadcast_to(np.eye(4),(len(u),4,4)); common=[eye,u[:,:,None]*eye]
for name,f in [('sin',np.sin),('tanh',np.tanh),('rational',lambda w:w/(1+w*w)),('atan',np.arctan),('root',lambda w:w/np.sqrt(1+w*w)),('polynomial',lambda w:w**5),('seventh',lambda w:w**7)]:
 for cubic in [False,True]:
  def d(p):
   non=f(u*p[1]); xs=[np.sin(p[0]*q),non]+([u**3] if cubic else []);return np.concatenate(common+[np.stack(xs,-1)],axis=2).reshape(len(u)*4,-1)
  def res(p):
   a=d(p);return a@ls(a,v.ravel())-v.ravel()
  opt=min((least_squares(res,[2.4,a],bounds=([1.8,.1],[2.9,6]),max_nfev=170) for a in [.5,1.25,2.2,3.3]),key=lambda o:np.linalg.norm(o.fun));print(name,cubic,np.sqrt(np.mean(opt.fun**2)),opt.x,np.round(ls(d(opt.x),v.ravel()),6))
