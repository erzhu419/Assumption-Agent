exec(open('/tmp/explore.py').read().split("a=s['panel_a']")[0]); from scipy.optimize import root,minimize_scalar
A=s['panel_a']; u,v=A['u'],A['v']; rows=np.linalg.norm(v,axis=1)<5; B=ls(aff(s['panel_b']['u']),s['panel_b']['v']);
for j in range(4):
 a=u[:,j]; b=u[:,(j+1)%4]; f=np.c_[np.ones(len(u)),a,b,a*b,a**3,b**3]; c=ls(f[rows],v[rows,j]); print('A',j,c,np.std((f@c-v[:,j])[rows]))
F=poly(u,3); co=ls(F[rows],v[rows]); print('cubic',co[15:]); z=np.array([root(lambda h:(poly(h[None,:],3)@co)[0]-r,r).x for r in x]); w=aff(z)@B
h=s['eval_times']; n=52
for kind in range(3):
 def form(r):
  if kind==0:return np.array([-np.expm1(-r[0]*h),-np.expm1(-r[1]*h)]).T
  if kind==1:return np.array([-np.expm1(-r[0]*h),h*np.exp(-r[1]*h)]).T
  return np.array([-np.expm1(-r[0]*h),h]).T
 def fun(r):
  f=form(r); return (ls(f,q[:n].T).T@f.T-q[:n]).ravel()
 r=least_squares(fun,[.7,2.5],bounds=(.05,12)).x; print('temporal',kind,r,np.std(fun(r)),np.std(ls(form(r),q.T).T@form(r).T-q))
for fn in ['ic','asinh','atan','tanh']:
 def feat(g):
  if fn=='ic':return 2/np.sqrt(3)*np.sinh(np.arcsinh(1.5*np.sqrt(3)*g*w)/3)
  return {'asinh':np.arcsinh,'atan':np.arctan,'tanh':np.tanh}[fn](g*w)
 for name,tar in [('y',y),('traj',q)]:
  for mode in range(5):
   def design(g):
    f=feat(g); return [aff(f),np.c_[aff(f),w],poly(f,2),np.c_[poly(f,2),w],np.c_[poly(f,2),poly(f,3)[:,-20:]]][mode]
   def loss(g):
    f=design(g)[:n]; return np.mean((f@ls(f,tar[:n])-tar[:n])**2)
   g=minimize_scalar(loss,bounds=(.15,12),method='bounded').x; f=design(g); pred=f[n:]@ls(f[:n],tar[:n]); print(fn,name,mode,round(g,5),round(loss(g)**.5,6),round(np.mean(abs(pred-tar[n:])),6),round(np.max(np.mean(abs(pred-tar[n:]),axis=0)),6));
   if fn=='ic' and mode in (0,1,2,3):print('coef',ls(f[:n],tar[:n]) if name=='y' else ls(f[:n],tar[:n])[:,[0,-1]].T)
