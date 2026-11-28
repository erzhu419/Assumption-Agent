import pickle,numpy as np
from scipy.optimize import least_squares, minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
ls=lambda f,v:np.linalg.lstsq(f,v,rcond=1e-10)[0]
one=lambda *f:np.column_stack((np.ones(len(f[0])),*f))
u,v=(s['panel_a'][k] for k in ('u','v'))
for j in range(4):
 a,b=u[:,j],u[:,(j+1)%4]; y=v[:,j]; keep=np.linalg.norm(v,axis=1)<5
 for f in (one(a),one(a,b*(a-1),a**3),one(a,b,a*b,a*a,a**3),one(u,u*u,u**3)):
  p=least_squares(lambda p:f@p-y,ls(f[keep],y[keep]),loss='cauchy',f_scale=.002).x
  e=np.abs(f@p-y); print('A',j,f.shape[1],np.round(p,5),np.quantile(e,[.5,.8,.9,.95,1]).round(5))
for pk,ks in [('panel_b',['v']),('panel_c',['v1','v2'])]:
 u=s[pk]['u']; cyc=u*np.roll(u,-1,axis=1)
 for k in ks:
  v=s[pk][k]
  def err(f): return np.sqrt(np.mean((f@ls(f,v)-v)**2))
  print(pk,k, [(n,round(err(f),6)) for n,f in [('linear',one(u)),('cyclic',one(cyc)),('cos',one(np.cos(2.4*u))),('sine',one(np.sin(2.4*u))),('poly',one(u,u*u,u**3)),('nonsep',one(u,u*u,cyc))]])
  def opt(fn,lo,hi):
   a=minimize_scalar(lambda x:err(fn(x)),bounds=(lo,hi),method='bounded'); return (round(a.x,6),round(a.fun,7))
  print('shapes',[(n,opt(f,lo,hi)) for n,f,lo,hi in [('soft',lambda q:one(u/(1+q*abs(u)),u),.025,12),('curve',lambda q:one(u,u*u,u*u/(1+q*u*u)),.02,8),('cos',lambda q:one(np.cos(q*u)),.1,7),('tanh',lambda q:one(np.tanh(q*u),u),.05,12),('sine',lambda q:one(np.sin(q*u)),.1,7),('cyc+tanh',lambda q:one(np.tanh(q*cyc),cyc),.05,12)]])
