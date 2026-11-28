import pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True)
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-10)[0]
aff=lambda x:np.c_[np.ones(len(x)),x]
rms=lambda e:np.sqrt(np.mean(np.sum(e*e,axis=1)))
u,v=s['panel_a'].values(); good=np.linalg.norm(v-u,axis=1)<3; u,v=u[good],v[good]; a=aff(u)
for name,f in [('rat',lambda x,c:x/(1+c*x*x)),('tanh',lambda x,c:np.tanh(c*x)),('sin',lambda x,c:np.sin(c*x)),('asinh',lambda x,c:np.arcsinh(c*x))]:
 def err(c):
  d=np.c_[a,f(u,c)]; return (d@ls(d,v)-v).ravel()
 for c0 in ([.6],[.6]*4):
  o=least_squares(err,c0,bounds=(.002,5),gtol=1e-12); d=np.c_[a,f(u,o.x)]; print('A',name,o.x,rms(err(o.x).reshape(-1,4)),ls(d,v).round(3))
u,v=s['panel_b'].values(); b=ls(aff(u),v); print('B',b,'singvals',np.linalg.svd(b[1:],compute_uv=False))
u=s['panel_c']['u']; a=aff(u); pairs=[(i,j) for i in range(4) for j in range(i+1,4)]
for k in ('v1','v2'):
 v=s['panel_c'][k]; print(k)
 for coord,z in [('raw',u),('B',aff(u)@b),('Binv',(u-b[0])@np.linalg.inv(b[1:])),('Btrans',u@b[1:].T)]:
  for name,f in [('soft',lambda x,c:x/(1+c*abs(x))),('tanh',lambda x,c:np.tanh(c*x)),('rat',lambda x,c:x/(1+c*x*x)),('sqrat',lambda x,c:x*x/(1+c*x*x)),('sin',lambda x,c:np.sin(c*x))]:
   for mix in [False,True]:
    def d(c):
     t=f(z,c); return np.c_[a,t,*([z[:,i]*z[:,j] for i,j in pairs] if mix else [])]
    def er(c):
     t=d(c); return (t@ls(t,v)-v).ravel()
    o=least_squares(er,[.5],bounds=(.002,5)); print(coord,name,mix,round(o.x[0],4),round(rms(er(o.x).reshape(-1,4)),6))
 print('affine residual paired fits')
 terms=[(f'{i}{sg}{j}',u[:,i]+sg*u[:,j]) for i,j in pairs for sg in [1,-1]]
 for name,fun in [('sq',lambda z:z*z),('cube',lambda z:z**3),('sat',lambda z:z*z/(1+.4*z*z)),('odd',lambda z:z/(1+.8*z*z))]:
  l=[]
  for label,t in terms:
   d=np.c_[a,fun(t)]; l.append((round(rms(v-d@ls(d,v)),5),label))
  print(name,sorted(l)[:4])
