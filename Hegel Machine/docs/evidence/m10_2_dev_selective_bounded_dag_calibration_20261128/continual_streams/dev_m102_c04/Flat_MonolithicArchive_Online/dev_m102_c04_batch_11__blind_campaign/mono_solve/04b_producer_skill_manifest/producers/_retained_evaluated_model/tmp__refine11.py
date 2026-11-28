exec(open('/tmp/probe11.py').read().split("a=s['panel_a']")[0])
import sys,importlib.util
spec=importlib.util.spec_from_file_location('arc','/tmp/hegel_continual_archive/dev_m102_c04_batch_00_solver.py'); arc=importlib.util.module_from_spec(spec); spec.loader.exec_module(arc)
u,v=s['panel_a']['u'],s['panel_a']['v']; good=np.max(abs(u-v),axis=1)<3
for j in range(4):
 a=u[:,j]; b=u[:,(j+1)%4]; c=u[:,(j-1)%4]; e=u[:,(j+2)%4]; print('channel',j)
 for name,cols in [('cyclic',(a,b,a*b,a**3)),('reverse',(a,c,a*c,a**3)),('both',(a,b,c,e,a*b,a*c,a*e,a**3)),('full',(u,u**3,prod(u,2),u**2))]:
  f=D(*cols)[good]; w=ls(f,v[good,j]); print(name,np.std(f@w-v[good,j]),np.round(w,5))
W=np.stack([ls(D(u[:,j],u[:,(j+1)%4],u[:,j]*u[:,(j+1)%4],u[:,j]**3)[good],v[good,j]) for j in range(4)],axis=1)
def cal(X):
 z=(X-W[0])/W[1]
 for _ in range(20):
  r=np.roll(z,-1,axis=1); err=W[0]+W[1]*z+W[2]*r+W[3]*z*r+W[4]*z**3-X; jac=np.zeros((len(z),4,4)); j=np.arange(4); jac[:,j,j]=W[1]+W[3]*r+3*W[4]*z*z; jac[:,j,(j+1)%4]=W[2]+W[3]*z; z-=np.linalg.solve(jac,err[:,:,None])[:,:,0]
 return z
u=s['panel_c']['u']; ops=[]
F={'soft':lambda x,k:x/(1+k*abs(x)),'rat':lambda x,k:x*x/(1+k*x*x),'log':lambda x,k:np.log1p(k*x*x),'cos':lambda x,k:np.cos(k*x),'square':lambda x,k:x*x,'oddrat':lambda x,k:x/(1+k*x*x)}
for key in ['v1','v2']:
 v=s['panel_c'][key]; cand=[]
 for name,fn in F.items():
  for lin in [0,1]:
   def design(x,k):return D(x,fn(x,k)) if lin else D(fn(x,k))
   def err(k):
    f=design(u,k); return (f@ls(f,v)-v).ravel()
   p=min([least_squares(err,[k],bounds=(.001,5)) for k in [.15,.8,2.2,4.]],key=lambda o:o.cost).x
   opt=least_squares(err,np.repeat(p,4),bounds=(.001,5)); cand.append((np.std(err(p)),np.std(opt.fun),name,lin,p,opt.x));
 print(key,sorted(cand,key=lambda a:a[0])[:8]); _,_,name,lin,p,k=min(cand,key=lambda a:a[0]); fn=F[name]; k=p
 def fit(x,fn=fn,k=k,lin=lin,v=v):
  def d(q):return D(q,fn(q,k)) if lin else D(fn(q,k))
  return d(x)@ls(d(u),v)
 ops.append(fit)
z=cal(tr['x']); p,r=[f(z) for f in ops]; b=s['panel_b']; q=D(z)@ls(D(b['u']),b['v']); basis=arc._temporal_basis(t,tr['traj'][:52]); scores=tr['traj']@basis
print('time residual',np.std(scores@basis.T-tr['traj']))
common,off=arc._features(p,r)
for name,f in [('common',D(common)),('common+off',D(common,off)),('common+xyz',D(common,z,prod(z,3))),('common+q',D(common,q)),('full',D(common,z,prod(z,3),off)),('common+zquad',D(common,z,z*z,prod(z,2))),('common+qpair',D(common,q,q*q,prod(q,2))),('z cubic',D(z,z*z,z**3,prod(z,2),prod(z,3)))]:
 yp=f@ls(f[:52],tr['y'][:52]); ap=f@ls(f[:52],scores[:52]); sp=ap@basis.T; print(name, np.mean(abs(yp[:52]-tr['y'][:52])),np.mean(abs(yp[52:]-tr['y'][52:])),np.mean(abs(sp[:52]-tr['traj'][:52])),np.mean(abs(sp[52:]-tr['traj'][52:])))
