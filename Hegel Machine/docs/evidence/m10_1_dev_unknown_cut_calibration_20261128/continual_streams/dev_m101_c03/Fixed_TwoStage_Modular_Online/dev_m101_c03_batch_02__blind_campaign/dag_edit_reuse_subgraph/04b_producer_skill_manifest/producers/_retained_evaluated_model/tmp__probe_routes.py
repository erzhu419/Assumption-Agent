exec(open('/tmp/probe_graph.py').read().split('opts=[]')[0])
from scipy.optimize import minimize_scalar
x=s['train']['x']; va=a['v']; A=ls(aug(a['u']),va)
for _ in range(20):
 r=np.linalg.norm(va-aug(a['u'])@A,axis=1); w=1/np.sqrt(1+(r/max(.0004,.8*np.median(r)))**2); A=ls(aug(a['u'])*w[:,None],va*w[:,None])
B=ls(aug(b['u']),b['v']); C2=ls(aug(np.cos(2.4*u)),c['v2'])
k=minimize_scalar(lambda k:np.linalg.norm(aug(np.c_[u,u/(1+k*abs(u))])@ls(aug(np.c_[u,u/(1+k*abs(u))]),v)-v),bounds=(.2,2),method='bounded').x
vf=lambda z:aug(np.c_[z,z/(1+k*abs(z))]); C1=ls(vf(u),v)
print('A/B maps:',A,B,sep='\n'); print('panel distribution:',[(p,np.round(s[p]['u'].min(0),3),np.round(s[p]['u'].max(0),3)) for p in ['panel_a','panel_b','panel_c']]);print('x percentiles:',np.round(np.quantile(x,[0,.025,.1,.5,.9,.975,1],axis=0),3)); print('sample x | y',np.c_[x,s['train']['y']][:8]);print('telemetry and summary',s['telemetry_vec'],s['train']['summary'])
aff=lambda z,F:aug(z)@F
inv=lambda z,F:np.linalg.solve(F[1:].T,(z-F[0]).T).T
sources={'x':x}
for nm,z in list(sources.items()):
 for an,az in [(nm,z),(nm+'a+',aff(z,A)),(nm+'a-',inv(z,A))]:
  sources[an]=az; sources[an+'b+']=aff(az,B);sources[an+'b-']=inv(az,B)
for nm,z in list(sources.items()):
 sources[nm+'half']=z*.5; sources[nm+'1.5']=z*1.5; sources[nm+'tanh']=np.tanh(z); sources[nm+'soft']=z/(1+abs(z)); sources[nm+'sin']=np.sin(z); sources[nm+'clip']=np.clip(z,-1.,1.); sources[nm+'asinh']=np.arcsinh(z)
for nm,z in list(sources.items())[:9]:
 for lm,w in list(sources.items())[:9]:
  if nm<lm:sources[nm+' avg '+lm]=(z+w)*.5; sources[nm+' sum '+lm]=z+w; sources[nm+' diff '+lm]=z-w; sources[lm+' diff '+nm]=w-z
T=lambda r:-np.expm1(-s['eval_times'][:,None]*r)
traj=s['train']['traj']; y=s['train']['y']; R=least_squares(lambda r:(T(r)@ls(T(r),traj.T)-traj.T).ravel(),[1.8,.4],bounds=([1.,.01],[10.,1.])).x
P=T(R); amps=ls(P,traj.T).T; target=np.c_[y,amps]; i=np.arange(52); j=np.arange(52,72)
H=aug(x); coef=ls(H[i],target[i]); resid=target-H@coef
names=list(sources); Z=np.array(list(sources.values())); F=np.array([vf(z)@C1 for z in Z]); G=np.array([aug(np.cos(2.4*z))@C2 for z in Z]);

def scan(F,G,fn,gn,H,target):
 Hi=np.linalg.pinv(H[i]); Yr=target-H@(Hi@target[i]); pairs=(F@G.transpose(0,2,1)) if False else np.einsum('nrd,mrd->nmr',F,G)
 best=[]
 for mode in range(2):
  J=pairs if mode==0 else F[:,None,:,0]*G[None,:,:,1]
  Jr=J-J[...,i]@Hi.T@H.T
  idx=[0,1] if mode==0 else [2]
  coef=Jr[...,i]@Yr[i][:,idx]/np.maximum(1e-12,np.sum(Jr[...,i]**2,axis=-1))[...,None]
  err=np.mean(abs(Jr[...,j,None]*coef[...,None,:]-Yr[j][:,idx]),axis=(-1,-2))
  entries=[(round(err[ii,jj],6),fn[ii],gn[jj],coef[ii,jj]) for ii,jj in np.array(np.unravel_index(np.argsort(err,axis=None)[:4],err.shape)).T];best.append(entries)
 return best
print('linear baseline Y,fast,slow',np.mean(abs(resid[j]),axis=0))
for fn,gn,fl,gl in [('C1','C2',F,G),('C2','C1',G,F),('C1','C1',F,F),('C2','C2',G,G),('C1','raw',F,Z),('raw','C2',Z,G),('C2','raw',G,Z),('raw','C1',Z,F)]:
 print(fn,gn,scan(fl,gl,names,names,H,target))
print('single nonlinear linear-base candidates:')
for V,nm in [(F,'C1'),(G,'C2'),(Z,'raw')]:
 print(nm,sorted((float(np.mean(abs(target[j]-aug(np.c_[x,f])[j]@ls(aug(np.c_[x,f])[i],target[i])))),name) for f,name in zip(V,names))[:8])
