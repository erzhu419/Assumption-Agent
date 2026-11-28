exec(open('/tmp/explore.py').read().split('operators={}')[0])
import importlib.util,glob
spec=importlib.util.spec_from_file_location('old',glob.glob('/tmp/hegel_continual_archive/*03_solver.py')[0]);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
tr=s['train']; X=np.vstack([tr['x'],s['test']['x']]); z=old._calibrate(s['panel_a'],X); print('z range',z.min(0),z.max(0)); print('z std',z[:72].std(0));
pa=s['panel_a']; uu=pa['u']; vv=pa['v']; a=I(uu); keep=np.argsort(np.linalg.norm(a@ls(a,vv)-vv,axis=1))[:-3]; u=uu[keep];v=vv[keep];r=np.roll(u,-1,axis=1); print('cal coeff',np.stack([ls(np.c_[np.ones(len(u)),u[:,i],r[:,i],u[:,i]*r[:,i],u[:,i]**3],v[:,i]) for i in range(4)]));
p=s['panel_b']; u=p['u'];v=p['v'];
bs={}; fs=dict(erf=erf,rational=lambda x:x/(1+x*x), tanh=np.tanh, sin=np.sin, atan=np.arctan, asinh=np.arcsinh, cubic=lambda x:x**3, quintic=lambda x:x**5, gauss=lambda x:x*np.exp(-x*x), sincos=lambda x:np.sin(x)+.2*np.cos(x));
for name,fn in fs.items():
 for extra in ('','sq','cyc','bias'):
  def D(w,q,fn=fn,extra=extra):
   p=fn(w*np.exp(q[:4])+(q[4:] if extra=='bias' else 0));k={'sq':w*w,'cyc':w*np.roll(w,-1,axis=1)};return np.c_[I(w),p,k[extra]] if extra in k else np.c_[I(w),p];
  def obj(q):d=D(u,q);return (d@ls(d,v)-v).ravel()
  f=min([least_squares(obj,np.r_[np.full(4,np.log(g)),np.zeros(4) if extra=='bias' else []],bounds=(-3,3),max_nfev=70) for g in (.6,1.4,3)],key=lambda f:f.cost);d=D(u,f.x);r=d@ls(d,v)-v; loo=np.mean((r/(1-np.diag(d@np.linalg.pinv(d)))[:,None])**2)**.5; print('Bfit',name,extra,np.sqrt(np.mean(r*r)),loo,np.exp(f.x[:4]),f.x[4:]);
  if extra=='' or loo<.006: bs[name+extra]=lambda w,D=D,q=f.x,c=ls(d,v):D(w,q)@c
bs['aff']=lambda w:I(w)@ls(I(u),v)
p=s['panel_c'];ua=p['u'];D1=lambda w:I(w*np.roll(w,-1,axis=1)); c1=ls(D1(ua),p['v1']);C1=lambda w:D1(w)@c1
D2=lambda w:np.c_[I(w),np.log1p(.365*w*w)]; c2=ls(D2(ua),p['v2']);C2=lambda w:D2(w)@c2
Y=np.c_[tr['y'],tr['traj']];n=52; valid=np.arange(n,72);
def err(d,k=1e-6):
 d=np.concatenate([np.ones((len(d),1)),(d-d[:n].mean(0))/np.maximum(d[:n].std(0),1e-9)],axis=1); a=d[:n]; pr=d[valid]@np.linalg.solve(a.T@a+k*np.eye(d.shape[1]), a.T@Y[:n]); return np.mean(np.abs(pr-Y[valid]),axis=0)[0],np.mean(np.abs(pr[:,1:]-Y[valid,1:]))
def cross(a,b):return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
rank=[]
for key,B in bs.items():
 nodes={'z':z,'b':B(z),'c':C1(z),'d':C2(z),'bc':C1(B(z)),'bd':C2(B(z)),'cb':B(C1(z)),'db':B(C2(z)),'cd':C2(C1(z)),'dc':C1(C2(z)),'bcd':C2(C1(B(z))),'bdc':C1(C2(B(z))),'cdb':B(C2(C1(z))),'dcb':B(C1(C2(z))),'cbd':C2(B(C1(z))),'dbc':C1(B(C2(z)))}
 for names in itertools.combinations(nodes,2):
  a,b=[nodes[h] for h in names];features={'lin':np.c_[a,b],'dot':np.c_[a,b,(a*b).sum(1)],'diag':np.c_[a,b,a*b],'cross':np.c_[a,b,cross(a,b)],'poly2':np.c_[a,b,poly(a,2),poly(b,2),cross(a,b)]}
  for fn,d in features.items():rank.append((*err(d,.001),key,names,fn))
  if 'z' not in names:
   for fn in ('lin','dot','diag','cross'):
    d=np.c_[z,features[fn]];rank.append((*err(d,.001),key,('z',)+names,fn))
print('best y'); print(*sorted(rank)[:18],sep='\n');print('best t'); print(*sorted(rank,key=lambda a:a[1])[:18],sep='\n')
