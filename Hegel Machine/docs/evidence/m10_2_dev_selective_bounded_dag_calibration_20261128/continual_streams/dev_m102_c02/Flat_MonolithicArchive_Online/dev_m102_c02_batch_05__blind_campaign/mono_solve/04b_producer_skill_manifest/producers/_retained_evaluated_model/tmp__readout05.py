import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from graph05 import *
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
U,V=s['panel_b']['u'],s['panel_b']['v']
def splineop(knots, ridge):
 def make(a):
  v=CubicSpline(knots,np.eye(len(knots)),axis=0,bc_type=((1,np.zeros(len(knots))),'natural'))(abs(a))
  return v[...,1:]*np.sign(a)[...,None]
 hu,hz=make(U),make(z); D=np.diff(np.eye(len(knots)),n=2,axis=0)[:,1:]; p=np.interp(knots[1:],np.sort(abs(U).ravel()),np.sort(abs(U).ravel())**2.5)
 for _ in range(250):
  H=A(hu@p); C=F(H,V); Q=np.einsum('ndk,dm->nmk',hu,C[1:]).reshape(V.size,-1); old=p.copy(); p=F(np.r_[Q,ridge*D],np.r_[(V-C[0]).ravel(),np.zeros(len(D))]);p/=np.linalg.norm(p)
  if np.linalg.norm(old-p)<1e-11:break
 return A(hz@p)@C,np.std(A(hu@p)@C-V),p/p[-1]
for k in [7,10,14,18]:
 d,e,p=splineop(np.linspace(0,1.25,k),.012);w['spl'+str(k)]=d;print('spline',k,e,p,flush=True)

def model(d):
 a=A(d); q=d[:,0]*d[:,2]+d[:,1]*d[:,3]; return np.c_[a,q]
def evalf(d):
 h=model(d); j=d.shape[1]; ts=[]
 lin=lambda p:p@F(p[:52],y[:52]); ir=np.arange(72)
 cub=[(i,i,i) for i in range(j)]
 pro=list(combinations_with_replacement(range(j),3)); quad=list(combinations_with_replacement(range(j),2)); idx=[np.arange(j)];
 for tag,fn in [('signed square',lambda d:d*abs(d)),('signed power 1.25',lambda d:d*abs(d)**.25),('pure3',lambda d:d**3),('rational',lambda d:d**3/(1+.22*d*d)),('opposite pair cube',lambda d:d*(d[:,[2,3,0,1]]**2)),('adjacent pair cube',lambda d:d*(np.roll(d,1,axis=1)**2)),('distinct',lambda d:prod(d,[p for p in pro if len(set(p))==3])),('all cubic',lambda d:prod(d,pro)),('all quadratic',lambda d:prod(d,quad)),('z',lambda d:z)]:
  f=fn(d); ts.append((tag,score(np.c_[h,f])));
 for tag,fn in [('tanh',lambda a,p:np.tanh(p*a)/p),('rat',lambda a,p:a/(1+p*a*a)),('asinh',lambda a,p:np.arcsinh(p*a)/p),('signed',lambda a,p:a/(1+p*abs(a))),('cubic',lambda a,p:a+p*a**3)]:
  ans=[]; pars=[]
  for i,l in folds:
   def pr(p):
    H=np.c_[np.ones(len(d)),fn(d,p),h[:,-1]]; c=F(H[i],y[i]);return H@c
   p=minimize_scalar(lambda p:np.mean((pr(p)[i]-y[i])**2),bounds=((-0.7,0.7) if tag=='cubic' else (.001,2)),method='bounded').x;pars.append(p); ans.append(np.mean(abs(pr(p)[l]-y[l])))
  ts.append((tag+' %.3g'%np.median(pars),np.r_[ans[0],np.mean(ans[1:]),np.max(ans[1:])]))
 print('READ',[(n,np.round(e,5)) for n,e in sorted(ts,key=lambda t:t[1][1])[:9]])
 # Orthogonal matching pursuit on each fold, with and without additional aux channels.
 for usez in [False,True]:
  for same in [False, True]:
   f=np.c_[prod(d,quad),prod(d,pro)] if same else prod(d,pro)
   if usez:f=np.c_[f,z,z**3,w['c1powered'],w['c2even']]
   vals=[];sel=[]
   for i,j in folds:
    co=F(h[i],f[i]); r=f-h@co; r/=np.maximum(np.linalg.norm(r[i],axis=0),1e-10); H=h; keep=[]; e=y[i]-H[i]@F(H[i],y[i]); preds=[]
    for k in range(10):
     dr=r[i]-H[i]@F(H[i],r[i]); scores=abs(dr.T@e)/np.maximum(np.linalg.norm(dr,axis=0),1e-10); scores[keep]=-1; v=scores.argmax(); keep.append(v); H=np.c_[h,r[:,keep]]; c=F(H[i],y[i]);e=y[i]-H[i]@c;preds.append(np.mean(abs(H[j]@c-y[j])))
    vals.append(preds);sel.append(keep)
   vals=np.array(vals);print('OMP',usez,same,np.round(np.c_[vals[0],vals[1:].mean(0),vals[1:].max(0)],4).tolist(),sel[0][:6])
for name in ['bvpowered','bvmixrat','spl7','spl10','spl14','spl18']:
 d=w[name];a=A(d);print('BASE',name,score(model(d)),'traj', np.mean(abs(a[52:72]@F(a[:52],yt[:52])-yt[52:])),flush=True);evalf(d)
