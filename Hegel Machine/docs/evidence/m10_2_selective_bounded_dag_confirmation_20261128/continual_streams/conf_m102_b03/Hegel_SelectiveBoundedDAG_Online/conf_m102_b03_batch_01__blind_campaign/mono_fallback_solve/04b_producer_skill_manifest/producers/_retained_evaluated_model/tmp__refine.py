exec(open('/tmp/dag.py').read().split('ops=')[0])
from itertools import combinations
n=52; a,b=Q(z),R(z); vv=np.c_[a,b]; U,V=np.triu_indices(8); uv=vv[:,U]*vv[:,V]; X=ones(vv)
def check(f,t):
 p=f@ls(f[:n],t[:n]);e=abs(p-t);return e[:n].mean(0),e[n:].mean(0)
for name,w in [('plain',X),('diag',np.c_[X,a*b]),('dot',np.c_[X,(a*b).sum(1)]),('cross',np.c_[X,(a[:,:,None]*b[:,None,:]).reshape(-1,16)]),('quadratic',np.c_[X,uv])]:
 for v in [np.zeros((len(y),0)),ts@S.T,ts]:print('read',name,v.shape[1],check(np.c_[w,v],y))
 print('spat',name,check(w,ts)[1])
print('times',T,'meta',s['telemetry_vec'],s['train']['summary']);print('svd',np.linalg.svd(ts,compute_uv=False)); 
def timefit(fun,start):
 def e(p):
  h=fun(p);return (h@ls(h,ts[:n].T)-ts[:n].T).ravel()
 r=least_squares(e,start,bounds=(0.001,30),max_nfev=160); h=fun(r.x);return np.mean(abs(r.fun)),np.mean(abs(h@ls(h,ts[n:].T)-ts[n:].T)),r.x,h
r=lambda p:-np.expm1(-p*T)
funs=[('rise',lambda p:p),('sin',lambda p:np.sin(p*T)),('dampedsine',lambda p:np.sin(p[0]*T)*np.exp(-p[1]*T)),('cos',lambda p:1-np.cos(p*T)),('dexp',lambda p:T*np.exp(-p*T)),('dexp2',lambda p:T*T*np.exp(-p*T)),('square',lambda p:r(p)**2),('linear',lambda p:T),('expsin',lambda p:r(p[0])*np.sin(p[1]*T))];out=[]
for name,f in funs:
 for q0 in ([.35,1.7,1.2],[.35,1.7,3.5],[1.7,4.,1.5],[.35,1.7,6.]):
  p0=list(q0)+([.5] if name in ['dampedsine','expsin'] else []);fun=lambda p:np.c_[r(p[0]),r(p[1]),f(p[2:] if len(p)>3 else p[2]) if name!='rise' else r(p[2])];out.append((*timefit(fun,p0),name))
for a0,b0,p,h0,nm in sorted(out,key=lambda v:v[0])[:10]:print('time',a0,b0,p,nm,'read',check(np.c_[X,ts@np.linalg.pinv(h0.T)],y))
# Determine which components of QR separately survive controlling the time modes.
F=np.c_[X,ts@S.T]; e=y-F@ls(F[:n],y[:n]); print('read residual and coeff',np.quantile(e,[0,.1,.5,.9,1]),ls(F[:n],y[:n]));
for cat,w in [('quadratic',uv),('z',z),('threeprod',np.array([z[:,j].prod(1) for j in combinations(range(4),3)]).T),('dots',np.c_[np.sum(a*b,1),np.sum(a*z,1),np.sum(b*z,1)])]:
 ww=w-F@ls(F[:n],w[:n]); score=abs(e[:n]@ww[:n])/np.linalg.norm(ww[:n],axis=0); print(cat,[(i,check(np.c_[F,w[:,i]],y)) for i in np.argsort(score)[-5:][::-1]])
