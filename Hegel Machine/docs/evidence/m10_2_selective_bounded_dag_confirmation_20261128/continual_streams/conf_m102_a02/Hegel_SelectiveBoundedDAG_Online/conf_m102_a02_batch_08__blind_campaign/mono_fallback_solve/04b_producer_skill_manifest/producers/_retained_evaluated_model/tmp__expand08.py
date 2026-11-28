exec(open('/tmp/read08.py').read().split("print('C range'")[0]); from scipy.optimize import least_squares
h=q[:,2]; N=np.ones(len(q)); X=np.c_[N,b]; rng=np.random.default_rng(412); folds=folds+[(np.setdiff1d(np.arange(52),j),j) for j in np.array_split(rng.permutation(52),4)];
def run(fun,p0=[],bounds=(-np.inf,np.inf),reg=0):
 es=[]; ps=[]; rs=[]; ws=[]
 for i,j in folds:
  def fit(p):
   f=fun(p)[i]; w=ls(f,y[i]);return f@w-y[i],w
  p=least_squares(lambda p:np.r_[fit(p)[0],np.sqrt(reg)*(p-np.array(p0))],p0,bounds=bounds,max_nfev=150).x if len(p0) else []
  r,w=fit(p);es.append(err(fun(p)[j]@w-y[j]));ps.append(p);rs.append(np.sqrt(np.mean(r*r)));ws.append(w)
 e=np.array(es);return np.mean(e[1:]),np.max(e[1:]),e[0],rs[0],fmt(ps[0]),fmt(ws[0])
R=[]
def add(k,f,p0=[],bd=(-np.inf,np.inf),reg=0):R.append((k,run(f,p0,bd,reg)))
for name,H in [('q',q),('x',x),('b',b)]:
 for kind in ['shift','tilt','squared','anis']:
  for mode in ['B','X','all','mixed','quad','mixed3']:
   a0=[.3,0] if kind=='shift' else ([.3] if kind=='squared' else [.3,0,0,0,0] if kind=='anis' else [.3,0,0,0]);bd=([0]+[-.7]*(len(a0)-1),[2]+[.7]*(len(a0)-1))
   def f(p):
    hh=H[:,2]; t=H[:,[0,1,3]]
    g=p[0]*(hh+(p[1] if kind=='shift' else t@p[1:] if kind=='tilt' else 0))**2
    if kind=='anis':g+=np.sum((t*p[1:4])**2,axis=1)+p[4]*hh
    rr=1/np.maximum(.25,1+g);rr=rr[:,None]
    if mode=='B':return np.c_[N,C,b*rr]
    if mode=='X':return np.c_[C,X*rr]
    if mode=='all':return np.c_[N,np.c_[b,C]*rr]
    if mode=='mixed':return np.c_[base,b*(hh**2*rr[:,0])[:,None]]
    if mode=='quad':return np.c_[N,C,b*rr,b*(hh*rr[:,0])[:,None]]
    return np.c_[base,b*(hh**2*rr[:,0])[:,None],b*hh[:,None]]
   add((name,kind,mode),f,a0,bd,.01 if kind in ['tilt','anis'] else 0)
for deg in (1,2,3,4):
 for mode in ['b','X','full','partial']:
  def f(p):
   hh=h[:,None]; d=np.c_[*[b*hh**k for k in range(1,deg+1)]]
   if mode=='b':return np.c_[base,d]
   if mode=='X':return np.c_[base,d,*[hh**k for k in range(1,deg+1)]]
   if mode=='full':return np.c_[full,d]
   return np.c_[base,b*hh**2,b[:,3]*h]
  add(('poly',deg,mode),f)
for k,v in sorted(R,key=lambda r:r[1][0])[:32]:print(k, tuple(np.round(v[:4],5)),v[4],v[5]);
