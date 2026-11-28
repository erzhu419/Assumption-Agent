exec(open('/tmp/explore.py').read().split("cv(af(x)")[0])
from itertools import combinations
from scipy.optimize import differential_evolution

def identify(u,v,fs):
 out=[]
 for name,f,lo,hi in fs:
  def res(p):
   q=af(f(u,*p)); return (q@fit(q,v)-v).ravel()
  if len(lo):
   p=min((least_squares(res, np.array(lo)+(np.array(hi)-lo)*s,bounds=(lo,hi),ftol=1e-11,xtol=1e-11,gtol=1e-11) for s in (.17,.44,.77)),key=lambda o:o.cost).x
  else:p=[]
  err=np.sqrt(np.mean(res(p)**2)); out.append((err,name,p,f,fit(af(f(u,*p)),v)))
 return sorted(out,key=lambda x:x[0])
common=[('dead',lambda x,t,k:np.sign(x)*np.maximum(abs(x)-t,0)**k,[0,.3],[.9,5]),('power',lambda x,k:np.sign(x)*abs(x)**k,[.15],[6]),('tanh',lambda x,k:np.tanh(k*x),[.01],[7]),('sin',lambda x,k:np.sin(k*x),[.01],[5]),('sinh',lambda x,k:np.sinh(k*x),[.01],[5]),('lin+cubic',lambda x,k:x+k*x**3,[-3],[10]),('lin+square',lambda x,k:x+k*x*x,[-3],[3]),('lin+rational',lambda x,k:x/(1+k*x*x),[0],[9]),('lin+cos',lambda x,k,c:x+c*np.cos(k*x),[.05,-3],[6,3]),('lin+rollprod',lambda x,k:x+k*x*np.roll(x,-1,axis=1),[-3],[3]),('hard',lambda x,t,k:np.sign(x)*np.maximum(abs(x)-t,0)**k*(abs(x)>t),[0,.3],[.9,5]),('poly3',lambda x:np.c_[x,x*x,x**3],[],[]),('pairquad',lambda x:np.c_[x,x*x,*[x[:,i]*x[:,j] for i,j in combinations(range(4),2)]],[],[])]
ops={}
for pn,key in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u=s[pn]['u']; v=s[pn][key]; results=identify(u,v,common)
 print(pn,key,[(round(e,7),n,p) for e,n,p,f,c in results]);
 for g in (np.arctanh,np.arcsinh):
  def res(q):
   d=q[0]; k=q[1]; w=np.clip((v-d)*k,-.999999,.999999) if g==np.arctanh else (v-d)*k; r=g(w); h=af(u); return ((h@fit(h,r)-r)/k).ravel()
  o=least_squares(res,[0,.3],bounds=([-2,.001],[2,3])); print(g.__name__,o.x,np.sqrt(np.mean(o.fun**2)))
 for e,n,p,f,c in results[:3]:print(n,c)
 ops[key]=[(n,lambda x,p=p,f=f,c=c:af(f(x,*p))@c) for e,n,p,f,c in results[:5]]
Y=np.c_[y,T]; report=[]
for bn,B in ops['v']:
 for cn,C in ops['v1']:
  for dn,D in ops['v2']:
   vals={'z':z,'b':B(z),'c':C(z),'d':D(z),'cb':C(B(z)),'bc':B(C(z)),'db':D(B(z)),'bd':B(D(z)),'dc':D(C(z)),'cd':C(D(z)),'dcb':D(C(B(z))),'cdb':C(D(B(z))),'cbd':C(B(D(z))),'dbc':D(B(C(z))),'bdc':B(D(C(z))),'bcd':B(C(D(z))),'bc+':B(z)+C(z),'bd+':B(z)+D(z),'cd+':C(z)+D(z),'cd*':C(z)*D(z),'dcb*':C(B(z))*D(B(z))}
   for name,w in vals.items():
    for kind, q in [('lin',af(w)),('bilin',np.c_[af(w),w[:,0]*w[:,2]+w[:,1]*w[:,3]]),('quad',af(common[-1][1](w)))]:
     err=np.mean(abs(q[52:]@fit(q[:52],Y[:52])-Y[52:]),axis=0); report.append((err[0],err[1:].mean(),name,kind,bn,cn,dn))
print('best y',*sorted(set(report))[:12],sep='\n'); print('best traj',*sorted(set(report),key=lambda r:r[1])[:12],sep='\n')
