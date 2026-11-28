import numpy as np,pickle,itertools as it
from scipy.optimize import least_squares,minimize_scalar,minimize
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; A,B,C=[s['panel_'+k] for k in 'abc']; x=s['train']['x']; y=s['train']['y']; T=s['train']['traj']; t=s['eval_times']; n=len(x)
def fit(d,y): return np.linalg.lstsq(d,y,rcond=1e-9)[0]
def aff(x): return np.column_stack((np.ones(len(x)),x))
def cross(x): return np.column_stack((np.ones(len(x)),x[:,0]*x[:,1],x[:,0]*x[:,3],x[:,1]*x[:,2],x[:,2]*x[:,3]))
a=np.vstack([np.zeros(4),np.ones(4)])
for _ in range(12):
 r=A['v']-a[0]-A['u']*a[1]; w=1/np.sqrt(1+(r*r).sum(1)/.02**2)
 a=np.array([fit(aff(A['u'][:,j:j+1])*w[:,None],A['v'][:,j]*w) for j in range(4)]).T
print('A',a)
bmods={'cubicrat':(lambda z,p:z**3/(1+p[0]*z*z),[.5]),'powerrat':(lambda z,p:z**3/(1+p[0]*z*z)**p[1],[.5,1]),'sine':(lambda z,p:z+p[1]*np.sin(p[0]*z),[2.8,-.4]),'tanh':(lambda z,p:z+p[1]*np.tanh(p[0]*z),[1.4,-.8]),'rat':(lambda z,p:z+p[1]*z/(1+p[0]*z*z),[.7,-1]),'sqrt':(lambda z,p:z+p[1]*z/np.sqrt(1+p[0]*z*z),[1.4,-1]),'atan':(lambda z,p:z+p[1]*np.arctan(p[0]*z),[1.4,-.8]),'exp':(lambda z,p:z+p[1]*z*np.exp(-p[0]*z*z),[.5,-1]),'poly':(lambda z,p:z**3+p[0]*z+p[1]*z**5,[0,-.3])}
cmods={'quad':(lambda z,p:z*z,[]),'rational':(lambda z,p:z*z/(1+p[0]*z*z),[.02]),'cos':(lambda z,p:np.cos(p[0]*z),[.5]),'tanh2':(lambda z,p:np.tanh(p[0]*z)**2,[.5]),'quartic':(lambda z,p:z*z+p[0]*z**4,[.03]),'abs':(lambda z,p:np.sqrt(1+p[0]*z*z),[.1]),'cube':(lambda z,p:z*z+p[0]*z**3,[0.01])}
def choose(panel,key,mods,linear):
 z=panel['u']; v=panel[key]; rows=[]
 for k,(f,p) in mods.items():
  def d(x,p):return aff(np.column_stack((x,f(x,p))) if linear else f(x,p))
  def r(p):
   D=d(z,p);return (D@fit(D,v)-v).ravel()
  pp=least_squares(r,p,max_nfev=200).x if p else np.array([]); e=np.std(r(pp)); print(key,k,pp,e); rows.append((e,k,pp,d,fit(d(z,pp),v)))
 e,k,p,_,w=min(rows,key=lambda q:q[0]); f=mods[k][0];return lambda z:aff(np.column_stack((z,f(z,p))) if linear else f(z,p))@w
b=choose(B,'v',bmods,False); h=choose(C,'v2',cmods,True); c1=fit(cross(C['u']),C['v1']); print('time',t,'ranges',x.min(0),x.max(0));
for mode in ('exp','discrete','expshift'):
 def curve(k):return -np.expm1(-k*t) if mode!='discrete' else -np.expm1(-k*np.arange(1,len(t)+1))
 def row(z):
  def d(k):return aff(curve(k)[:,None]) if mode=='expshift' else curve(k)[:,None]
  def loss(k):
   D=d(k);r=z-D@fit(D,z);return r@r
  k=minimize_scalar(loss,bounds=(.003,1.5),method='bounded').x;return [k,loss(k),*fit(d(k),z)]
 pp=np.array([row(z) for z in T]); print(mode,'rms',np.sqrt(pp[:,1].mean()/len(t)),np.round(pp[:6],4))
folds=[np.arange(n-20,n)]+[i for i in np.array_split(np.arange(n),4)]
Y=np.column_stack((y,T)); kr=[]; rr=[]
for coord,z in enumerate([x,(x-a[0])/a[1],x*a[1]+a[0]]):
 bs,cs,hs=b(z),cross(z)@c1,h(z); print('feature ranges',coord,[np.std(v,0).round(2) for v in [bs,cs,hs]])
 P={'x':z,'b':bs,'c':cs,'h':hs,'u':np.column_stack((z,bs,cs,hs))}
 for names in ['x','xb','xc','xh','bc','bch','xbch']:
  U=np.column_stack([P[k] for k in names]); U=(U-U.mean(0))/U.std(0); d2=((U[:,None]-U)**2).sum(-1)/U.shape[1]; xx=U@U.T/U.shape[1];
  for ker in ['rbf','matern','rq','poly']:
   for l in [.35,.6,1,1.7,3,5]:
    d=np.sqrt(d2)/l
    K={'rbf':np.exp(-.5*d*d),'matern':(1+np.sqrt(5)*d+5*d*d/3)*np.exp(-np.sqrt(5)*d),'rq':1/(1+.5*d*d),'poly':(1+xx/l)**4}[ker]+1+xx
    for reg in [.00003,.001,.015,.1]:
     er=[]
     for v in folds:
      i=np.setdiff1d(np.arange(n),v); R=abs(K[np.ix_(v,i)]@np.linalg.solve(K[np.ix_(i,i)]+reg*np.eye(len(i)),Y[i])-Y[v]); er.append([R[:,0].mean(),R[:,1:].mean()])
     kr.append((np.max(er),np.max(er[0]),coord,names,ker,l,reg,np.round(er,4).tolist()))
 # Compact cubic grammar: grids of diagonal components, cross-panel pairings, and square/pair interactions.
 def inter(X,Z):return (X[:,:,None]*Z[:,None,:]).reshape(n,-1)
 q=np.column_stack([z[:,i]*z[:,j] for i,j in it.combinations(range(4),2)]); tri=np.column_stack([z[:,i]*z[:,j]*z[:,k] for i,j,k in it.combinations(range(4),3)]); 
 D0=aff(np.column_stack((z,bs,cs,hs))); libs={'diags':np.column_stack([v for u in [z,bs,cs,hs] for v in [u*u,u**3]]),'raw2':q,'triple':tri,'xc':inter(z,cs),'xb':inter(z,bs),'xh':inter(z,hs),'bc':inter(bs,cs),'bh':inter(bs,hs),'ch':inter(cs,hs),'xc2':inter(z*z,cs),'x2b':inter(z*z,bs),'cx2':inter(z,cs*cs),'cc':inter(cs,cs),'hh':inter(hs,hs),'bb':inter(bs,bs),'osc':np.column_stack([f(j*u) for u in [z,bs,cs,hs] for f in (np.sin,np.cos) for j in (1,2,3,4)]),'xyz':inter(z, q)}
 for tag,keys in [('base',()),*[(k,(k,)) for k in libs],('xall',('xc','xb','xh','raw2','triple')),('pair',('xc','xb','xh','bc','bh','ch')),('all',tuple(libs)),('structure',('xc','diags','triple','xyz','osc'))]:
  D=np.column_stack([D0,*[libs[k] for k in keys]]); D=D/np.maximum(np.std(D,0),.1); DD=D@D.T
  for rg in [.0001,.003,.03,.2,1,5]:
   er=[]
   for v in folds:
    i=np.setdiff1d(np.arange(n),v); R=abs(DD[np.ix_(v,i)]@np.linalg.solve(DD[np.ix_(i,i)]+rg*np.eye(len(i)),Y[i])-Y[v]);er.append([R[:,0].mean(),R[:,1:].mean()]);
   rr.append((np.max(er),np.max(er[0]),coord,tag,rg,np.round(er,4).tolist()))
print('KERNEL',*sorted(kr)[:5],sep='\n');print('K pub',*sorted(kr,key=lambda a:a[1])[:5],sep='\n');print('RIDGE',*sorted(rr)[:7],sep='\n');print('R pub',*sorted(rr,key=lambda a:a[1])[:5],sep='\n')
