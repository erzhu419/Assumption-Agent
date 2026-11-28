from probe import *
from itertools import combinations,product
from scipy.optimize import least_squares

def cal(x):
 u,v=s['panel_a']['u'],s['panel_a']['v']; k=np.linalg.norm(v,axis=1)<4
 def F(x,p):return aff(np.c_[x,np.sign(x)*abs(x)**p[0]])
 def res(p):
  H=F(u[k],p);return (H@fit(H,v[k])-v[k]).ravel()
 p=least_squares(res,[2.],gtol=1e-13).x; C=fit(F(u[k],p),v[k]);print('cal',p,np.std(res(p)),C.round(4))
 return np.array([least_squares(lambda z:(F(z[None],p)@C)[0]-r,r,gtol=1e-12).x for r in x])
z=cal(tr['x']); b=s['panel_b']; B=fit(aff(b['u']),b['v']); c=s['panel_c']; u=c['u']
def F(x,p):return np.sign(x)*abs(x)**p[0]/(1+p[1]*abs(x)**p[2])
def G(x,p):return np.logaddexp(0,p[0]*x+p[1])
def pp(f,v,p0):
 def err(p):
  h=aff(f(u,p));return (h@fit(h,v)-v).ravel()
 p=least_squares(err,p0,gtol=1e-12).x; M=fit(aff(f(u,p)),v);return lambda x:aff(f(x,p))@M
C=pp(F,c['v1'],[.97,.94,1.26]); D=pp(G,c['v2'],[2.2,0]); Bf=lambda x:aff(x)@B
ops={'b':Bf,'c':C,'d':D}; X={'z':z}
for k in range(3):
 for nm,x in list(X.items()):
  if len(nm)!=k+1:continue
  for key,fn in ops.items():
   if key not in nm:X[nm+key]=fn(x)
Y=np.c_[tr['y'],tr['traj']]; n=52

def errs(H,Y=Y):
 R=H@fit(H[:n],Y[:n])-Y; e=np.abs(R[n:]).mean(0); q=np.sqrt((R[:n]**2).mean(0));return [e[0],e[1:].mean(),q[0],q[1:].mean()]

def opp(x):return x[:,0]*x[:,2]+x[:,1]*x[:,3]
def pair(x,y):return x*np.roll(y,-1,axis=1)
if __name__=='__main__':
 L=[]
 for nm,x in X.items():
  for k,f in enumerate([x,np.c_[x,opp(x)],np.c_[x,a._quad(x)],np.c_[x,x*abs(x)],np.c_[x,x**3],np.c_[x,x/(1+.22*x*x)]]):L.append((errs(aff(f)),nm,k))
 for (nm,x),(nj,y) in combinations(X.items(),2):
  for k,f in enumerate([np.c_[x,y],np.c_[x,y,opp(x),opp(y)],np.c_[x,y,x*y],np.c_[x,y,pair(x,y)],np.c_[x,y,x/(1+.22*y*y)],np.c_[x,y,x*np.tanh(2.15*y)],np.c_[x,y,opp(x),opp(y),x*y],np.c_[x,y,a._quad(x),a._quad(y)]]):L.append((errs(aff(f)),nm,nj,k))
 for i in range(4):print('rank',i,*[(np.round(r[0],6),r[1:]) for r in sorted(L,key=lambda r:r[0][i])[:12]],sep='\n')
 print('z bounds',z.min(0),z.max(0),'t',t,'Y range',Y.min(0),Y.max(0))
