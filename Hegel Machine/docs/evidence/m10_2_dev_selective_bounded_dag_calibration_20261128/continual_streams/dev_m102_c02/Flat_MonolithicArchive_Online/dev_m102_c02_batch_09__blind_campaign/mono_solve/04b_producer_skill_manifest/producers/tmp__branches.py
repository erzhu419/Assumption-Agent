from probe import *
from itertools import combinations
u,v=s['panel_a']['u'],s['panel_a']['v']; r=np.roll(u,-1,axis=1); h=np.zeros((len(u),4,11)); j=np.arange(4); h[:,j,j]=1; h[:,j,4+j]=u; h[:,:,-3:]=np.stack((r,u*r,u**3),-1)
k=np.ones(len(u),bool)
for i in range(9):
 p=fit(h[k].reshape(-1,11),v[k].ravel()); e=np.linalg.norm(h@p-v,axis=1); k=e<max(5*np.median(e),1e-8)
print('A errors',np.sort(e)); F=lambda v: p[:4]+p[4:8]*v+p[8]*np.roll(v,-1,axis=1)+p[9]*v*np.roll(v,-1,axis=1)+p[10]*v**3
print('A inverse errors',np.sort(np.linalg.norm(F(z)-tr['x'],axis=1))[-10:]); print('A range',u.min(0),u.max(0))
B=fit(aff(s['panel_b']['u']),s['panel_b']['v']); b=aff(z)@B
u=s['panel_c']['u']; V=s['panel_c']['v1']; Q=s['panel_c']['v2'];
def f(x,p):return np.sign(x)*abs(x)**p[0]/(1+p[1]*abs(x)**p[2])
def res(p):
 H=aff(f(u,p)); return (H@fit(H,V)-V).ravel()
p=least_squares(res,[1,1,1.3],gtol=1e-12).x; C=fit(aff(f(u,p)),V)
H=aff(np.c_[u,u**2]); print('C2 coefs',fit(H,Q)); print('C2 max residual',np.max(abs(H@fit(H,Q)-Q),0)); print('C1 max residual',np.max(abs(res(p).reshape(-1,4)),0))
D={'z':z,'b':b};
for k in list(D):
 X=D[k]; D['c'+k]=aff(f(X,p))@C; D['d'+k]=aff(np.c_[X,X**2])@fit(H,Q)
Y=np.c_[tr['y'],tr['traj']];
def score(F):
 P=F[52:]@fit(F[:52],Y[:52]); e=np.mean(abs(P-Y[52:]),axis=0); r=np.std((F[:52]@fit(F[:52],Y[:52])-Y[:52]),axis=0); return np.array([e[0],e[1:].mean(),r[0],r[1:].mean()])
L=[]
for n in range(1,5):
 for keys in combinations(D,n):
  for k in [None,*keys]:
   F=aff(np.column_stack([D[x] for x in keys]+([a._quad(D[k])] if k else []))); L.append((list(score(F)),keys,k))
for i in [0,1,2,3]:
 print('best',i,*sorted(L,key=lambda q:q[0][i])[:6],sep='\n')
