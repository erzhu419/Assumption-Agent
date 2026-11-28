import contextlib,io
with contextlib.redirect_stdout(io.StringIO()): exec(open('/tmp/structure.py').read())
from scipy.optimize import minimize_scalar
from itertools import combinations_with_replacement

def link(u,k,m):
 if m=='tanh':return np.tanh(k*u)
 if m=='softplus':return np.logaddexp(-k*u,k*u)
 if m=='sin':return np.sin(k*u)
 if m=='gauss':return np.exp(-(k*u)**2)
 if m=='rational':return 1/(1+(k*u)**2)
 if m=='odd_rational':return u/(1+(k*u)**2)
 if m=='atan':return np.arctan(k*u)

def panel(u,v,m,idx):
 def design(k,X):return aff(np.c_[X,link(X,k,m)])
 def f(k):
  d=design(k,u[idx]);return np.mean((d@fit(d,v[idx])-v[idx])**2)
 r=minimize_scalar(f,bounds=(.1,4.5),method='bounded'); k=r.x; w=fit(design(k,u[idx]),v[idx]);return k,w,lambda X:design(k,X)@w
ops=[]
for V in (R,S):
 results=[]
 for m in ('tanh','softplus','sin','gauss','rational','odd_rational','atan'):
  k,w,g=panel(U,V,m,np.arange(18)); results.append((np.mean(abs(g(U[18:])-V[18:])),m,k))
 print('panel 56-equivalent split',sorted(results)); m=min(results)[1]; k,w,g=panel(U,V,m,np.arange(len(U))); print('full panel fit',m,k,'rms',np.std(g(U)-V),'coefficients',w); ops.append(g)

# Fit time modes independently of prediction, then inspect structural features.
def td(p):return -np.expm1(-ts[:,None]*p)
for n in (56,72):
 r=least_squares(lambda p:(td(p)@fit(td(p),T[:n].T)-T[:n].T).ravel(),[.4,1.8],bounds=([.05,.8],[.8,4])); print('time fit',n,r.x)
D=td(r.x); C=fit(D,T.T).T; Y=np.c_[C,y]
spaces={'z':z,'Bz':z@B+b,'invBz':(z-b)@np.linalg.inv(B),'x':x}
for k,W in list(spaces.items()):
 spaces[k+'C1']=ops[0](W); spaces[k+'C2']=ops[1](W);spaces[k+'both']=np.c_[W,ops[0](W),ops[1](W)]
for name,W in spaces.items():
 E=aff(W); P=E[56:]@fit(E[:56],Y[:56]); print(name,'linear MAE',np.mean(abs(P-Y[56:]),0),'traj',np.max(np.mean(abs(P[:,:2]@D.T-T[56:]),0)))
 if W.shape[1]>4:continue
 results=[]
 for m in ('tanh','softplus','sin','gauss','rational','odd_rational','atan'):
  for k in (.5,1,1.5,2,2.5):
   E=aff(np.c_[W,link(W,k,m)]); P=E[56:]@fit(E[:56],Y[:56]); results.append((m,k,*np.mean(abs(P-Y[56:]),0)))
 print('best per output', [min(results,key=lambda a:a[j]) for j in (2,3,4)])
