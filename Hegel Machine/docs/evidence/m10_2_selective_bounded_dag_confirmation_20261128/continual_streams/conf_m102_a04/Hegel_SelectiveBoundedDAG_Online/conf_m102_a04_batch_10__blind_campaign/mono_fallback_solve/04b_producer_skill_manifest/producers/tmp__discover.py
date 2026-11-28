import numpy as np,pickle,sys
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
sys.path.insert(0,'/tmp/hegel_continual_archive')
from conf_m102_a04_batch_00_solver import _align,_panel_maps
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; T=tr['traj']; y=tr['y']; t=s['eval_times']; x=tr['x']; n=52
np.set_printoptions(precision=5,suppress=True,linewidth=140)
def one(X): return np.c_[np.ones(len(X)),X]
def ls(X,Y): return np.linalg.lstsq(X,Y,rcond=1e-9)[0]
p=s['panel_a']; u,v=p['u'],p['v']; o,a=_align(p); print('diagonal A',o,a,'median',np.median(abs(v-o-u*a),0)); F=one(u); keep=np.ones(len(u),bool)
for i in range(12):
 b=ls(F[keep],v[keep]); r=np.linalg.norm(F@b-v,axis=1);keep=np.argsort(r)[:38]
b=ls(F[keep],v[keep]); print('full A',b,'sorted residual',np.sort(np.linalg.norm(F@b-v,axis=1)))
z=np.linalg.solve(b[1:].T,(x-b[0]).T).T; zd=(x-o)/a
p=s['panel_b']; B=ls(one(p['u']),p['v']); print('B',B)
def basis(q): return -np.expm1(-t[:,None]*q)
def fit(fun,init):
 def res(q):
  D=fun(q); return (D@ls(D,T[:n].T)-T[:n].T).ravel()
 q=least_squares(res,init,bounds=(.001,15),xtol=1e-12,ftol=1e-12,gtol=1e-12).x; D=fun(q); print('time',q,np.std(res(q)),np.mean(abs(D@ls(D,T[n:].T)-T[n:].T)));return D
D=fit(basis,[.4,2]);fit(lambda q:np.c_[t,-np.expm1(-t*q[0])],[1.]); fit(lambda q:-np.expm1(-t[:,None]*q)*np.exp(-t[:,None]*np.array([0,.22])),[.4,2]);print('time telemetry',np.std(basis([.22,2.15])@ls(basis([.22,2.15]),T.T)-T.T))
Y=np.c_[ls(D,T.T).T,y]; print('Y stats',Y.mean(0),Y.std(0))
G={'x':x,'zd':zd,'z':z,'Bx':one(x)@B,'Bzd':one(zd)@B,'Bz':one(z)@B}
for key in list(G):
 H=G[key]; G['C'+key]=_panel_maps(s['panel_c'],H)
def eval(F):
 P=one(F); A=P@ls(P[:n],Y[:n]); return np.mean(abs(A[:n]-Y[:n]),axis=0),np.mean(abs(A[n:]-Y[n:]),axis=0)
for key,V in G.items():
 V2=np.column_stack([V[:,i]*V[:,j] for i,j in combinations_with_replacement(range(V.shape[1]),2)]);print(key,'lin',eval(V),'quad',eval(np.c_[V,V2]));
 for fn in [np.tanh,np.sin,lambda a:a/(1+abs(a)),lambda a:a/(1+a*a),lambda a:a*a/(1+a*a)]: print(' ',eval(np.c_[V,fn(V)])[1],end='')
 print()
