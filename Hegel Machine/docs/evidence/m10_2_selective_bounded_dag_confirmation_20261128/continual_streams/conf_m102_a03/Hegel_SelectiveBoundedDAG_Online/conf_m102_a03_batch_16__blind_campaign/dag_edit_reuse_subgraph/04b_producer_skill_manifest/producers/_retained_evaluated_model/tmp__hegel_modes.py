import numpy as np,pickle,sys
from scipy.optimize import least_squares
from scipy.linalg import solve
sys.path.insert(0,'/holdout_task/_frozen_nodes');import node_00 as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];d=np.load('/tmp/hegel_public_features.npz');X,R,T,y,t=[d[k] for k in ('X','R','T','y','t')];z=X[:,:4];one=np.ones((len(z),1));q=m._quad
folds=[(np.arange(52),np.arange(52,72))]+[(np.setdiff1d(np.arange(72),j),j) for j in np.array_split(np.random.default_rng(19).permutation(72),4)]
B=-np.expm1(-np.outer(t,[.404,1.857])); V=np.linalg.lstsq(B,T.T,rcond=None)[0].T
u=s['panel_c']['u'];v1=s['panel_c']['v1'];v2=s['panel_c']['v2']; aff=lambda x:np.c_[np.ones(len(x)),x]; fit=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-10)[0]
loss=lambda h:(aff(np.tanh(h*u))@fit(aff(np.tanh(h*u)),v1)-v1).ravel();h=least_squares(loss,np.ones(1),bounds=(.01,8)).x;print('C1 profile',h,np.std(loss(h))); c1=aff(np.tanh(h*z))@fit(aff(np.tanh(h*u)),v1); c2=np.c_[aff(z),q(z)]@fit(np.c_[aff(u),q(u)],v2)
b,hb=m._branch(s['panel_b'],z); w=z-.4*np.sin(hb[0]*z); ba=m._branch(s['panel_b'],z,'sinatan')[0]; print('summary',s['train']['summary']);
for name,D in {'X':X,'R':R,'z':z,'w':w,'ba':ba,'C1':c1,'C2':c2,'C1+C2':np.c_[c1,c2],'w+C2':np.c_[w,c2],'ba+C2':np.c_[ba,c2]}.items():
 if name not in ('X','R'): D=np.c_[one,D,q(D)]
 print('sv residual',name,np.round(np.sqrt(np.mean((V-D@fit(D,V))**2,0)),4))

def reg(A,Y,p,w=None):
 if w is None:w=np.ones(len(A))
 return solve(A.T@(A*w[:,None])+np.diag(p),A.T@(Y*w[:,None]),assume_a='pos')
def scores(D,parts,kind,Y):
 out=[]
 for a,b in folds:
  S=np.maximum(np.sqrt(np.mean(D[a]**2,0)),1e-9); A=D[a]/S; J=Y[a]; p=np.asarray(parts)+1e-9
  W=reg(A,J,p)
  if kind:
   for fac in (8,4,2,1,1,1):
    for it in range(8):
     E=A@W-J
     e=np.sqrt(np.mean(E**2,axis=1)); tau=fac*kind
     wt=1/(1+(e/tau)**2)
     W=reg(A,J,p,wt)
  e=D[b]/S@W-Y[b]
  out.append(np.mean(np.abs(e@B.T if Y.shape[1]==2 else e)))
 return np.array(out)
rr=[]
for name,D in [('X',X),('Wdiag',np.c_[z,one,z*z,w*w,w]),('C1',np.c_[one,c1,q(c1)]),('C2',np.c_[one,c2,q(c2)]),('C12',np.c_[one,c1,c2,q(np.c_[c1,c2])]),('C12add',np.c_[one,c1,c2,q(c1),q(c2)]),('wC1',np.c_[one,w,c1,q(w),q(c1)]),('wC2',np.c_[one,w,c2,q(w),q(c2)]),('baC2',np.c_[one,ba,c2,q(ba),q(c2)])]:
 for l in (.0001,.002,.01,.03,.1,.3):
  for robust in (0,.008,.018,.035,.07):
   e=scores(D,np.full(D.shape[1],l),robust,V);rr.append((e[0]+e[1:].mean()+.5*max(e),name,l,robust,np.round(e,4)))
for x in sorted(rr,key=lambda x:x[0])[:16]:print(x)
rr=[]
for main in (.0001,.001,.01,.03,.1):
 for quad in (.00001,.001,.03,.3,3):
  for harm in (.001,.03,.3,3,300):
   for robust in (0,.006,.015,.03):
    e=scores(R,np.r_[np.full(4,main),np.full(10,quad),1e-7,np.full(10,quad),np.full(16,harm),np.full(4,main)],robust,y[:,None]);rr.append((e[0]+e[1:].mean()+.5*max(e),main,quad,harm,robust,np.round(e,4)))
for x in sorted(rr,key=lambda x:x[0])[:8]:print('read',x)
