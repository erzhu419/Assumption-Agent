import numpy as np,pickle,sys
from scipy.optimize import least_squares
from scipy.linalg import svdvals
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True,linewidth=140)
def aff(u):return np.c_[np.ones(len(u)),u]
ijs=[(i,j) for i in range(4) for j in range(i,4)]
def quad(u):return np.column_stack([aff(u),*[u[:,i]*u[:,j] for i,j in ijs]])
def ls(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
a,b,c=(s['panel_'+k] for k in 'abc'); u,v=a.values(); H=aff(u); keep=np.linalg.norm(H@ls(H,v)-v,axis=1)<2; m=ls(H[keep],v[keep]); B=ls(aff(b['u']),b['v']); u=c['u']; v=c['v2']; pairs=[(0,1),(0,3),(1,2),(2,3)]
def c1(z):return np.column_stack([np.ones(len(z)),*[z[:,i]*z[:,j] for i,j in pairs]])
C=ls(c1(u),c['v1']); print('C1 structured RMS',np.std(c1(u)@C-c['v1']))
for name,f in [('quadratic',lambda z,p:z+p[0]*z*z),('cubic',lambda z,p:z+p[0]*z*z+p[1]*z**3),('quartic',lambda z,p:z+p[0]*z*z+p[1]*z**3+p[2]*z**4),('sinquad',lambda z,p:np.sin(p[1]*z)/p[1]+p[0]*z*z),('rational',lambda z,p:(z+p[0]*z*z)/(1+p[1]*z*z)),('rational2',lambda z,p:z/(1+p[1]*z*z)+p[0]*z*z)]:
 n={'quadratic':1,'quartic':3}.get(name,2)
 def err(p):
  F=aff(f(u,p)); return (F@ls(F,v)-v).ravel()
 p=least_squares(err,[.45]+[.1]*(n-1),max_nfev=100).x; print(name,p,np.std(err(p))); 
 if name=='quartic':p2=p; M=ls(aff(f(u,p)),v)
def c2(z):return aff(z+p2[0]*z*z+p2[1]*z**3+p2[2]*z**4)@M
x=s['train']['x']; z=(x-m[0])/np.diag(m[1:]); t=s['eval_times']; T=s['train']['traj']; y=s['train']['y']; W=np.c_[T,y];
def bas(p):return -np.expm1(-t[:,None]*p)
def model(F,I,J,pen=0):
 def fit(A,h):
  if not pen:return ls(A,h)
  G=A.T@A+np.diag(np.r_[0,np.full(A.shape[1]-1,pen)]);return np.linalg.solve(G,A.T@h)
 def er(p):
  K=bas(p); H=F[I]@fit(F[I],T[I]); return (H@K@ls(K.T@K,K.T)-T[I]).ravel()
 p=least_squares(er,[.42,1.83],bounds=([.05,1],[1.0,4]),max_nfev=25).x
 A=fit(F[I],W[I]); K=bas(p); pred=F[J]@A; pred[:,:8]=pred[:,:8]@K@ls(K.T@K,K.T); return np.mean(abs(pred-W[J]),axis=0),p
feat={}
def add(name,F):feat[name]=F
add('fullquad',quad(z));
for key,k in [('cal',z),('raw',x),('fw',aff(x)@m),('postB',aff(z)@B),('inverseB',(z-B[0])@np.linalg.inv(B[1:]))]:
 a=c1(k)@C; b=c2(k); ab=(a[:,:,None]*b[:,None,:]).reshape(-1,16); dot=np.trace(ab.reshape(-1,4,4),axis1=1,axis2=2); linear=np.c_[aff(k),a,b]; small=np.c_[aff(k),a,b,ab[:,1],dot];
 for nm,F in [('linear',linear),('l+dot+cross',small),('l+prods',np.c_[linear,ab]),('q+prods',np.c_[quad(k),ab]),('q+dot',np.c_[quad(k),dot]),('q+dot+tanh',np.c_[quad(k),dot,np.tanh(a),np.tanh(b)]),('l+dot+tanh',np.c_[small,np.tanh(a),np.tanh(b)])]:add(key+'/'+nm,F)
I=np.arange(56); J=np.arange(56,72); folds=[(np.setdiff1d(np.arange(72),jj),jj) for jj in np.array_split(np.random.default_rng(14).permutation(72),4)]; stats=[]
for nm,F in feat.items():
 for pen in [0,.03,.3] if F.shape[1]>=14 else [0,.03]:
  e,p=model(F,I,J,pen); er=np.array([[e2[-1],e2[:-1].mean()] for i,j in folds for e2,_ in [model(F,i,j,pen)]]); stats.append((sum(er.mean(0)),nm,pen,np.r_[e[-1],e[:-1].mean()],er.mean(0),er.max(0),p))
for a,nm,p,e,em,ew,r in sorted(stats)[:23]:print(nm,p,'56/16',e,'CV mean',em,'CV worst',ew,'rates',r)
