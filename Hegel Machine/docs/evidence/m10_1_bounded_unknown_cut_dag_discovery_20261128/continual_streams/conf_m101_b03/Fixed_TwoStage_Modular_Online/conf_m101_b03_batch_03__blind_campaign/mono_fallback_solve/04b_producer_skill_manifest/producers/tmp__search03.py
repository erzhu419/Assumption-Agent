import pickle,sys,importlib.util
import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=len(tr['x']); t=s['eval_times']; T=tr['traj']; y=tr['y']; x=tr['x']; a=np.arange(52); b=np.arange(52,72)
def load(i):
 p=f'/tmp/hegel_continual_archive/conf_m101_b03_batch_0{i}_solver.py'; sp=importlib.util.spec_from_file_location('arc',p); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
m=load(1); c=m._calibrate(s['panel_a']); z=m._inverse(x,c); print('calibration',c.round(5)); X=np.c_[np.ones(n),z]; P=s['panel_b']; B=X@np.linalg.lstsq(np.c_[np.ones(len(P['u'])),P['u']],P['v'],rcond=None)[0];
from itertools import combinations
from scipy.optimize import minimize_scalar
from scipy.linalg import cho_factor,cho_solve
P=s['panel_c']; u=P['u']; fit=lambda A,V:np.linalg.lstsq(A,V,rcond=1e-10)[0]
def aff(q):return np.c_[np.ones(len(q)),q]
def cyc(q):return q*np.roll(q,-1,axis=1)
def op(j):
 v=P['v'+str(j)]; scores=[]
 for name,fn in [('tanh',lambda x,k:np.tanh(k*x)),('rat',lambda x,k:x/(1+k*x*x)),('sin',lambda x,k:np.sin(k*x)),('log',lambda x,k:np.log1p(k*x*x)),('cos',lambda x,k:np.cos(k*x)),('sqrt',lambda x,k:np.sqrt(1+k*x*x)),('exp',lambda x,k:np.exp(-k*x*x))]:
  for ext in (False,True):
   f=lambda q,k:np.c_[aff(fn(q,k)),q] if ext else aff(fn(q,k))
   def loss(k):
    A=f(u,k); e=[]
    for i in range(4):
     g=np.arange(len(u))%4!=i;e.extend(A[~g]@fit(A[g],v[g])-v[~g])
    return np.sqrt(np.mean(np.square(e)))
   grid=np.linspace(.1,4,24); ii=np.argmin([loss(k) for k in grid]); k=minimize_scalar(loss,bounds=(grid[max(0,ii-1)],grid[min(len(grid)-1,ii+1)]),method='bounded').x
   scores.append((loss(k),name,ext,k,f(z,k)@fit(f(u,k),v),fit(f(u,k),v)))
 scores.sort(key=lambda q:q[0]); print('panel',j,[q[:4] for q in scores[:8]],'coef',scores[0][5].round(4));return scores[0][4]
p,q=op(1),op(2)
h=lambda r:-np.expm1(-t[:,None]*r)
r=least_squares(lambda r:(h(r)@fit(h(r),T[a].T)-T[a].T).ravel(),[.35,1.9,5.2],bounds=(.001,25.)).x; H=h(r); G=fit(H,T.T).T; Y=np.c_[y,G,T[:,-1]]
views={'z':z,'b':B,'p':p,'q':q}; groups={**views}
for nm,U in views.items():
 groups[nm+'2']=PolynomialFeatures(2,include_bias=False).fit_transform(U)[:,4:]
 groups[nm+'nl']=np.column_stack([f(k*U) for f in (np.sin,np.cos,np.tanh,lambda w:np.log1p(w*w)) for k in (.7,1.5,2.5)])
for i,j in combinations(views,2):
 U,V=views[i],views[j]; groups[i+j]=np.einsum('ij,ik->ijk',U,V).reshape(n,-1)
base=np.c_[z,p,q]; scores=[]
for key in [()]+[(k,) for k in groups]+list(combinations([k for k in groups if k not in views],2)):
 A=np.c_[base,*[groups[k] for k in key]]; C=A[a].mean(0); D=np.maximum(A[a].std(0),1e-9); A=(A-C)/D; K=A@A.T
 for lam in (.00001,.001,.03,.3):
  pred=Y[a].mean(0)+K[np.ix_(b,a)]@cho_solve(cho_factor(K[np.ix_(a,a)]+lam*np.eye(len(a))),Y[a]-Y[a].mean(0)); E=pred-Y[b]; scores.append((np.r_[np.mean(abs(E),0),np.mean(abs(E[:,1:4]@H.T))],key,lam))
for j in range(7 if False else 6):
 print('top',j,[(np.round(e,4),k,l) for e,k,l in sorted(scores,key=lambda r:r[0][j])[:5]])
np.savez('/tmp/spatial03.npz',z=z,B=B,p=p,q=q,H=H,G=G,y=y,T=T)
