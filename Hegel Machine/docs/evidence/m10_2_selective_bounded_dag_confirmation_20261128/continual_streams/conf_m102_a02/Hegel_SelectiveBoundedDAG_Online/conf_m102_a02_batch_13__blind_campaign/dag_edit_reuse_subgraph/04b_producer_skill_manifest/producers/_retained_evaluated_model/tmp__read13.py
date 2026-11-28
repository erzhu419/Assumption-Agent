import pickle,numpy as np
from scipy.optimize import root, minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
def af(q,j):
 a=q[:,j]; b=q[:,(j+1)%4];return np.c_[np.ones(len(q)), a,b,a*b,a**3]
u,v=[s['panel_a'][k] for k in 'uv']; good=np.linalg.norm(v-u,axis=1)<5; co=[ls(af(u[good],j),v[good,j]) for j in range(4)]; print('A',co)
def fw(q):return np.column_stack([af(q,j)@co[j] for j in range(4)])
def inv(x):return np.array([root(lambda q:fw(q[None,:])[0]-r,r).x for r in x])
def bf(q,p):return np.c_[np.ones(len(q)),np.cos(q*p)]
u,v=[s['panel_b'][k] for k in 'uv']; p=minimize_scalar(lambda p:np.sum((bf(u,p)@ls(bf(u,p),v)-v)**2),bounds=(1,4),method='bounded').x; w=ls(bf(u,p),v)
q=s['panel_c']['u']; b=bf(q,p)@w
for z in [s['panel_c'][k] for k in ('v1','v2')]:
 for name,a in [('q',q),('b',b),('both',np.c_[q,b]),('inter', np.c_[q,b,(q[:,:,None]*b[:,None,:]).reshape(-1,16)])]:
  a=np.c_[np.ones(len(a)),a]; print('C',name,np.sqrt(np.mean((a@ls(a,z)-z)**2)))
 print('C stats',np.round(z.std(0),4),np.round(np.corrcoef(np.c_[q,b,z].T)[-4:,:8],2))
tr=s['train']; q=inv(tr['x']); b=bf(q,p)@w; c=b*np.roll(q,-1,axis=1); print('q range',q.min(0),q.max(0), 'inverse residual',abs(fw(q)-tr['x']).max()); print('traj singular values',np.linalg.svd(tr['traj'],compute_uv=False));
for name,a in [('b',b),('q',q),('bc',np.c_[b,c]),('bcs',np.c_[b,c.sum(1)]),('qbc',np.c_[q,b,c])]:
 a=np.c_[np.ones(len(a)),a]; print(name,[(np.mean(abs(a@ls(a,z)-z)),np.max(abs(a@ls(a,z)-z))) for z in [tr['y'],tr['traj']]])
for g in [.22,1.25,2.15]:
 for i in range(4):
  h=q[:,i]; gate=1/(1+g*h*h); a=np.c_[np.ones(len(q)),b*gate[:,None],c,b*h[:,None]**3*gate[:,None]]; print('gate',g,i,np.mean(abs(a@ls(a,tr['y'])-tr['y'])))
from itertools import combinations_with_replacement as cr
from sklearn.linear_model import OrthogonalMatchingPursuit
qc=s['panel_c']['u']; bc=bf(qc,p)@w; gc=np.cos(p*qc); g=np.cos(p*q)
print('panel lengths', {k:{a:np.shape(b) for a,b in v.items()} for k,v in s.items() if isinstance(v,dict)})
def cv(f,y):
 return [round(np.mean(abs(f[j]@ls(f[i],y[i])-y[j])),6) for j in [np.arange(52,72),np.arange(0,20),np.arange(20,40),np.arange(40,60)] for i in [np.setdiff1d(np.arange(72),j)]]
def pair(a): return np.column_stack([a[:,i]*a[:,j] for i,j in cr(range(a.shape[1]),2)])
def cross(a,b): return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
for name,a in [('b',b),('b+b2',np.c_[b,pair(b)]),('b+q2',np.c_[b,pair(q)]),('b+q*b',np.c_[b,cross(q,b)]),('b+q+q*b',np.c_[b,q,cross(q,b)]),('b+diag q2',np.c_[b,q*q]),('b+diag q4',np.c_[b,q*q,q**4]),('b+q2*b',np.c_[b,cross(q*q,b)]),('b+sin',np.c_[b,np.sin(p*q)]),('b+cos2',np.c_[b,np.cos(2*p*q)]),('b+tanh',np.c_[b,np.tanh(b)]),('b+diag b2',np.c_[b,b*b])]:
 f=np.c_[np.ones(len(q)),a]; print('scalar',name,cv(f,tr['y']),'train',np.mean(abs(f@ls(f,tr['y'])-tr['y'])))
for key in ('v1','v2'):
 z=s['panel_c'][key]; print('C',key)
 for nam,d in [('b',bc),('cos',gc),('tanh',np.tanh(qc)),('sin',np.sin(p*qc)),('square',qc**2),('cube',qc**3)]:
  f=np.c_[np.ones(len(qc)),qc,d]; e=f@ls(f,z)-z; print(nam,np.sqrt(np.mean(e**2)), 'LOO', np.sqrt(np.mean((e / (1-np.einsum('ij,ji->i',f,np.linalg.pinv(f)))[:,None])**2)))
 print('linear coefficients',np.round(ls(np.c_[np.ones(len(qc)),qc,gc],z),4))
