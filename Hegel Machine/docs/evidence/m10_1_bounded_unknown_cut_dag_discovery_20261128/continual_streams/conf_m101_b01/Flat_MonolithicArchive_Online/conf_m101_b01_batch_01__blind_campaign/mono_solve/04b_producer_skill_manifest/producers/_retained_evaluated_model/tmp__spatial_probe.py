from operators import *
from scipy.spatial.distance import cdist
from scipy.linalg import solve
from scipy.special import erf
from sklearn.model_selection import KFold
T=S['train']; X=cal(np.vstack([T['x'],S['test']['x']])); Y=np.c_[T['y'],T['traj']]; times=S['eval_times']
def model(pn,vn,kind):
 u=S[pn]['u']; v=S[pn][vn]
 if kind=='b':
  def d(x,k): return np.c_[aff(x),erf(k*x)]
  k=minimize_scalar(lambda k:metric(d(u,k),v)[1],bounds=(.5,2.5),method='bounded').x
 elif kind=='c':
  def d(x,k): return np.c_[np.ones(len(x)),np.arctan(k*x)]
  k=minimize_scalar(lambda k:metric(d(u,k),v)[1],bounds=(.5,3),method='bounded').x
 else:
  def d(x,k): return np.c_[aff(x),x*x]
  k=1
 w=fit(d(u,k),v); print(kind,'scale',k,'operator coefficient',np.round(w,3)); return lambda x:d(x,k)@w
B=model('panel_b','v','b'); C=model('panel_c','v1','c'); D=model('panel_c','v2','d')
coords={'raw':X,'B':B(X),'C':C(X),'D':D(X),'CB':C(B(X)),'DB':D(B(X)),'BC':B(C(X)),'BD':B(D(X)),'C+D':np.c_[C(X),D(X)],'CB+DB':np.c_[C(B(X)),D(B(X))],'B+X':np.c_[B(X),X],'C+X':np.c_[C(X),X],'D+X':np.c_[D(X),X]}
for a in (.4,1.,2.):
 for name in ['raw','B','C','D','CB','DB','BC','BD']:
  coords[f'{name}/tanh{a}']=np.tanh(a*coords[name]);coords[f'{name}/atan{a}']=np.arctan(a*coords[name])
sp=[(np.arange(52),np.arange(52,72)),*KFold(4,shuffle=True,random_state=91).split(Y)]
def score(pred):
 e=np.array([[np.mean(abs(v[:,0]-Y[j,0])),np.mean(abs(v[:,1:]-Y[j,1:]))] for v,(_,j) in zip(pred,sp)]); return np.r_[e[0],e[1:].mean(0),e[1:].max(0)]
rows=[]
for name,x in coords.items():
 for kind in ['aff','diag2','quad','cubic','rbf0.6','rbf1.2','rbf2.4','matern1.2']:
  if kind=='aff': f=aff(x)
  elif kind=='diag2': f=np.c_[aff(x),x*x]
  elif kind=='quad': f=poly(x,2)
  elif kind=='cubic': f=poly(x,3)
  else:
   dist=cdist(x,x)/np.sqrt(x.shape[1])/float(kind[-3:]);K=np.exp(-.5*dist**2) if kind.startswith('rbf') else (1+5**.5*dist+5/3*dist**2)*np.exp(-5**.5*dist)
  for lam in ([.0001,.01,.1] if kind!='aff' else [.0001]):
   pr=[]
   for i,j in sp:
    if kind.startswith(('rbf','matern')):
     avg=Y[i].mean(0);pr.append(avg+K[np.ix_(j,i)]@solve(K[np.ix_(i,i)]+lam*np.eye(len(i)),Y[i]-avg,assume_a='pos'))
    else:
     u=f[i]; pr.append(f[j]@solve(u.T@u+lam*np.eye(u.shape[1]),u.T@Y[i],assume_a='pos'))
   e=score(pr);rows.append((name,kind,lam,e))
for col in (2,3):
 print('BEST',col)
 for a,b,c,d in sorted(rows,key=lambda row:row[3][col]+.4*row[3][col+2])[:20]:print(a,b,c,np.round(d,5))
print('BASIC')
for a,b,c,d in rows:
 if '/' not in a and b in ('aff','diag2','quad') and c==.0001:print(a,b,np.round(d,5))
q=T['traj'][:52];
def timefit(p):
 a=-np.expm1(-np.exp(p[:,None])*times); return (q@np.linalg.pinv(a)@a-q).ravel()
r=least_squares(timefit,[-1,1]); print('time rates',np.exp(r.x),'rms',np.std(r.fun)); a=-np.expm1(-np.exp(r.x[:,None])*times); print('amplitude example',np.round((T['traj']@np.linalg.pinv(a))[:12],4))
