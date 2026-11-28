exec(open('/tmp/probe.py').read().split('np.set_printoptions')[0])
from scipy.optimize import least_squares
v=p['v2']; T=s['eval_times']; print('n,t',n,T)
def fit(a,z):return np.linalg.lstsq(a,z,rcond=1e-10)[0]
def basis(q,w,kind):
 a=[np.ones((len(q),1)),np.cos(q*w[:4])]
 if kind==1:a += [q]
 if kind==2:a += [q*q]
 if kind==3:a += [np.sin(q*w[:4])]
 if kind==4:a += [np.cos(q*w[4:])]
 if kind==5:a += [np.sin(q*q*w[4:])]
 if kind==6:a += [np.cos(q*w[:4])**2]
 return np.column_stack(a)
for kind in range(7):
 for k in [19,24]:
  def fun(w):
   A=basis(u[:k],w,kind);return (A@fit(A,v[:k])-v[:k]).ravel()
  R=least_squares(fun,[2.3]*4+([1.2]*4 if kind in (4,5) else []),bounds=(.2,6),max_nfev=300); A=basis(u,R.x,kind); E=A@fit(A[:k],v[:k])-v; print(kind,k,np.round(R.x,4),round(np.mean(abs(E[:k])),6),round(np.mean(abs(E[k:])),6) if k<24 else '-')
  if k==24 and kind==0: print('mix',np.round(fit(A,v),4))
  if k==24 and kind in (0,2,4,5,6):
   c=basis(x,R.x,kind)@fit(A,v); q=np.c_[x,b,c]; Z=np.c_[y,tr]; k0=52
   for name,a in [('linear',q),('plus q2',np.c_[q,q*q]),('latent v2+b quad',np.c_[q,*[q[:,i]*q[:,j] for i in range(4,12) for j in range(i,12)]]),('v2 quad',np.c_[q,*[c[:,i]*c[:,j] for i in range(4) for j in range(i,4)]]),('b v2 cross',np.c_[q,*[c[:,i]*b[:,j] for i in range(4) for j in range(4)]]),('x v2 cross',np.c_[q,*[c[:,i]*x[:,j] for i in range(4) for j in range(4)]])]:
    a=np.c_[np.ones(len(a)),a][:n];e=np.abs(a@fit(a[:k0],Z[:k0])-Z);print('read',kind,name,np.round([e[:k0,0].mean(),e[k0:,0].mean(),e[:k0,1:].mean(),e[k0:,1:].mean()],4))
