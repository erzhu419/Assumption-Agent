from fitops import *
p=s['panel_c']; u=p['u']; v=p['v1']; mask=np.arange(len(u))%4!=0
from scipy.special import expit

def fam(x,w,k):
 if k=='power': h=x/(1+w[0]*x*x)**w[1]
 if k=='softsign': h=x/(1+w[0]*abs(x))
 if k=='asinh': h=np.arcsinh(w[0]*x)
 if k=='sinc': h=x*np.sinc(w[0]*x/np.pi)**2
 if k=='mixed': h=np.c_[np.sin(w[0]*x),np.sin(w[1]*x)]
 return np.c_[aff(x),h]
for kind,w in [('power',[2,.5]),('softsign',[2]),('asinh',[3]),('sinc',[2]),('mixed',[3,6])]:
 for keep in (np.ones(len(u),bool),mask):
  def resid(w):
   f=fam(u[keep],w,kind); return (f@ls(f,v[keep])-v[keep]).ravel()
  z=least_squares(resid,w,bounds=(.05,30),gtol=1e-12).x
  f=fam(u,z,kind); e=f@ls(f[keep],v[keep])-v
  print(kind,z,np.sqrt(np.mean(e[keep]**2)),np.sqrt(np.mean(e[~mask]**2)))
X,x,b,bx,c,d=features(s); c=op(u,v,x,'atan',2.4)[0]; n=len(s['train']['y']); T=np.c_[s['train']['y'],s['train']['traj']]; tr=np.arange(52); te=np.arange(52,n)
F=np.c_[np.ones(len(x)),c,d,(c[:,:,None]*d[:,None,:]).reshape(len(x),16)]
names=['1']+[f'{a}{i}' for a in 'cd' for i in range(4)]+[f'c{i}d{j}' for i in range(4) for j in range(4)]
w=ls(F[tr],T[tr]); print('weights [y, trajectory first, middle, last]'); print('\n'.join(f'{a:5} {row}' for a,row in zip(names,np.round(w[:,[0,1,4,8]],4))))
for which in (0,1):
 V=T[:,0:1] if which==0 else T[:,1:]; active=list(range(F.shape[1])); errs=[]
 while len(active)>2:
  a=F[:,active]; e=abs(a[te]@ls(a[tr],V[tr])-V[te]); errs.append((len(active),round(e.mean(),5),[names[j] for j in active]));
  active.remove(min(active,key=lambda j: np.linalg.norm(F[tr][:,[i for i in active if i!=j]]@ls(F[tr][:,[i for i in active if i!=j]],V[tr])-V[tr])))
 print('pruning',which,[(a,b) for a,b,_ in errs]); print('supports',*[r for r in errs if r[0] in (9,12,15)])
