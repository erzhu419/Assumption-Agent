exec(open('/tmp/routes.py').read().split('for typ in range(8):')[0])
from scipy.optimize import least_squares
from sklearn.linear_model import Ridge, OrthogonalMatchingPursuit
from itertools import combinations_with_replacement
for mode in range(5):
 def rdes(z,k):
  q=z*z; a,b=k
  h=1+a*q+b*q*q
  return np.c_[np.ones(len(q)),q/h] if mode==0 else np.c_[np.ones(len(q)),q/h,q*q/h] if mode==1 else np.c_[np.ones(len(q)),q/(1+a*q),q/(1+b*q)] if mode==2 else np.c_[np.ones(len(q)),(q-1)/(1+a*q+b*q*q)] if mode==3 else np.c_[np.ones(len(q)),q*(1-b*q)/(1+a*q)]
 def res(k):
  f=rdes(u,k);return (f@ls(f,v)-v).ravel()
 k=least_squares(res,[.25,.1],bounds=([0,0],[10,10])).x; w=ls(rdes(u,k),v); print('pade',mode,k,np.linalg.norm(res(k)))
 if mode==1: rk,rw=k.copy(),w.copy(); rfun=rdes

def r1(z):
 q=z*z; h=1+rk[0]*q+rk[1]*q*q;return np.c_[np.ones(len(q)),q/h,q*q/h]@rw
raw=t['x']; traj=t['traj']; basis=old['_temporal_basis'](s['eval_times'],traj[:52]); Y=np.c_[t['y'],traj@basis]; print('time abs residual',np.mean(abs(traj-traj@basis@basis.T))); print('Y stdev & min/max',Y.std(0),Y.min(0),Y.max(0)); print('y from scores',np.mean(abs(affine(Y[52:,1:])@ls(affine(Y[:52,1:]),Y[:52,0])-Y[52:,0])))

def prod(a,b,sym=False):
 return np.column_stack([a[:,i]*b[:,j] for i in range(a.shape[1]) for j in range(i if sym else 0,b.shape[1])])
for i,z in enumerate((x,raw,forward(x),inverse(x),forward(raw),inverse(raw))):
 a,b=p1(z),r1(z); g={'z':z,'p':a,'r':b,'zp':prod(z,a),'zr':prod(z,b),'pp':prod(a,a,1),'pr':prod(a,b),'rr':prod(b,b,1),'dot':(a*b).sum(1)[:,None],'zz':prod(z,z,1)}
 scores=[]
 for feats in [('z','p','r'),('z','p','r','dot'),('z','p','r','pr'),('z','p','r','zp'),('z','p','r','zr'),('z','p','r','pp'),('z','p','r','rr'),('z','p','r','zz'),('z','p','r','dot','zz'),('z','p','r','dot','zp','zr'),tuple(g)]:
  f=np.column_stack([g[k] for k in feats]); n=52; mean=f[:n].mean(0); std=f[:n].std(0); f=(f-mean)/np.maximum(std,1e-10); sc=[]
  for alpha in (1e-7,.001,.01,.1,1,5):
   reg=Ridge(alpha=alpha).fit(f[:n],Y[:n]); ep=reg.predict(f[n:])-Y[n:]; et=reg.predict(f[:n])-Y[:n]; error=[np.mean(abs(ep[:,0])),np.mean(abs(ep[:,1:]@basis.T)),np.mean(abs(et[:,0])),np.mean(abs(et[:,1:]@basis.T))]; sc.append((sum(error[:2]),alpha,np.round(error,4)))
  scores.append((min(sc,key=lambda x:x[0]),feats))
 print('route',i,*sorted(scores,key=lambda x:x[0][0])[:6],sep='\n')
