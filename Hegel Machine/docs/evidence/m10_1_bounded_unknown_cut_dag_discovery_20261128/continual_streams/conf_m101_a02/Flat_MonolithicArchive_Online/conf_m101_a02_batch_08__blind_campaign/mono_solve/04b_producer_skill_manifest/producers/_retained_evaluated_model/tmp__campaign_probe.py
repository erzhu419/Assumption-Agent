import pickle, numpy as np, sys
from scipy.optimize import least_squares, minimize_scalar
from scipy.linalg import solve
sys.path.insert(0,'/tmp/hegel_continual_archive'); import conf_m101_a02_batch_06_solver as old
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=len(tr['y']); times=s['eval_times']; fit=old._fit; aff=old._aff

def cal(panel,x):
 u,v=panel['u'],panel['v']; a=aff(u); r=np.linalg.norm(v-a@fit(a,v),axis=1); k=np.argsort(r)[:-3]; rough=fit(a[k],v[k]); j=np.arange(4); h=(j+1)%4
 def des(z):
  q=z[:,h]; return np.stack((np.ones_like(z),z,q,q*q,z**3),axis=-1)
 d=des(u); c=np.array([fit(d[k,i],v[k,i]) for i in j]); z=np.linalg.solve(rough[1:].T,(x-rough[0]).T).T
 for _ in range(10):
  jac=np.zeros((len(z),4,4)); jac[:,j,j]=c[:,1]+3*c[:,4]*z*z; jac[:,j,h]=c[:,2]+2*c[:,3]*z[:,h]; z-=np.linalg.solve(jac,((des(z)*c).sum(-1)-x)[...,None])[...,0]
 return z

def panel(p,key,z,basis,init,bounds):
 u,v=p['u'],p[key]
 def res(p):
  f=basis(u,p); return (f@fit(f,v)-v).ravel()
 q=least_squares(res,init,bounds=bounds,ftol=1e-11,xtol=1e-11,gtol=1e-11).x; f=basis(u,q); print(key,q,np.std(res(q)),flush=True); return basis(z,q)@fit(f,v)

def coords():
 old._calibrate=cal; old._panel=panel; return old._coords(s)

def cv(f):
 for l in (52,56):
  x=dict(s,train={k:(v[:l] if k!='summary' else v) for k,v in tr.items()},test={'x':tr['x'][l:]}); o=f(x); print('split',l,np.mean(abs(o['y_hat']-tr['y'][l:])),np.mean(abs(o['state_pred']-tr['traj'][l:])))

if __name__=='__main__':
 z,b,c,d=coords(); cv(old.solve_batch); z,b,c,d=(q[:n] for q in (z,b,c,d)); y=tr['y']; state=tr['traj']; np.set_printoptions(precision=3,suppress=True)
 def prod(a,b):return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
 def gram(x):
  x=(x-x.mean(0))/(x.std(0)+1e-9); return x@x.T/x.shape[1]
 Z,B,C,D=map(gram,(z,b,c,d)); groups=dict(z=Z,b=B,c=C,d=D,zz=Z**2,bb=B**2,cc=C**2,dd=D**2,zb=Z*B,zc=Z*C,zd=Z*D,bc=B*C,bd=B*D,cd=C*D,zzz=Z**3,bbb=B**3)
 amp=old._shared_amplitude(times,state,np.c_[aff(z),b,c,d]); print('amp/range',np.std(amp),np.corrcoef(y,amp)[0,1]);
 for names in ['z','b','c','d','z b','z b c d','z b c d zz bb','z b c d zz bb zb','z b c d zz bb zb zc zd','z b c d zz bb zb bc bd','z b c d zz bb zb cc dd','z b bb zb','z b c d zz bb zb zzz bbb']:
  k=1+sum(groups[q] for q in names.split()); a=[]
  for l in [52,56]:
   rr=[]
   for r in [.00001,.001,.03,.15]:
    o=old._predict(k,np.c_[y,amp,state][:l],r); rr.append(np.mean(abs(o-np.c_[y,amp,state][l:]),axis=0)[[0,1,9]])
   a.append(rr)
  print(names,np.array(a))
 for lab,t in [('y',y),('amp',amp),('y-amp',y-amp),('y+amp',y+amp)]:
  for label,f in [('z',aff(z)),('b',aff(b)),('c',aff(c)),('d',aff(d)),('all',np.c_[aff(z),b,c,d])]:
   cc=fit(f[:52],t[:52]); print(lab,label,np.mean(abs(f[52:]@cc-t[52:])),cc if f.shape[1]<6 else '')
