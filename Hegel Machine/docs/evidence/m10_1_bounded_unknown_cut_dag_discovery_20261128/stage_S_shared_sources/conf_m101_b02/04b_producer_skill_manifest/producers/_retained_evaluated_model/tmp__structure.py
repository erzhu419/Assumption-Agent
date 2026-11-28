from compact import *
from scipy.spatial.distance import cdist
from itertools import product
B,C,D,_=operators(s); f=B(x); c=C(f); d=D(f)

def mult(a,b): return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
def K(a): return a@a.T
def ridge(k,tr,va,l):
 h=k[np.ix_(tr,tr)]; kr=k[np.ix_(va,tr)]; mu=Y[tr].mean(0)
 return (kr-kr.mean(1,keepdims=True)-h.mean(0)+h.mean())@np.linalg.solve(h-h.mean(0)-h.mean(1,keepdims=True)+h.mean()+l*np.eye(len(tr)),Y[tr]-mu)+mu

def err(p,va):
 e=abs(p-Y[va]);return np.r_[e[:,0].mean(),e[:,1:].mean(),e[:,-1].mean()]

k0=K(np.c_[f,c,d,mult(f,c)]); kfd=K(mult(f,d)); kcd=K(mult(c,d)); kr=[K(mult(f,f)),K(mult(c,c)),K(mult(d,d))]; R=cdist(f,f,'sqeuclidean'); Rc=cdist(c,c,'sqeuclidean'); Rd=cdist(d,d,'sqeuclidean')
kernels={'base':k0,'fd':k0+.09*kfd,'cd':k0+.09*kcd,'both':k0+.09*(kfd+kcd),'poly':k0+.09*(kfd+kcd+sum(kr)),'krbf':k0+.03*np.exp(-R/.8),'kcrbf':k0+.03*np.exp(-Rc/.8),'additive':k0+.03*sum(np.exp(-cdist(v[:,None],v[:,None],'sqeuclidean')/.20) for v in f.T),'mixrbf':k0+.03*(np.exp(-R/.8)+np.exp(-Rc/.8)+np.exp(-Rd/.8))}
config=list(product(kernels,[.00005,.0003,.001,.003,.01])); pred=np.array([[ridge(kernels[k],tr,va,l) for k,l in config] for tr,va in []]) if False else [[ridge(kernels[k],tr,va,l) for k,l in config] for tr,va in folds]; E=np.array([[err(p,va) for p in pe] for pe,(_,va) in zip(pred,folds)])
for j in range(3):
 for i in np.argsort(E[1:6,:,j].mean(0)+.5*E[-1,:,j])[:5]:print('fixed',j,config[i],E[0,i],E[1:6,i].mean(0),E[-1,i]);
baseidx=config.index(('base',.0003)); ensemble=[config.index(q) for q in [('base',.0003),('fd',.003),('cd',.0003),('both',.001)]]
for o,(tr,va) in enumerate(folds):
 rng=np.random.RandomState(753+o); inner=np.array_split(rng.permutation(tr),4); inner.append(tr[np.argsort(np.linalg.norm(x[tr],axis=1))[-max(10,len(tr)//5):]])
 e=np.array([[err(ridge(kernels[k],np.setdiff1d(tr,j),j,l),j) for k,l in config] for j in inner]); scores=e[:4].mean(0)+.35*e[4]; chosen=np.argsort(scores,axis=0)[:5]; P=np.mean(np.asarray(pred[o])[chosen[:,0]],axis=0); P[:,1:]=np.mean(np.asarray(pred[o])[chosen[:,1]],axis=0)[:,1:]; print('outer',o,'base',E[o,baseidx],'fixed blend',err(np.mean(np.asarray(pred[o])[ensemble],0),va),'nested blend',err(P,va))

# Fit global temporal shapes on the prescribed fitting rows, compare the held-out projection floor.
t=s['eval_times']; tr,va=folds[0]; funcs={'exp':lambda t,a:-np.expm1(-a*t),'rat':lambda t,a:t/(1+a*t),'sin':lambda t,a:np.sin(a*t),'log':lambda t,a:np.log1p(a*t),'damped':lambda t,a:t*np.exp(-a*t),'atan':lambda t,a:np.arctan(a*t)}; r=[]
for (n1,f1),(n2,f2) in product(funcs.items(),repeat=2):
 def g(p):return np.c_[f1(t,p[0]),f2(t,p[1])]
 def res(p):
  h=g(p);return (h@fit(h,z[tr].T)-z[tr].T).ravel()
 for p in [[.4,.8],[.8,1.8],[.4,2.8]]:
  q=least_squares(res,p,bounds=(.001,6),max_nfev=120); h=g(q.x); ev=h@fit(h,z[va].T)-z[va].T; r.append((np.std(q.fun),np.std(ev),n1,n2,q.x))
for v0 in sorted(r,key=lambda a:a[0])[:8]: print('temporal',v0)
