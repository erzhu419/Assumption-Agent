import numpy as np,pickle,itertools,sys
from scipy.optimize import brentq,least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-9)[0]
D=lambda *a:np.c_[np.ones(len(a[0])),*a]
def latent(p,x):
 u,v=p['u'],p['v']; f=lambda z:.18*np.tanh(1.8*z)-.32*np.sin(2.4*np.roll(z,2,axis=-1)); good=np.max(abs(v-u),axis=1)<3; c=np.asarray([ls(D(u[good,j]),(v-f(u))[good,j]) for j in range(4)]); sol=(x-c[:,0])/c[:,1]
 for _ in range(240):sol=(x-c[:,0]-f(sol))/c[:,1]
 return sol
z=latent(s['panel_a'],tr['x']); print('inversion range',z.min(0),z.max(0)); p=s['panel_b']; m=s['panel_c']
B=lambda a:D(np.cos(2.6*a)); C=lambda a:D(a*np.roll(a,-1,axis=1)); E=lambda a,q:D(a,1/(1+(q*a)**2)); u=m['u'];v=m['v2']; p2=least_squares(lambda q:(E(u,q)@ls(E(u,q),v)-v).ravel(),[.4]).x
print('C2',p2,'coeff',ls(E(u,p2),v)); funcs=[lambda a:B(a)@ls(B(p['u']),p['v']),lambda a:C(a)@ls(C(u),m['v1']),lambda a:E(a,p2)@ls(E(u,p2),v)]
A={'z':z}
for tree in ['',*[''.join(k) for t in [1,2] for k in itertools.product('bcd',repeat=t)]]:
 for key,fn in zip('bcd',funcs): A[tree+key]=fn(A[tree or 'z'])
y=tr['y']; T=tr['traj']; S,V=np.linalg.svd(T[:52],full_matrices=False)[1:]; Y=np.c_[y,T@V[:2].T]; print('svd',S,'V',V[:2]);
def ev(a):
 q=ls(a[:52],Y[:52]); r=a@q-Y; return np.mean(abs(r[:52]),axis=0),np.mean(abs(r[52:]),axis=0)
out=[]
for names in itertools.combinations(list(A)[:13],2):
 a,b=[A[k] for k in names]; c=a[:,:,None]*b[:,None,:];
 for kind,extra in [('plain',[]),('trace',[np.trace(c,axis1=1,axis2=2)]),('trace+',[np.trace(c,axis1=1,axis2=2),c[:,1,0]]),('all',[c.reshape(72,16)])]:
  m=D(a,b,*extra); fit,te=ev(m);out.append((fit,te,names,kind))
for j in range(3):
 print('TARGET',j)
 for row in sorted(out,key=lambda x:x[1][j])[:12]:print(row)
for k in ('b','c','d','bc','bd','cb','cd','bcd','bdc'):
 a,b=A['z'],A[k]; c=a[:,:,None]*b[:,None,:]; print('z+',k,ev(D(a,b,np.trace(c,axis1=1,axis2=2),c[:,1,0])))
print('y range',y.min(),y.max(),'correlations',np.corrcoef(Y.T))
np.savez('/tmp/feat15.npz',**A,Y=Y,V=V)
