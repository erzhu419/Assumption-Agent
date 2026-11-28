exec(open('/tmp/explore11.py').read().split("ops={'z':z}")[0])
from itertools import combinations
from sklearn.metrics.pairwise import rbf_kernel
u=s['panel_c']['u']; v=s['panel_c']['v2']; v1=s['panel_c']['v1']; i,j=np.triu_indices(4,1)
rates=s['telemetry_vec']; print('telemetry',rates)
def c2(x,p):return aff(x+p[0]*(1-np.cos(p[1]*x)))
def c1(x,p):return np.c_[aff(x),x*np.exp(-p[0]*x*x),x/(1+p[1]*abs(x)**p[2])]
fits=[]
for key,fn,V,start,b in [('c2',c2,v,(.22,2.15),([-.8,.3],[.8,7])),('c1',c1,v1,(6,1,1.1),([.1,.01,.3],[20,10,4]))]:
 def fun(p):
  f=fn(u,p);return (f@ls(f,V)-V).ravel()
 p=least_squares(fun,start,bounds=b,gtol=1e-13).x; W=ls(fn(u,p),V); print(key,p,np.std(fun(p)),W.round(5));fits.append((fn,p,W))
def calc(x):
 A,B=[fn(x,p)@w for fn,p,w in fits];return A,B
A,B=calc(z); T=tr['traj']; splits=[np.arange(72)]+[np.random.default_rng(i).permutation(72) for i in range(1,9)]
def report(f,pen=0,roc=False):
 fs=f.copy(); mu=fs.mean(axis=0); sd=fs.std(axis=0); k=sd>1e-10; fs=(fs[:,k]-mu[k])/sd[k]; fs=aff(fs); es=[]
 for ids in splits:
  aa,bb=ids[:52],ids[52:]; X=fs[aa]; w=ls(X,y[aa]) if not pen else np.linalg.solve(X.T@X+np.diag(np.r_[0,np.full(X.shape[1]-1,pen)]),X.T@y[aa]); es.append(np.mean(abs(fs[bb]@w-y[bb])))
 return np.round([es[0],np.mean(es[1:]),max(es)],5)
def prods(a,b):return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
def diag(a,b):return np.c_[a*b,a*np.roll(b,1,axis=1),a*np.roll(b,2,axis=1),a*np.roll(b,3,axis=1)]
Pr=z/(1+1.02*abs(z)**1.1225); G=z*np.exp(-6.22*z*z); C=1-np.cos(2.15*z);
D={'z':z,'A':A,'B':B,'zA':np.c_[z,A],'zB':np.c_[z,B],'AB':np.c_[A,B],'zAB':np.c_[z,A,B],'PR':np.c_[z,Pr,G,C]}
for nm,h in D.items():
 print(nm,'aff',report(h),'quad',*[report(quad(h),pen) for pen in [0,.01,.1,1]],'trans',*[report(np.c_[h,np.tanh(h),np.tanh(2.15*h),h/(1+abs(h)),h*np.exp(-h*h),np.cos(2.15*h)],pen) for pen in [.01,.1,1]])
for nm,h in [('zz',z),('aa',A),('bb',B),('ab',np.c_[A,B]),('zab',np.c_[z,A,B])]:
 F=np.c_[quad(h),Pr,G,C]; print('mix',nm,*[report(F,p) for p in [0,.01,.1,1]])
for nm,h in [('ABprod',prods(A,B)),('ABsame',A*B),('ABx',np.c_[prods(A,B),A,B]),('zAprod',prods(z,A)),('zBprod',prods(z,B)),('GPC',np.c_[Pr,G,C]),('latentGate',np.c_[A/(1+.22*B*B),B/(1+.22*A*A),A*np.tanh(2.15*B),B*np.tanh(2.15*A)]),('latentAllGate',np.c_[prods(A,1/(1+.22*B*B)),prods(B,1/(1+.22*A*A)),prods(A,np.tanh(2.15*B)),prods(B,np.tanh(2.15*A))])]:
 for k,base in [('base',z),('full',np.c_[z,A,B]),('quad',quad(z))]:
  print(nm,k,*[report(np.c_[base,h],p) for p in [0,.03,.3]])
