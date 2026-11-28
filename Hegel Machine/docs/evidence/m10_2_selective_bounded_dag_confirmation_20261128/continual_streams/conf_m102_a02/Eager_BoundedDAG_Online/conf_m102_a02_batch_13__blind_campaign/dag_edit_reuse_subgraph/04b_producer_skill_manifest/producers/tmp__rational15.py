import numpy as np,pickle
from scipy.optimize import least_squares,minimize_scalar
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; Y=np.column_stack([s['train']['y'],s['train']['traj']]);
def fit(a,b):return np.linalg.lstsq(a,b,rcond=1e-9)[0]
def aff(a):return np.c_[np.ones(len(a)),a]
def cal(p,x):
 u,v=p['u'],p['v']; w=np.roll(u,-1,axis=1); C=[]
 for i in range(u.shape[1]):
  a,b=u[:,i],w[:,i]; D=np.c_[np.ones(len(a)),a,b,a*b,a**3]; c=least_squares(lambda c:D@c-v[:,i],fit(D,v[:,i]),loss='cauchy',f_scale=.002).x; m=abs(D@c-v[:,i])<.002; C.append(fit(D[m],v[m,i]))
 c=np.array(C).T
 def f(z):
  w=np.roll(z,-1);return c[0]+c[1]*z+c[2]*w+c[3]*z*w+c[4]*z**3
 def j(z):
  a=np.diag(c[1]+c[3]*np.roll(z,-1)+3*c[4]*z*z);i=np.arange(len(z));a[i,(i+1)%len(z)]=c[2]+c[3]*z;return a
 return np.array([least_squares(lambda z:f(z)-v,v,jac=j,gtol=1e-11).x for v in x])
def panel(p,key,x,f,bounds,linear=True):
 U,V=p['u'],p[key];D=lambda a,w:np.c_[aff(a) if linear else np.ones(len(a)),f(w*a)];
 def loss(w):
  a=D(U,w); return np.mean((a@fit(a,V)-V)**2)
 o=minimize_scalar(loss,bounds=bounds,method='bounded'); print(key,o.x,o.fun);return D(x,o.x)@fit(D(U,o.x),V)
z=cal(s['panel_a'],x); g=panel(s['panel_b'],'v',z,np.cos,(2,3),False)
def cs(a):
 return [panel(s['panel_c'],'v1',a,np.tanh,(1.5,3)),panel(s['panel_c'],'v2',a,lambda x:1/(1+x*x),(.2,.8))]
hz,kz=cs(z); hg,kg=cs(g)
folds=[np.arange(52,72),*np.array_split(np.random.default_rng(9).permutation(len(z)),4)]
def scores(D):
 E=[]
 for v in folds:
  t=np.setdiff1d(np.arange(len(D)),v); r=abs(D[v]@fit(D[t],Y[t])-Y[v]);E.append([r[:,0].mean(),r[:,1:].mean()]);
 return np.round(np.array(E)[[0, np.argmax(np.max(E,axis=1))]],4).tolist()
def q(a):return np.column_stack([a[:,i]*a[:,j] for i,j in combinations_with_replacement(range(a.shape[1]),2)])
from itertools import combinations, product
from scipy.linalg import qr
from scipy.spatial.distance import cdist
u=np.cos(2.600025*z); y=Y[:,0]; ids=np.arange(len(y)); B=aff(g); Q=q(g); P=np.c_[B,Q]; pairs=list(combinations_with_replacement(range(4),2)); S=Q[:,[pairs.index((0,2)),pairs.index((1,3))]]
def ev(fn):
 a=[]
 for v in folds:
  t=np.setdiff1d(ids,v); p=fn(t,v); a.append(np.mean(abs(y[v]-p)))
 return np.round(a,5)
def regress(D,t,v):return D[v]@fit(D[t],y[t])
cs={'g':g,'u':u,'z':z,'h':hg,'k':kg,'hz':hz,'kz':kz}; E={}; den={}
for k,a in cs.items():
 for i in range(4):
  den[k,str(i)]=a[:,i]
 for i,j in combinations(range(4),2):
  for c in [1,-1,.5,-.5]:den[k,i,j,c]=a[:,i]+c*a[:,j]
 for i,j in combinations_with_replacement(range(4),2):den[k,i,j,'prod']=a[:,i]*a[:,j]
# Whole-readout rational models, with denominator profiling on training only.
for D in [B,np.c_[B,S],P]:
 for ks in [list(den),[k for k in den if k[0] in ['u','g']]]:
  H=np.column_stack([den[k]**2 for k in ks]); C=[]
  for v in folds:
   t=np.setdiff1d(ids,v); opt=[]
   for j,k in enumerate(ks):
    d=H[:,j]
    def loss(c,ret=False):
     A=D/(1+c*d[:,None]); U=qr(A[t],mode='economic')[0]; r=y[t]-U@(U.T@y[t]); l=np.mean((r/(1-(U**2).sum(1)))**2);return (l,regress(A,t,v)) if ret else l
    m=minimize_scalar(loss,bounds=(0,8),method='bounded',options={'xatol':.003});opt.append((m.fun,k,m.x,j))
   oo=sorted(opt)[:3]; C.append([(k,round(c,3),round(l**.5,5),round(abs(loss2-y[v]).mean(),5)) for l,k,c,j in oo for d in [H[:,j]] for loss2 in [loss(c,True)[1]]]);
  print('whole',D.shape[1],len(ks),C,flush=True)
# Explicit additional features, screened with leave-one-out inside each training split.
for k,d in den.items():
 for c in [.22,.5,1.25,2.15,4]:
  h=1/(1+c*d*d)
  E[k,c,'R']=h
  if len(k)<=2 or k[0] in ('g','u'):
   for nk,a in [('u',u),('g',g),('z',z)]:
    for i in range(4):E[k,c,nk,i]=h*a[:,i]
  if k[0] in ('g','u'):
   for i in range(2):E[k,c,'pair',i]=h*S[:,i]
 for c in [1.25,2.15,2.6]:
  E[k,c,'T']=np.tanh(c*d); E[k,c,'C']=np.cos(c*d);E[k,c,'S']=np.sin(c*d)
for name,a in cs.items():
 for i,j,k in combinations_with_replacement(range(4),3):E[name,i,j,k,'cube']=a[:,i]*a[:,j]*a[:,k]
keys=list(E); A=np.column_stack(list(E.values()));
for bn,D in [('aff',B),('cross',np.c_[B,S]),('quad',P)]:
 for v in folds:
  t=np.setdiff1d(ids,v); cur=D; sel=[]
  for step in range(5):
   U=qr(cur[t],mode='economic')[0]; r=y[t]-U@(U.T@y[t]); R=A[t]-U@(U.T@A[t]); n=np.linalg.norm(R,axis=0); R/=np.maximum(n,1e-8); h=(U**2).sum(1)[:,None]; cv=(r[:,None]-R*(r@R))/(np.maximum(1-h-R*R,.025)); s=np.mean(cv*cv,axis=0);s[n<1e-6]=np.inf; j=np.argmin(s);cur=np.c_[cur,A[:,j]];sel.append((keys[j],round(s[j]**.5,4),round(abs(regress(cur,t,v)-y[v]).mean(),4)))
  print('sparse',bn,sel,flush=True)
# Broad extrapolation comparison: residual kernel plus polynomial trend.
for dn,a in [('g',g),('z',z),('u',u),('z_u',np.c_[z,u]),('z_g',np.c_[z,g])]:
 a=a/a.std(0); d=cdist(a,a,'sqeuclidean');
 for bn,D in [('lin',B),('cross',np.c_[B,S]),('poly',P)]:
  vout=[]
  for v in folds:
   t=np.setdiff1d(ids,v); X=D[t]; Q1=qr(X,mode='economic')[0]; J=np.eye(len(t))-Q1@Q1.T; best=(np.inf,None,None)
   for w in [.005,.014,.04,.1,.25,.6,1.2]:
    for K in [np.exp(-w*d),(1+w*d)**-1,np.sqrt(1+w*d)*-1]:
     KT=K[np.ix_(t,t)]; lam,U=np.linalg.eigh(J@KT@J); lam=np.maximum(0,lam); U=J@U; Uy=U.T@y[t]
     for reg in [1e-7,1e-5,.0003,.01,.1]:
      M=U/(lam+reg); alpha=M@Uy; e=np.mean(np.abs(alpha/np.maximum((M*U).sum(1),1e-10)));
      if e<best[0]:best=(e,alpha,K)
   e,alpha,K=best; pred=D[v]@fit(X,y[t]-K[np.ix_(t,t)]@alpha)+K[np.ix_(v,t)]@alpha;vout.append((round(e,5),round(abs(y[v]-pred).mean(),5)))
  print('kernel',dn,bn,vout,flush=True)
