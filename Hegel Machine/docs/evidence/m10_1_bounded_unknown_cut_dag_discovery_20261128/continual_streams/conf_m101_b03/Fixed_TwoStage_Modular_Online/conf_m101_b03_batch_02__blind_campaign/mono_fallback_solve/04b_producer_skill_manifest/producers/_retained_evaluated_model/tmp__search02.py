exec(open('/tmp/probe02.py').read().split('for name,Z')[0])
from scipy.optimize import minimize_scalar
from scipy.linalg import cho_factor,cho_solve
from itertools import combinations_with_replacement as cwr
C=s['panel_c']; u=C['u']; cyc=[(i,(i+1)%4) for i in range(4)]
def cross(x):return np.column_stack([x[:,i]*x[:,j] for i,j in cyc])
M=np.linalg.lstsq(np.c_[np.ones(len(u)),cross(u)],C['v1'],rcond=None)[0]
def f2(x,k):return np.c_[np.ones(len(x)),x,np.log1p(k*x*x)]
def loss(k):
 f=f2(u,k);return np.mean((f@np.linalg.lstsq(f,C['v2'],rcond=None)[0]-C['v2'])**2)
k=minimize_scalar(loss,bounds=(.05,1.5),method='bounded').x; D=np.linalg.lstsq(f2(u,k),C['v2'],rcond=None)[0]; print('C2',k,loss(k)**.5,'orthogonal ratios',np.linalg.svd(D[1:5],compute_uv=False),np.linalg.svd(D[5:],compute_uv=False))
zB=z@B[1:]+B[0]; zI=(z-B[0])@np.linalg.inv(B[1:]); X={}
for n,v in [('z',z),('b',zB),('i',zI)]:
 P=np.c_[np.ones(len(v)),cross(v)]@M; Q=f2(v,k)@D
 for tag,F in [(n,v),('p'+n,P),('q'+n,Q),('r'+n,np.log1p(k*v*v)),('m'+n,cross(v))]:
  for j,c in enumerate(F.T):X[tag+str(j)]=c
G={}; fn={}
def add(n,c,g):fn[n]=c;G.setdefault(g,[]).append(n)
for n,x in X.items():
 add(n,x,'base:'+n[:-1])
 for name,f in [('sin',np.sin),('tanh',np.tanh),('cos',np.cos),('rat',lambda v:1/(1+v*v)),('log',lambda v:np.log1p(v*v)),('root',lambda v:np.sqrt(1+v*v)),('atan',np.arctan)]:
  for c in [.4,.7,1,1.4,2,2.7,3.5]:add(f'{name}{c}:{n}',f(c*x),'uni:'+n[:-1])
 for j in range(2,5):add(f'{n}^{j}',x**j,'uni:'+n[:-1])
for n,m in combinations(X,2):
 x,v=X[n],X[m]; tag=tuple(sorted([n[:-1],m[:-1]])); gr='pair:'+':'.join(tag)
 if n[:-1][-1]!=m[:-1][-1] and not (len(n)==2 and len(m)==2):continue
 add(f'{n}*{m}',x*v,gr)
 for j in [.5,1,1.5,2]:
  add(f'{n}/(1+{j}{m}^2)',x/(1+j*v*v),gr);add(f'{m}/(1+{j}{n}^2)',v/(1+j*x*x),gr)
  for sg,pm in [(1,'+'),(-1,'-')]:
   for f in [np.sin,np.cos,np.tanh]:add(f'{f.__name__}{j}({n}{pm}{m})',f(j*(x+sg*v)),gr)
for i in (3,4):
 for p in cwr(range(4),i): add('z'+''.join(map(str,p)),np.prod(z[:,p],axis=1),'poly'+str(i))
base=np.c_[np.ones(len(z)),z,cross(z),z*z]; oracle=np.c_[base,Y[:,1:]]; w=np.linalg.lstsq(oracle[a],y[a],rcond=None)[0]; Y1=Y.copy(); Y1[:,2]=(Y[:,2]+(y-oracle[:,:-1]@w[:-1])/w[-1])/2
keys=np.array(list(fn)); fs=np.column_stack(list(fn.values()));
def omp(target,base,j):
 qq=np.linalg.qr(base[a])[0]; F=fs[a]-qq@(qq.T@fs[a]); norms=np.linalg.norm(F,axis=0); valid=norms>1e-7; F=F[:,valid]/norms[valid]; kv=keys[valid]; res=target[a]-qq@(qq.T@target[a]); sel=[]; X0=base
 for i in range(9):
  idx=np.argmax(abs(res@F)); n=kv[idx]; sel.append(n); X0=np.c_[X0,fn[n]]; beta=np.linalg.lstsq(X0[a],target[a],rcond=None)[0];res=target[a]-X0[a]@beta; p=X0[q]@beta
  print(j,i+1,n,round(np.mean(abs(res)),5),round(np.mean(abs(p-target[q])),5))
for j in range(3):omp(Y1[:,j],base,j)
# Test whether the residual is parsimoniously predicted by a single feature family.
qq=np.linalg.qr(base[a])[0];
for j in (1,2):
 target=Y1[:,j]; rr=target[a]-qq@(qq.T@target[a]); ranked=[]
 for g,names in G.items():
  if g.startswith('base:'):continue
  cols=np.column_stack([fn[n] for n in names]); F=cols[a]-qq@(qq.T@cols[a]); U,S,V=np.linalg.svd(F,full_matrices=False); ns=np.linalg.norm(F); best=None
  for alpha in (1e-5,.0003,.01,.1):
   co=(V.T*(S/(S*S+alpha*ns*ns)))@(U.T@rr); w0=np.linalg.lstsq(base[a],target[a]-cols[a]@co,rcond=None)[0]; e0=target[a]-base[a]@w0-cols[a]@co; e1=target[q]-base[q]@w0-cols[q]@co; cand=(np.mean(abs(e1)),np.mean(abs(e0)),g,alpha); best=min(best,cand) if best else cand
  ranked.append(best)
 print('group fits',j,sorted(ranked)[:12])
