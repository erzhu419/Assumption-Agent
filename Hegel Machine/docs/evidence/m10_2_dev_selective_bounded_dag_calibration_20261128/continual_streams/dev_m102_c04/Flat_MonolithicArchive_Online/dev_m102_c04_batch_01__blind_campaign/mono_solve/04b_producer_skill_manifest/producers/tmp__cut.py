from dag import *
from scipy.special import expit
from scipy.optimize import least_squares
from itertools import combinations,product
v=np.c_[y,t]; U,S,V=np.linalg.svd(v[:52,1:],full_matrices=False); vv=np.c_[y,t@V[:3].T]; vv=vv/vv[:52].std(0); print('joint singular',S,flush=True)
D=paths(x,1); Z={}
for n,z in D.items():
 Z[n]=z
 for m,f in [('bT',lambda z:z@cb[1:].T),('b0',lambda z:z@cb[1:]),('B2',B),('b2',Bi)]:
  if len(n)==1:Z[n+m]=f(z)
for n,z in Z.copy().items():
 if len(n)<=3 and n[-1:] not in 'QH': Z[n+'C1']=Q(z);Z[n+'C2']=H(z)
r=[]
def run(n,f):
 f=np.c_[np.ones(len(x)),f]; a=f[:52]; w=ls(a,v[:52]); e=np.abs(f@w-v); r.append((e[:52,0].mean(),e[52:,0].mean(),e[:52,1:].mean(),e[52:,1:].mean(),n))
for n,z in Z.items():
 for c in [.22,.5,1.,1.25,2.]:
  for o in [1,-1]:
   p=np.column_stack([z[:,i]+o*c*z[:,j] for i,j in combinations(range(4),2)]); a=np.c_[z,p]
   for k in [1.25,2.4,4.8]:
    for label,b in [('tri',np.c_[np.sin(k*a),np.cos(k*a)]),('tan',np.tanh(k*a)),('hinge',abs(a)),('rat',a/(1+.22*a*a)),('sqr',p*p)]:
     run((n,c,o,k,label),np.c_[z,b]);run((n,c,o,k,label,'skip'),np.c_[x,z,b])
for j in range(4):print('pair',j,*sorted(r,key=lambda a:a[j])[:5],sep='\n',flush=True)
# Seek single-index nonlinear components not necessarily aligned with a panel axis.
from scipy.optimize import minimize
rng=np.random.default_rng(887); P=poly(x,1); feature=[]
for mode in ['sin','hinge','tan','rational','threshold','gated']:
 for ti in [0,1,2,3,4]:
  b=vv[:52,ti:ti+1] if ti<4 else vv[:52,:2]; p=P[:52]; qq=np.linalg.qr(p)[0]; b=b-qq@(qq.T@b)
  def model(w,p):
   l=p@w
   if mode=='sin':return np.c_[np.sin(l),np.cos(l)]
   if mode=='hinge':return abs(l)[:,None]
   if mode=='tan':return np.tanh(l)[:,None]
   if mode=='rational':return (l/(1+l*l))[:,None]
   if mode=='threshold':return expit(18*l)[:,None]
   return p*expit(l)[:,None]
  def obj(w):
   h=model(w,p); h=h-qq@(qq.T@h); return np.sum((h@ls(h,b)-b)**2)
  sol=sorted((minimize(obj,w,method='BFGS',options={'maxiter':85}) for w in np.r_[rng.normal(size=(20,5))*rng.uniform(.5,4,(20,1)),np.c_[np.zeros(4),np.eye(4)*2.4],np.c_[np.zeros(4),cb[1:]*2.4]]),key=lambda o:o.fun)
  for o in sol[:2]:
   f=np.c_[P,model(o.x,P)]; pred=f@ls(f[:52],v[:52]);e=np.abs(pred-v); print('ridge',mode,ti,round(o.fun,4),np.round(o.x,3),np.round([e[:52,0].mean(),e[52:,0].mean(),e[:52,1:].mean(),e[52:,1:].mean()],4),flush=True);feature.append(model(o.x,P))
