import numpy as np,pickle,sys
from scipy.optimize import least_squares
from scipy.linalg import qr
sys.path.insert(0,'/holdout_task/_frozen_nodes');import node_00 as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];d=np.load('/tmp/hegel_public_features.npz');X,R,T,y=(d[k] for k in ('X','R','T','y'));z=X[:,:4];u,v=[s['panel_b'][k] for k in ('u','v')];qu=lambda z:m._quad(z)
b,h=m._branch(s['panel_b'],z); w=z-.4*np.sin(h[0]*z)
print('branch linear transform',np.round(np.linalg.lstsq(np.c_[np.ones(len(z)),w],b,rcond=None)[0],3));print('archived diagnostics');print(open('/tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json').read()[:1800]);
rng=np.random.default_rng(19); folds=[(np.arange(52),np.arange(52,72))]+[(np.setdiff1d(np.arange(72),j),j) for j in np.array_split(rng.permutation(72),4)]
def fit(A,B,p): return np.linalg.lstsq(np.r_[A,np.diag(np.sqrt(p))],np.r_[B,np.zeros((len(p),B.shape[1]))],rcond=1e-10)[0]
def evaluate(D,Y,prior,regs,robust=False):
 Y=Y.reshape(len(Y),-1); out=[]
 for l in regs:
  es=[]
  for a,c in folds:
   A=D[a]; scale=np.maximum(np.sqrt(np.mean(A*A,0)),1e-9);A=A/scale; B=Y[a];p=l*np.asarray(prior); W=fit(A,B,p)
   if robust:
    for _ in range(8):
     e=np.sqrt(np.mean((A@W-B)**2,axis=1));wt=(1+(e/max(np.median(e)*2,1e-5))**2)**(-.5);W=fit(A*wt[:,None],B*wt[:,None],p)
   es.append(np.mean(abs((D[c]/scale)@W-Y[c])))
  out.append((l,es[0],np.mean(es[1:]),max(es[1:])))
 return min(out,key=lambda x:x[1]+x[2]+x[3]*.3)
regs=np.geomspace(.0001,10,16)
alt={'b':b,'w':w};
for kind in ('sinatan','sintanh','warp','rational','sincos'):alt[kind]=m._branch(s['panel_b'],z,kind)[0]
for name,f in alt.items():
 for typ,D in [('diagonal',np.c_[z,np.ones(len(z)),z*z,f*f,f]),('full',np.c_[z,np.ones(len(z)),qu(z),qu(f),f]),('mixed',np.c_[z,np.ones(len(z)),qu(z),qu(f),f,z*f])]:
  print(name,typ,'ridge',np.round(evaluate(D,T,np.ones(D.shape[1]),regs),5),'robust',np.round(evaluate(D,T,np.ones(D.shape[1]),regs,True),5))
for name,D,Y in [('X',X,T),('R',R,y)]:
 for l in (1e-5,1e-4,.001,.01,.1):
  es=[]
  for a,c in folds:
   A=D[a];scale=np.maximum(np.sqrt(np.mean(A*A,0)),1e-9);A=A/scale;B=Y[a].reshape(len(a),-1);W=fit(A,B,np.full(D.shape[1],l));
   for k in range(35):
    p=l/(.01+np.linalg.norm(W,axis=1)/np.sqrt(B.shape[1]))**1.25;W=fit(A,B,p)
   es.append(np.mean(abs((D[c]/scale)@W-Y[c].reshape(len(c),-1))))
  print('adaptive',name,l,np.round(es,5))
for opt in [('diagonal',np.r_[0:5,5,9,12,14,15,19,22,24,25:29]),('Bdiag',np.r_[0:5,15,19,22,24,25:29]),('BdiagQ',np.r_[0:15,15,19,22,24,25:29])]:
 D=X[:,np.unique(opt[1])];print(opt[0],np.round(evaluate(D,T,np.ones(D.shape[1]),regs),5))
