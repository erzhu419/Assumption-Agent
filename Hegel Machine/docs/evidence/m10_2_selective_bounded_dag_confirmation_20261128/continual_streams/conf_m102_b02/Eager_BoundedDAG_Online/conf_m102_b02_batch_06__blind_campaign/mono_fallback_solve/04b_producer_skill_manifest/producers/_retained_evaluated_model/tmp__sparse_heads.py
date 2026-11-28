from probe import *
from scipy.linalg import solve
from scipy.optimize import minimize_scalar
from itertools import combinations,product
import importlib.util
p='/tmp/hegel_continual_archive/conf_m102_b02_batch_05_solver.py'; q=importlib.util.spec_from_file_location('old',p); old=importlib.util.module_from_spec(q); q.loader.exec_module(old)
def coords(s,x):
 z=old.coords(s,x); p=s['panel_b']; h=lambda a:D(np.sign(a)*np.maximum(abs(a)-.35,0)**1.5); b=h(z)@fit(h(p['u']),p['v']); p=s['panel_c']; f=lambda a:D(a*np.roll(a,-1,axis=1)); c=f(z)@fit(f(p['u']),p['v1']); g=lambda a:D(np.c_[a,np.cos(1.25*a)]); d=g(z)@fit(g(p['u']),p['v2']);return z,b,c,d

def build(z,b,c,d):
 F=[]; N=[]; G=[]
 def add(a,n,g):
  a=np.asarray(a); a=a[:,None] if a.ndim==1 else a
  F.extend(a.T);N.extend([str(n)+':'+str(j) for j in range(a.shape[1])]);G.extend([g]*a.shape[1]);
 def uni(a,n,g):
  add(a,n,g)
  for k in [.6,1.25,2.,3.,4.]:
   for fun,fn in [('s',np.sin),('c',np.cos),('h',np.tanh),('r',lambda u:u/(1+u*u)),('e',erf)]:add(fn(k*a),(n,fun,k),g)
  for k in [0,.15,.22,.35,.5,.7]:
   for pw in [1,1.5,2]:
    h=np.maximum(abs(a)-k,0)**pw;add(h,(n,'even',k,pw),g);add(h*np.sign(a),(n,'odd',k,pw),g)
 for n,a in zip('zbcd',[z,b,c,d]):uni(a,n,0 if n in 'zb' else 3)
 for k in [-2.,-1.25,-1.,-.7,-.3,.22,.3,.7,1.,1.25,2.]:
  uni(z+k*b,('zb',k),1)
  for i in range(4):uni(z[:,i,None]+k*b,('allzb',i,k),2)
 for n,a in [('zz',z),('bb',b),('cc',c),('dd',d)]:
  for k in [-1.25,-1.,-.7,.7,1.,1.25]:
   Q=np.column_stack([a[:,i]+k*a[:,j] for i,j in combinations(range(4),2)]);uni(Q,(n,k),2 if n in ['zz','bb'] else 4)
 for k in [-1.,-.22,.22,1.]:
  for n,a in [('c',c),('d',d)]:uni(z+k*a,('z'+n,k),4);uni(b+k*a,('b'+n,k),4)
 for n,a,h in [('zz',z,z),('zb',z,b),('bb',b,b),('zc',z,c),('zd',z,d),('bc',b,c),('bd',b,d)]:
  for i,j in product(range(4),repeat=2):
   if n in ['zz','bb'] and i>j:continue
   v=a[:,i]*h[:,j];add(v,(n,i,j,'prod'),1 if n in ['zz','zb','bb'] else 4)
   for k in [.22,1.25]:
    for u,ln in [(a[:,i],'a'),(h[:,j],'b')]:
     add(v*u,(n,i,j,ln,'3prod'),2); add(v/(1+k*u*u),(n,i,j,ln,'rprod',k),2);add(a[:,i]/(1+k*h[:,j]**2),(n,i,j,'quot',k),2)
 return np.array(F).T,np.array(N),np.array(G)

def run(F,N,G,base,Y,folds):
 opts=[]
 for gr in [0,1,2,3,4]:
  for bias in [0,.5]:
   for order in [0,1,2]:
    scores=[]
    for a,i in folds:
     ind=np.where(G<=gr)[0];Q=F[:,ind];A=base[:,:1] if order==0 else base[:,:5] if order==1 else base; A=A.copy(); P=np.linalg.pinv(A[a]); Q=Q-A@(P@Q[a]); norm=np.linalg.norm(Q[a],axis=0);good=norm>1e-5; Q=Q[:,good]/norm[good];ind=ind[good]; prior=(1+bias*G[ind]); pred=[];names=[];es=[]
     for yy in Y.T:
      init=A@(P@yy[a]);y=(yy-init)[a];t=Q[a];sel=[];r=y.copy();oo=[]
      for it in range(18):
       mag=abs(t.T@r)/prior;mag[sel]=-1; j=np.argmax(mag);sel.append(j);w=fit(t[:,sel],y);r=y-t[:,sel]@w;pp=init+Q[:,sel]@w;oo.append(pp[i]);
      pred.append(oo);names.append(N[ind[sel]]);es.append(np.sqrt(np.mean(r*r)))
     pred=np.array(pred); scores.append(abs(pred-Y[i].T[:,None,:]).mean(2));
    E=np.array(scores);opts.append((E,gr,bias,order,names))
 for l in range(2):
  rows=[(E[:,l,j].mean(),E[0,l,j],E[:,l,j].max(),gr,b,o,j+1) for E,gr,b,o,_ in opts for j in range(E.shape[2])];print('sparse',l,'mean',sorted(rows)[:8]);print('gate',sorted(rows,key=lambda r:r[1])[:6]);
  best=min(rows);print('selected',next(n[l][:best[-1]] for _,g,b,o,n in opts if (g,b,o)==best[3:6]))
if __name__=='__main__':
 tr=s['train']; z,b,c,d=coords(s,tr['x']);T=tr['traj'];y=tr['y']; U,S,V=np.linalg.svd(T,full_matrices=False); Y=np.c_[y,(T@V[0])*V[0,-1]]; rng=np.random.default_rng(777); folds=[(np.arange(52),np.arange(52,72))]+[(q[:52],q[52:]) for q in (rng.permutation(72) for _ in range(5))];
 F,N,G=build(z,b,c,d); print(F.shape,flush=True); base=D(np.c_[z,b]); print('base coef',fit(base,Y)); run(F,N,G,base,Y,folds)
