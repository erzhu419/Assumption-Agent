import numpy as np
from fit14 import aff,fit
from itertools import combinations_with_replacement
m=np.load('/tmp/m14.npz'); z,b,c,d=[m[k] for k in 'zbcd']; Y=m['Y']; n=52
h=aff(np.c_[b,b*np.roll(z,-1,axis=1)])
beta=fit(h[:n],Y[:n]); P=h@beta; e=Y-P
print('state beta',np.round(beta[:,1:],3));print('y beta',beta[:,0]); print('errs',abs(e[:n]).mean(0),abs(e[n:]).mean(0));print('residual y largest',[(i,round(e[i,0],5),np.round(z[i],2),np.round(b[i],2)) for i in np.argsort(-abs(e[:,0]))[:12]])
w=np.load('/tmp/blocks14.npz'); B={k:w[k] for k in w.files if k not in ['A','Y']}; B.update({f'state{i}':P[:,i] for i in [1,2,4,8]}); B['p']=P[:,0]; B.update({'b':b,'c':c,'d':d})
base={}
for k,x in dict(z=z,b=b,c=c,d=d,p=P[:,0:1],bc=b*np.roll(z,-1,axis=1),amps=P[:,[1,8]],th=np.sign(z)*np.maximum(abs(z)-.35,0)**1.5).items():
 for j,a in enumerate(x.T):base[k+str(j)]=a
for (k,x),(l,y) in combinations_with_replacement(base.items(),2):
 B[f'{k}*{l}']=x*y
 for lam in [.22,1.25,2.15]:
  B[f'{k}/(1+{lam}{l}²)']=x/(1+lam*y*y);B[f'{l}/(1+{lam}{k}²)']=y/(1+lam*x*x)
for k,x in base.items():
 for a in [.22,.35,.5,.8,1.25,2.15,2.6,3.5]:
  for nm,f in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('soft',lambda x:np.logaddexp(0,x))]:B[f'{nm}({a}{k})']=f(a*x)
 B[f'{k}^3']=x**3;B[f'{k}^4']=x**4
for i in range(4):
 for j in range(4):
  for k in range(j,4): B[f'b{i}z{j}z{k}']=b[:,i]*z[:,j]*z[:,k]
def greedy(H, group):
 for it in range(7):
  er=Y[:,0]-H@fit(H[:n],Y[:n,0]); cand=[]
  for k,x in group.items():
   x=x.reshape(len(z),-1); rr=x-H@fit(H[:n],x[:n]); g=rr@fit(rr[:n],er[:n]); q=er-g
   cand.append((np.mean(q[:n]**2),k,np.mean(abs(q[:n])),np.mean(abs(q[n:]))))
  top=sorted(cand)[:5]; print('step',it,top,flush=True); k=top[0][1];H=np.c_[H,group[k]]; group.pop(k)
print('groups');greedy(h,B.copy());print('columns');greedy(h,{f'{k}:{j}':col for k,x in B.items() if x.ndim==1 or len(k)<12 for j,col in enumerate(x.reshape(len(z),-1).T)})
