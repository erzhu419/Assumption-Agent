import numpy as np,pickle
from scipy.optimize import least_squares
from fit14 import maps,aff,fit
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; Y=np.c_[t['y'],t['traj']]; tt=s['eval_times']; M=maps(s,t['x']); z=M['z']
def basis(p):return -np.expm1(-np.exp(p[:,None])*tt)
def err(p):
 h=basis(p);return (Y[:,1:]@fit(h.T,Y[:,1:].T).T*0).ravel() if False else (fit(h.T,Y[:,1:].T).T@h-Y[:,1:]).ravel()
p=least_squares(err,np.log([.45,2.]),bounds=(np.log([.015,.1]),np.log([3,15]))).x
H=basis(p); A=fit(H.T,Y[:,1:].T).T
print('modes',np.exp(p),np.std(err(p)), 'y from A',np.mean(abs(aff(A)@fit(aff(A)[:52],Y[:52,0])-Y[:,0])),flush=True)
Y=np.c_[Y[:,0],A,Y[:,-1]]
ops={}
for name,k,l,fn in [('B','panel_b','v',lambda x:np.sign(x)*np.maximum(abs(x)-.35,0)**1.5),('C','panel_c','v1',lambda x:x/(1+.62*abs(x)**1.07)**1.57),('D','panel_c','v2',lambda x:np.logaddexp(0,2.2*x))]:
 u,v=s[k]['u'],s[k][l]; co=fit(aff(fn(u)),v); print(name,'op',np.round(co,3));ops[name]=lambda x,fn=fn,co=co: aff(fn(x))@co
blocks={'z':z}; scales=[.22,.35,1.,1.25,2.15]
def add(k,x):
 if np.max(np.abs(x))<1e5 and np.all(np.isfinite(x)):blocks[k]=x
for a,f in ops.items():add(a,f(z))
base=list(blocks.items())
for a,x in base:
 for b,y in base:
  if a==b:continue
  for c,f in ops.items():
   add(f'{c}({a})',f(x)); add(f'{c}({a}*{b})',f(x*y));
   for i in [0,1,2,3]:
    q=np.roll(y,i,axis=1)
    for sc in [.22,1.,1.25,2.15]:
     add(f'{c}({a}/(1+{sc}{b}r{i}²))',f(x/(1+sc*q*q)));add(f'{c}({a}+{sc}{b}r{i})',f(x+sc*q));add(f'{c}({a}-{sc}{b}r{i})',f(x-sc*q))
for a,x in list(blocks.items()):
 if len(a)<=4:
  for b,y in list(blocks.items()):
   if len(b)>4:continue
   for i in range(4):
    q=np.roll(y,i,axis=1)
    for sc in scales:
     add(f'{a}/(1+{sc}{b}r{i}²)',x/(1+sc*q*q));add(f'{a}*tanh({sc}{b}r{i})',x*np.tanh(sc*q))
    add(f'{a}*{b}r{i}',x*q)
  for c,f in ops.items():add(f'{c}({a})',f(x))
print('blocks',len(blocks),flush=True)
# rank each block by held-in fit on first 52, keeping held-out for reporting only
for key,b in [('-',None),*base]:
 scores=[]
 for k,x in blocks.items():
  h=aff(x if b is None else np.c_[b,x]); er=abs(h@fit(h[:52],Y[:52])-Y);scores.append((k,er[:52].mean(0),er[52:].mean(0)))
 for j in range(4):
  print(key,j,[(k,round(a[j],5),round(b[j],5)) for k,a,b in sorted(scores,key=lambda s:s[1][j])[:6]],flush=True)
np.savez('/tmp/blocks14.npz',**blocks,A=A,Y=Y)
