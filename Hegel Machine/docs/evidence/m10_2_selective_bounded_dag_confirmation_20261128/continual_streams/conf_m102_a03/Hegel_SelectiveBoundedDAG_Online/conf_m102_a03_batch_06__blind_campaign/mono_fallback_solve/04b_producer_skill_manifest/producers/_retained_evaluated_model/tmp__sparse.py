import io,contextlib
with contextlib.redirect_stdout(io.StringIO()):
 from readout import *
from sklearn.linear_model import lars_path
from scipy.special import erf

def dictionary(z,b,c,d,p,q):
 stuff={}
 def add(n,w):
  w=np.asarray(w).reshape(len(z),-1)
  for j,col in enumerate(w.T): stuff[f'{n}:{j}']=col
 D={'z':z,'b':b,'c':c,'d':d,'p':p,'q':q}
 for n,w in D.items():
  add(n,w); add(n+'quad',pol(w,2)[:,5:]); add(n+'cube',pol(w,3)[:,15:]);
  for k in [.75,1.25,2.,2.8,3.5,5.]:
   add(f'sin{k}{n}',np.sin(k*w)); add(f'cos{k}{n}',np.cos(k*w)); add(f'tanh{k}{n}',np.tanh(k*w)); add(f'rat{k}{n}',w/(1+(k*w)**2));
  for i,j in combinations(range(4),2):
   for sign in [-1,1]:
    h=w[:,i]+sign*w[:,j]
    for k in [1.25,2.5,4.]:
     add(f'sin{k}{n}{i}{j}{sign}',np.sin(k*h)); add(f'cos{k}{n}{i}{j}{sign}',np.cos(k*h)); add(f'tanh{k}{n}{i}{j}{sign}',np.tanh(k*h))
 for n,w in D.items():
  for m,g in D.items():
   if n>=m:continue
   add(n+m,w[:,:,None]*g[:,None,:]);
   if 'b' in (n,m):
    add(n+m+'w2',w[:,:,None]**2*g[:,None,:]); add(n+m+'g2',w[:,:,None]*g[:,None,:]**2)
    for k in [1.25,2.5,4.]:
     add(n+m+f'sinw{k}',np.sin(k*w)[:,:,None]*g[:,None,:]); add(n+m+f'sing{k}',w[:,:,None]*np.sin(k*g)[:,None,:]); add(n+m+f'tanhw{k}',np.tanh(k*w)[:,:,None]*g[:,None,:]); add(n+m+f'tanhg{k}',w[:,:,None]*np.tanh(k*g)[:,None,:])
 return np.array(list(stuff)),np.column_stack(list(stuff.values()))
N,A=dictionary(z,b,c,d,p,q); np.savez('/tmp/latent.npz',x=x,y=y,t=t,z=z,b=b,c=c,d=d,p=p,q=q)
for target in [y,t@V[0]]:
 for jj in [np.arange(52),np.random.default_rng(10).permutation(72)[:52],np.random.default_rng(17).permutation(72)[:52]]:
  kk=np.setdiff1d(np.arange(72),jj); B=(A-A[jj].mean(0))/np.maximum(A[jj].std(0),1e-10); Y=target-target[jj].mean(); _,act,C=lars_path(B[jj],Y[jj],method='lasso',max_iter=42); e=np.mean(abs(B[kk]@C-Y[kk,None]),axis=0); i=e.argmin(); ids=np.flatnonzero(abs(C[:,i])>1e-9); print('lars',i,e[i],list(zip(N[ids],np.round(C[ids,i],4))))
  base=np.ones((72,1)); ids=[]; opt=[]
  for i in range(25):
   L=np.linalg.qr(base[jj])[0]; instead=B[jj]-L@(L.T@B[jj]); r=Y[jj]-L@(L.T@Y[jj]); w=abs(instead.T@r)/np.maximum(np.linalg.norm(instead,axis=0),1e-8); w[ids]=0; k=w.argmax(); ids.append(k);base=np.c_[base,B[:,k]]; co=fit(base[jj],target[jj]); opt.append((np.mean(abs(base[kk]@co-target[kk])), np.std(base[jj]@co-target[jj]),N[ids].tolist()))
  print('omp best',min(opt,key=lambda x:x[0]));print('omp prefix',[(r[0],r[1],r[2][-1]) for r in opt[:12]])
