exec(open('/tmp/latent.py').read().split("print('cal'")[0])
from scipy.optimize import least_squares
from sklearn.linear_model import OrthogonalMatchingPursuit

def library(Z):
 X=Z.T; d={f'z{i}':a for i,a in enumerate(X)}
 for i in range(4):
  for j in range(i,4):
   d[f'z{i}*z{j}']=X[i]*X[j]
   for k in range(j,4):d[f'z{i}*z{j}*z{k}']=X[i]*X[j]*X[k]
  for j in range(i+1,4):
   for sign in [1,-1]: d[f'z{i}{sign:+}z{j}']=X[i]+sign*X[j]
 L=dict(d)
 for n,a in d.items():
  if n.count('*')>=2:continue
  for k in [.5,1,1.25,2,3.14]:
   for nm,fn in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('abs',abs),('rat',lambda x:x/(1+x*x)),('relu',lambda x:np.maximum(x,0))]:L[f'{nm}({k}*{n})']=fn(k*a)
 return L

def greedy(L,Y,I,J,steps=8,base=None):
 names=list(L); F=np.array(list(L.values())).T; base=np.ones((len(F),1)) if base is None else base; sel=[]
 for i in range(steps):
  A=np.c_[base,F[:,sel]]; Q=np.linalg.qr(A[I])[0]; E=Y[I]-Q@(Q.T@Y[I]); G=F[I]-Q@(Q.T@F[I]); norm=np.linalg.norm(G,axis=0); sc=np.abs(E@G)/np.maximum(norm,1e-9);sc[norm<1e-8]=-1; sel.append(np.argmax(sc)); A=np.c_[base,F[:,sel]]; c=ls(A[I],Y[I]); print(names[sel[-1]],np.round([np.mean(abs((A@c-Y)[I])),np.mean(abs((A@c-Y)[J]))],4), np.round(c,3) if i in (2,steps-1) else '',end='; ')
 print()

I=np.arange(52);J=np.arange(52,72);U,S,V=np.linalg.svd(t[I],full_matrices=False); T=t@V[:3].T
for Z in (z,x):
 L=library(Z)
 for j,Y in enumerate([y,*T.T]):print('train target',j);greedy(L,Y,I,J,10)
for pn in ['panel_a','panel_b','panel_c']:
 p=s[pn]; u=p['u'];L=library(u); I,J=(np.where(g)[0][:35],np.where(g)[0][35:]) if pn=='panel_a' else (np.arange(18),np.arange(18,24))
 for vn,v in p.items():
  if vn=='u':continue
  for j,Y in enumerate(v.T):
   B=aff(u) if pn!='panel_a' else np.c_[np.ones(len(u)),u[:,j],u[:,(j+1)%4],u[:,j]*u[:,(j+1)%4]]
   print(pn,vn,j,np.std((v-aff(u)@ls(aff(u),v))[:,j])); greedy(L,Y,I,J,3,B)
   if pn!='panel_a':
    for nm,fn in [('tanh',np.tanh),('sin',np.sin),('rat',lambda a:a/(1+.22*a*a)),('linear',lambda a:a)]:
     sol=least_squares(lambda c:fn(B[I]@c)-Y[I],ls(B[I],Y[I])); print(nm,np.round([np.std(sol.fun),np.mean(abs(fn(B[J]@sol.x)-Y[J]))],3),end='; ')
    print()
print('y residual linear state',np.std(y-np.c_[aff(z),T]@ls(np.c_[aff(z),T],y)))
print('time fits')
for name,func in [('log',lambda p:np.log1p(p*s['eval_times'])),('exp',lambda p:1-np.exp(-p*s['eval_times'])),('rat',lambda p:s['eval_times']/(1+p*s['eval_times'])),('sqrt',lambda p:np.sqrt(1+p*s['eval_times'])-1)]:
 sol=least_squares(lambda p:p[0]*func(p[1])-V[0],[-1,1],bounds=([-100,1e-6],[100,100]));print(name,sol.x,sol.fun)
