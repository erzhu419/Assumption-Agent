exec(open('/tmp/search06.py').read().split('from scipy.spatial.distance')[0])
Y=Y[:72]; z=z[:72]; a=a[:72]; b=b[:72]; c=c[:72];d=d[:72];q=q[:72];r=r[:72]; x=x[:72];
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from sklearn.linear_model import orthogonal_mp
F={'z':z,'a':a,'b':b,'c':c,'d':d,'q':q,'r':r,'x':x};
for j in [0,1,2,3]:
 for pp in [.5,1,1.25,1.5,2]:
  F[f'warp{j}{pp}']=soft(z,[[.22,.30,.35,.4][j],pp]);
for nm,H in list(F.items())[:8]:
 for g in [.22,.54,1.25]:
  F[f'{nm}R{g}']=H/(1+g*H*H)
from itertools import combinations
base=[]
def chk(label,H):
 P=H[52:]@fit(H[:52],Y[:52]);e=np.mean(abs(P-Y[52:]),axis=0);base.append((e[0],e[1:].mean(),label,rm(H[:52]@fit(H[:52],Y[:52])-Y[:52])))
for nm,h in F.items():
 chk(nm,terms(h,2)[0]);
 for k,G in list(F.items())[:8]:
  if k==nm:continue
  chk(nm+' '+k,np.column_stack((terms(h,2)[0],G,h*G,h*np.roll(G,1,axis=1),h*np.roll(G,2,axis=1),h*np.roll(G,3,axis=1))))
print('dictionary linear best',*sorted(base,key=lambda p:p[0]+p[1])[:15],sep='\n')
for keys in [('z','a'),('z','b'),('a','q'),('z','a','q'),('b','c','d'),('b','q','r'),('z','a','b','q','r'),('z','x','a','b')]:
 A=np.column_stack([F[k] for k in keys]); H,L=terms(A,2); H=H[:,1:]; L=L[1:]; names=[f'{k}{i}' for k in keys for i in range(4)]; m=H[:52].mean(0);sd=H[:52].std(0)+1e-8; H=(H-m)/sd; errs=[]
 for reg in [1e-6,.0001,.002,.01,.1,1]:
  for dg in [1,2]:
   W=H[:,:len(names)] if dg==1 else H; ym=Y[:52].mean(0);P=W[52:]@solve(W[:52].T@W[:52]+reg*np.eye(W.shape[1]), W[:52].T@(Y[:52]-ym),assume_a='pos')+ym;E=np.mean(abs(P-Y[52:]),axis=0);errs.append((E[0],E[1:].mean(),reg,dg));
 print('ridge',keys,min(errs,key=lambda e:e[0]+e[1]));
 for i in [0,8]:
  yy=Y[:,i]; mu=yy[:52].mean();res=[]
  for k in [3,5,7,10,15,20,26]:
   p=orthogonal_mp(H[:52]/np.sqrt(52),yy[:52]-mu,n_nonzero_coefs=k)/np.sqrt(52); inds=np.where(abs(p)>1e-8)[0]; res.append((np.mean(abs(H[52:]@p+mu-yy[52:])),rm(H[:52]@p+mu-yy[:52]),k,[(tuple(names[j] for j in L[t]),round(p[t],3)) for t in inds]));
  print('sparse',i,min(res))
