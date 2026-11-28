from compose_probe import *
from scipy.linalg import svd
print('summary',tr['summary'],'y',np.round(y,3)); print('raw/x ranges',[(k,a.min(0),a.max(0)) for k,a in A.items()]);print('x sample',np.round(x[:15],3))
print('C/D weights');u=s['panel_c']['u'];print(np.round(fit(bil(u),s['panel_c']['v1']),3));print(np.round(fit(cos(u,2.4),s['panel_c']['v2']),3))
U,S,V=svd(z[:52],full_matrices=False); zz=z@V[:3].T; target=np.c_[y,zz];
A.update({'s'+k:np.sin(2.4*a) for k,a in list(A.items())});A.update({'o'+k:np.cos(2.4*a) for k,a in list(A.items()) if not k.startswith('s')});
base={f'{k}{i}':a[:,i] for k,a in A.items() for i in range(4)}
F=dict(base)
for k,a in list(A.items()):
 if not k.startswith(('s','o')):
  for fun,fn in [('t',np.tanh),('r',lambda v:v/(1+abs(v))),('q',lambda v:v/(1+v*v))]:
   for i in range(4):F[f'{fun}({k}{i})']=fn(a[:,i]);
for (k,a),(l,b) in combinations_with_replacement(base.items(),2):
 F[k+'*'+l]=a*b
 F[k+'/(1+'+l+'^2)']=a/(1+b*b)
for (k,a),(l,b) in combinations(A.items(),2):
 if not k.startswith(('s','o')) and not l.startswith(('s','o')):
  F[k+'.'+l]=(a*b).sum(1)
labels=list(F); X=np.column_stack(list(F.values())); X-=X[:52].mean(0); X/=np.maximum(X[:52].std(0),1e-12)
for j,Y in enumerate(target.T):
 b=Y[:52].mean(); R=Y[:52]-b; selected=[]
 for it in range(14):
  i=np.argmax(abs(X[:52].T@R));selected.append(i);Z=X[:,selected];w=fit(Z[:52],Y[:52]-b);R=Y[:52]-b-Z[:52]@w
  if it<5 or it in [7,10,13]: print('OMP',j,it,labels[i],round(w[-1],4),round(np.std(R),5),round(np.mean(abs(Y[52:]-b-Z[52:]@w)),5))
print('univariate correlation')
for a in target.T:
 ix=np.argsort(abs(X[:52].T@(a[:52]-a[:52].mean())))[-12:][::-1];print([(labels[i],round(np.corrcoef(X[:52,i],a[:52])[0,1],3)) for i in ix]);
print('time vectors',np.round(V[:3],5))
