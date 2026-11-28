exec(open('/tmp/panel_probe.py').read().split('for name,(fun,box)')[0]); import itertools as it
pairs=list(it.combinations(range(4),2)); n4=np.arange(4); p=np.roll(n4,-1); m=np.roll(n4,1)
def cv(name,fun,box=None):
 q,c=fit(fun,box,np.arange(n)); errors=[]
 for J in folds:
  qi,ci=fit(fun,box,np.setdiff1d(np.arange(n),J)); errors.append(np.abs(fun(u[J],qi)@ci-v[J]))
 print(f'{name:25} {q:8.4f} {np.mean(abs(fun(u,q)@c-v)):9.6f} {np.mean(errors):9.6f} {max(np.mean(e) for e in errors):9.6f}',flush=True)
 return c
base=lambda x,q:b.one(x)
for label,T in {'cyc product':lambda x:x*x[:,p], 'all products':lambda x:np.c_[*[x[:,i]*x[:,j] for i,j in pairs]], 'pure cubic':lambda x:x**3,'cyc x^2*y':lambda x:x*x*x[:,p],'cyc x*y^2':lambda x:x*x[:,p]**2,'cyc sum cubed':lambda x:(x+x[:,p])**3,'cyc diff cubed':lambda x:(x-x[:,p])**3,'triple':lambda x:x*x[:,p]*x[:,m],'cyc antisym':lambda x:x*x[:,p]*(x-x[:,p]),'cyc sym':lambda x:x*x[:,p]*(x+x[:,p]),'cyc mixed quadratic':lambda x:x*x[:,p]+.5*x*x[:,m]}.items():
 cv(label,lambda x,q:np.c_[base(x,q),T(x)])
for label,T in {'own':lambda x:x,'cyc product':lambda x:x*x[:,p], 'cyc sum':lambda x:x+x[:,p], 'cyc diff':lambda x:x-x[:,p], 'cyc sum half':lambda x:x+.5*x[:,p], 'cyc diff half':lambda x:x-.5*x[:,p]}.items():
 for tag,h in [('sin',lambda x,q:np.sin(q*x)),('cos',lambda x,q:np.cos(q*x)),('atan',lambda x,q:np.arctan(q*x)),('rat',lambda x,q:x/(1+q*x*x))]:
  cv(label+' '+tag,lambda x,q:np.c_[base(x,q),h(T(x),q)],(.05,6))
for K in [2,3]:
 terms=[t for d in range(2,K+1) for t in it.combinations_with_replacement(range(4),d)]; D=np.c_[*[np.prod(u[:,t],axis=1) for t in terms]]; B=b.one(u); resid=lambda a:a-B@lin(B,a); D=resid(D); norms=np.linalg.norm(D,axis=0); D/=norms; R=resid(v)
 print('polynomial OMP',K)
 for j in range(4):
  rr=R[:,j]; sel=[]
  for _ in range(5):
   k=np.argmax(abs(D.T@rr)); sel.append(k); coef=lin(D[:,sel],R[:,j]);rr=R[:,j]-D[:,sel]@coef
   print(j,terms[k],round(np.linalg.norm(rr),5),np.round(coef/norms[sel],4),end='; ')
  print()
