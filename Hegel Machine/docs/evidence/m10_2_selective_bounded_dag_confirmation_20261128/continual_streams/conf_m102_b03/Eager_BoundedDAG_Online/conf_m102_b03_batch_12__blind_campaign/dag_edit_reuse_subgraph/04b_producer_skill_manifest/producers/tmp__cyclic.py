exec(open('/tmp/probe.py').read().split('for name, aa')[0])
from itertools import product
v=vs[0]; pairs=[(1,0),(2,1),(3,2),(0,3)]; n=len(z)-20
funcs={'sin':np.sin,'tanh':np.tanh,'soft':b._soft,'id':lambda x:x,'rat':lambda x:x/(1+x*x)}
out=[]
for name,f in funcs.items():
 for bits in product(range(2),repeat=4):
  ii,jj=np.array([p[::(-1 if bit else 1)] for p,bit in zip(pairs,bits)]).T
  def mat(x,p):return D(np.c_[x,x[:,ii]*f(p*x[:,jj])])
  def fun(p):
   A=mat(u,p); return (A@ls(A,v)-v).ravel()
  sols=[least_squares(fun,[g],bounds=(.03,6),gtol=1e-12,max_nfev=70) for g in [0.85,1.75,2.9]]
  sol=min(sols,key=lambda x:x.cost); out.append((np.std(sol.fun),name,bits,sol.x,ls(mat(u,sol.x),v)))
for r in sorted(out,key=lambda r:r[0])[:12]: print(r[:4]);
err,name,bits,p,w=min(out,key=lambda r:r[0]);print('best coefs',w)
ii,jj=np.array([t[::(-1 if bit else 1)] for t,bit in zip(pairs,bits)]).T
c=D(np.c_[z,z[:,ii]*funcs[name](p*z[:,jj])])@w
fit=least_squares(lambda p:(lambda A:A@ls(A,vs[1])-vs[1])(D(np.c_[u,np.cos(u*p)])).ravel(),[2.4]); d=D(np.c_[z,np.cos(z*fit.x)])@ls(D(np.c_[u,np.cos(u*fit.x)]),vs[1]); print('n',n,'cos',fit.x)
g=D(np.c_[c,d,(c[:,:,None]*d[:,None,:]).reshape(len(z),16)]);h=D(np.c_[c,d,np.sum(c*d,axis=1)]); print('CV',score(h,tr['y']),score(g,tr['traj']),'full scalar basis',score(g,tr['y']))
