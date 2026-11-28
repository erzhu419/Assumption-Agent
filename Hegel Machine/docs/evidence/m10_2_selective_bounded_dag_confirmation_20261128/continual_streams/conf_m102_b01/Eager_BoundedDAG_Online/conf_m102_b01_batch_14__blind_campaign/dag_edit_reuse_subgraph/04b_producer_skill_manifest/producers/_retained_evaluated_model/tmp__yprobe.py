exec(open('/tmp/compose.py').read().split('for j in [')[0]); from scipy.optimize import least_squares
z=V['b']; n=len(Y); t=s['eval_times']; rep=[np.random.default_rng(j).permutation(n)[:20] for j in range(5)];
def cv(x,y):
 out=[]
 for ii in rep:
  jj=np.setdiff1d(np.arange(n),ii); a=x[jj]; coef=np.linalg.lstsq(a,y[jj],rcond=1e-8)[0]; out.append(np.mean(abs(x[ii]@coef-y[ii])))
 return np.round([np.mean(out),max(out)],5)
def phi(a):return np.sinh(np.arcsinh(a)/3)
rows=[]
for g in [0.8,1.4,2.15,3.,4.6,7.]:
 for f in [np.arcsinh,np.arctan,np.tanh,phi]:
  v=f(g*z); d=ab(np.c_[z,v]);
  for j,e in enumerate([d,np.c_[d,Q(v)],np.c_[d,Q(z)],np.c_[d,Q(v),V['c'],V['d']],ab(np.c_[v,Q(v)]),ab(np.c_[z,Q(z)])]):rows.append((cv(e,Y),g,f.__name__,j))
print('interaction CV',sorted(rows,key=lambda e:e[0][1])[:15]); print('sorted y, B2', np.round(np.c_[Y,z[:,2]][np.argsort(Y)].T,3))
for transform in [lambda x:x, np.arcsinh, np.arctan, np.tanh, phi]:
 v=transform(2.15*z); a=ab(np.c_[z,v]); q=np.linalg.lstsq(a,Y,rcond=None)[0]; m=a@q; print('residual',transform.__name__,np.round(np.c_[Y,m,Y-m][np.argsort(m)][::3],3).tolist())
# Fit an outer nonlinear response to a low-dimensional latent readout.
F={'id':lambda x:x,'sin':np.sin,'tanh':np.tanh,'atan':np.arctan,'asinh':np.arcsinh,'inv3':phi,'rat':lambda x:x/(1+x*x),'sat':lambda x:x/np.sqrt(1+x*x)}
results=[]
for g in [1.5,2.15,3.3,5.0]:
 for nm,f in F.items():
  v=f(g*z)
  for ft,d in [('z',ab(z)),('v',ab(v)),('zv',ab(np.c_[z,v])),('zq',ab(np.c_[z,Q(z)])),('vq',ab(np.c_[v,Q(v)]))]:
   if nm=='id' and g!=1.5: continue
   jj=np.where(idx)[0]; ii=np.where(~idx)[0]; ds=d[jj]; yy=Y[jj]; base=np.linalg.lstsq(ds,yy,rcond=1e-8)[0]
   for nm2,f2 in F.items():
    if nm2=='id': continue
    def pr(p,xx):return p[0]+p[1]*f2(xx@p[2:]);
    best=None
    for k in (0.6,1.5,3.):
     p=least_squares(lambda p:pr(p,ds)-yy,np.r_[0,1/k,base*k],max_nfev=100)
     if best is None or p.cost<best.cost:best=p
    results.append((np.mean(abs(pr(best.x,d[ii])-Y[ii])),np.mean(abs(best.fun)),g,nm,ft,nm2))
print('outer results', sorted(results)[:18]); print('trajectory asinh family',[(g,cv(ab(np.c_[z,np.arcsinh(g*z)]),T).tolist()) for g in [1.5,1.85,2.15,2.6]]); print('trajectory inv3 family',[(g,cv(ab(np.c_[z,phi(g*z)]),T).tolist()) for g in [1.5,1.85,2.15,2.6]])
