exec(open('/tmp/operators.py').read().split('res=[]')[0])
from scipy.optimize import least_squares
from scipy.linalg import qr
z0=B(A(x)); tv=Y[:,1:]; rng=np.random.default_rng(228)
splits=[(np.arange(52),np.arange(52,72))]+[(p[:52],p[52:]) for p in (rng.permutation(72) for _ in range(9))]
for name,Z in [('raw',x),('AB',z0)]:
 for degree in [2,3]:
  for typ in ['radial','diagonal','rank1','rank2']:
   if degree==3 and typ=='rank2': continue
   E=[]
   for a,b in splits:
    z=(Z-Z[a].mean(0))/Z[a].std(0); p=np.c_[aff(z),*[mono(z,k) for k in range(2,degree+1)]]; pa=p[a]; n=len(a)
    _,_,v=np.linalg.svd(tv[a],full_matrices=False); v=v[:2]; ty=np.c_[Y[:,0],tv@v.T]; preds=[]
    for j,y in enumerate(ty.T):
     if j==2: preds.append(np.zeros(len(b)));continue
     ya=y[a]; dim={'radial':1,'diagonal':4,'rank1':4,'rank2':8}[typ]; reg=.025 if degree==3 else .004
     G=np.diag(np.r_[np.zeros(5),np.full(10,.002),np.full(pa.shape[1]-15,reg)])**.5
     def den(w,z):
      if typ=='radial':return 1+w[0]**2*np.sum(z*z,axis=1)
      if typ=='diagonal':return 1+(z*z)@(w*w)
      return 1+np.sum((z@w.reshape(-1,4).T)**2,axis=1)
     def f(w,ret=False):
      d=pa/den(w,z[a])[:,None]; c=np.linalg.lstsq(np.r_[d,G],np.r_[ya,np.zeros(len(G))],rcond=None)[0]
      if ret:return (p[b]@c)/den(w,z[b]);
      return np.r_[d@c-ya,G@c,.024*w]
     starts=[np.full(dim,.18)] if typ in ('radial','diagonal') else [.20*np.random.default_rng(i).normal(size=dim) for i in range(3)]
     opt=min([least_squares(f,w,max_nfev=55,bounds=(-2,2)) for w in starts],key=lambda r:r.cost)
     preds.append(f(opt.x,True))
    E.append([abs(preds[0]-Y[b,0]).mean(),abs(np.array(preds[1:]).T@v-tv[b]).mean()]);
   E=np.array(E); print(name,degree,typ,np.round(np.r_[E[0],E.mean(0),E.max(0)],5),flush=True)
