exec(open('/tmp/readout_probe.py').read().split('for name,p in')[0])
from itertools import combinations_with_replacement
c=s['panel_c']; aa=s['panel_a']; raw=s['train']['x']; lat=np.cbrt(q)
tr=np.arange(54); te=np.arange(54,72)
for nm,x in [('lat',lat),('cubic',q),('raw',raw),('lat squared',lat**2),('latent+B',aff(np.cos(f*lat))@co),('cubic+B',aff(np.cos(f*q))@co)]:
 for fn in [np.cos,np.sin,np.tanh]:
  for degree in [1,2]:
   for target in [y,t]:
    def loss(g,cv=False):
     a=mat(fn(g*x),degree); ii=tr if cv else np.arange(n); jj=te if cv else ii
     return np.mean((a[jj]@np.linalg.lstsq(a[ii],target[ii],rcond=None)[0]-target[jj])**2)**.5
    fs=np.linspace(.1,9,180); vals=[loss(g) for g in fs]; j=np.argmin(vals); opt=minimize_scalar(loss,bounds=(fs[max(0,j-1)],fs[min(len(fs)-1,j+1)]),method='bounded'); print(nm,fn.__name__,degree,'y' if target.ndim==1 else 'traj',*[round(h,5) for h in (opt.x,opt.fun,loss(opt.x,True))])
