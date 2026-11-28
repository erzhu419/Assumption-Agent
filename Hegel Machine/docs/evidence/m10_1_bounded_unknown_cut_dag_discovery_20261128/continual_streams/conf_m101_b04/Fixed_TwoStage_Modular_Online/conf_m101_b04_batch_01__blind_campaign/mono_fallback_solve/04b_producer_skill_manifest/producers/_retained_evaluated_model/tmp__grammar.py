exec(open('/tmp/probe.py').read().split("print('A robust")[0])
from scipy.linalg import qr
from scipy.optimize import minimize_scalar
from itertools import product
p=s['panel_c']; u=p['u']; Z=lin(u); Q=poly(u,2)
print('quadratic coefficients C2',fit(np.c_[Z,Q],p['v2']).round(4)); print('quadratic order',list(combinations_with_replacement(range(4),2)))
dirs=np.array([a for a in product(range(-2,3),repeat=4) if 0<np.count_nonzero(a)<=3 and next(z for z in a if z)!=-1 and next(z for z in a if z)!=-2 and np.gcd.reduce(a)==1]);
funcs={'sin':np.sin,'cos':np.cos,'tanh':np.tanh,'rat':lambda x:1/(1+x*x),'sat':lambda x:x/(1+x*x),'square':np.square,'cube':lambda x:x**3};
for k in ('v1','v2'):
 P=Z if k=='v1' else np.c_[Z,Q]; N=qr(P,mode='full')[0][:,P.shape[1]:].T; V=p[k]; print(k,'residual singular values',np.linalg.svd(N@V,compute_uv=False));
 for j in range(4):
  cols=[]; vals=[]; base=P; r=V[:,j];
  for _ in range(3):
   N=qr(base,mode='full')[0][:,base.shape[1]:].T; residual=N@r; scores=[]
   for name,f in funcs.items():
    for w in [.4,.7,1.,1.3,1.6,2.,2.5,3.,4.]:
     H=f(w*(u@dirs.T)); T=N@H; c=abs(residual@T)/np.maximum(np.linalg.norm(T,axis=0),1e-10); idx=c.argmax(); scores.append((c[idx],name,w,idx,H[:,idx]));
   _,name,w,idx,h=max(scores,key=lambda a:a[0]); base=np.c_[base,h]; cols.append((name,w,dirs[idx].tolist())); vals.append(np.sqrt(np.mean((r-base@fit(base,r))**2)))
  print(j,list(zip(cols,np.round(vals,6))))
