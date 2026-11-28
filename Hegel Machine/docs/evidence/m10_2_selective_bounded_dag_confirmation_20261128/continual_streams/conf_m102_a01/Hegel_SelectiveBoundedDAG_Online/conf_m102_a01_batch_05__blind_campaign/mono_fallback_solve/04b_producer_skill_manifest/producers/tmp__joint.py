exec(open('/tmp/explore.py').read().split("p=s['panel_c'];")[0])
from scipy.linalg import solve
from scipy.optimize import least_squares
S=t['traj'];U,d,V=np.linalg.svd(S[:52],full_matrices=False);Y=np.c_[t['y'],S@V[:2].T];print('SVD',d,'corr',np.corrcoef(Y.T)); print('times',s['eval_times']);print('B',b)
for g in [x,z,h,q,f(x)@b]:
 A=np.c_[f(g),Y[:,1:]]; print('y conditional state+linear',np.linalg.lstsq(A[:52],Y[:52,0],rcond=None)[0],np.mean(abs(A[52:]@np.linalg.lstsq(A[:52],Y[:52,0],rcond=None)[0]-Y[52:,0])))
 break
rng=np.random.default_rng(502); splits=[(np.arange(52),np.arange(52,72))]+[(a[:40],a[40:]) for a in [rng.permutation(52) for _ in range(5)]]
# Stationary kernels on affine, saturating and interaction geometries, with profiled ridge.
coords=[x,z,h,q,f(x)@b,f(q)@b]; ks=[]
for j,g in enumerate(coords):
 for typ,v in [('raw',g)]+[(name+str(c),fun(g,c)) for c in [.22,.65,1.25,2.2] for name,fun in [('rat',lambda a,c:a/(1+c*a*a)),('tanh',lambda a,c:np.tanh(c*a)),('sin',lambda a,c:np.sin(c*a))]]:
  G=v@v.T; D=np.maximum(G.diagonal()[:,None]+G.diagonal()[None,:]-2*G,0)
  for ell in [.45,.75,1.15,1.7,2.5,4]: ks.append(((j,typ,'rbf',ell),np.exp(-D/(2*ell**2))))
  for c in [.1,1,10]:ks.append(((j,typ,'poly',c),(1+c*G)**4))
base=1+x@x.T; R=[]
for name,k in ks:
 k=k/np.mean(k.diagonal()[:52]);
 for lam in [.0003,.003,.03]:
  errs=[]; pred=[]
  for tr,te in splits:
   mat=(base+k)[np.ix_(tr,tr)]+lam*np.eye(len(tr)); p=(base+k)[np.ix_(te,tr)]@solve(mat,Y[tr],assume_a='pos');errs.append(np.mean(abs(p-Y[te]),0));pred.append(p)
  e=np.array(errs);R.append((name,lam,e,pred))
for j in [0,1]:
 for sl in [slice(0,1),slice(1,None)]:
  print('kernel',j,sl,[(n,l,e[0,:2].round(4),e[1:,:2].mean(0).round(4)) for n,l,e,p in sorted(R,key=lambda r:r[2][sl,j].mean())[:8]])
 for count in [5,20,50]:
  ps=sorted(R,key=lambda r:r[2][1:,j].mean())[:count];print('ens',j,count,np.mean(abs(np.mean([r[3][0][:,j] for r in ps],0)-Y[52:,j])))
# Exact univariate response grammar with free per-coordinate coefficients, plus products.
from itertools import combinations
ij=np.array(list(combinations(range(4),2))); i,j=ij.T
for ycol in [0,1]:
 out=[]
 for k,g in enumerate(coords):
  for nm,fun in [('rat',lambda a,c:a/(1+c*a*a)),('tanh',lambda a,c:np.tanh(c*a)),('sin',lambda a,c:np.sin(c*a)),('gate',lambda a,c:a*(1+.22*np.tanh(c*a))),('pair',lambda a,c:a)]:
   for mode in [0,1,2,3]:
    def F(c):
     v=fun(g,c[0]); a=np.c_[f(v),v[:,i]*v[:,j],v*v];
     if mode==1:return np.c_[a,g]
     if mode==2:return np.c_[a,g,g[:,i]*g[:,j]]
     if mode==3:return np.c_[a,g,np.prod(v[:,np.array(list(combinations(range(4),3)))],axis=2)]
     return a
    es=[]
    for tr,te in splits:
     def run(c,ret=False):
      A=F(c); w=np.linalg.lstsq(A[tr],Y[tr,ycol],rcond=1e-8)[0];return A[te]@w if ret else A[tr]@w-Y[tr,ycol]
     sol=least_squares(run,[1.25],bounds=(.06,4),max_nfev=40);es.append(np.mean(abs(run(sol.x,True)-Y[te,ycol])))
    out.append((k,nm,mode,es[0],np.mean(es[1:])))
 print('param',ycol,sorted(out,key=lambda a:a[-1])[:12],sorted(out,key=lambda a:a[-2])[:8])
