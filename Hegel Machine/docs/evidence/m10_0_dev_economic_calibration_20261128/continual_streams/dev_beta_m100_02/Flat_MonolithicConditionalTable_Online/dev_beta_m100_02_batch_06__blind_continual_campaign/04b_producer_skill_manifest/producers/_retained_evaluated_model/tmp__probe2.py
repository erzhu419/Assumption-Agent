exec(open('/tmp/rosette_probe.py').read().split('for j in range(4):')[0])
from scipy.optimize import least_squares
q=(bank==0)&(abs(obs)<5); u=clean&iso&(x[:,0]!=0); z=np.r_[sig[q],x[u,0]]; v=np.r_[obs[q],y[u,0]]
funcs.update({'erf':lambda z,a: __import__('scipy').special.erf(a*z),'asinh':lambda z,a:np.arcsinh(a*z),'exp':lambda z,a:z*np.exp(-a*z*z),'rat4':lambda z,a:z/(1+a*z**4),'sin':lambda z,a:np.sin(a*z)})
configs=[]
for n in funcs:
 for extras in [[],[1],[3],[1,3],[1,3,5]]: configs.append((n+str(extras),lambda z,p,n=n,extras=extras:[funcs[n](z,p[0])]+[z**j for j in extras],1))
for n in funcs:
 for m in funcs:
  if n<=m: configs.append((n+'+'+m,lambda z,p,n=n,m=m:[funcs[n](z,p[0]),funcs[m](z,p[1]),z],2))
for numer in [(1,), (1,3), (1,3,5)]:
 for den in [(2,4),(2,4,6)]:
  for extra in [[],[1]]:
   configs.append((f'quot{numer}/{den}+{extra}',lambda z,p,numer=numer,den=den,extra=extra:[z**j/(1+sum(a*z**k for a,k in zip(p,den))) for j in numer]+[z**j for j in extra],len(den)))
out=[]
for name,f,np_ in configs:
 def fit(p,zz=z,vv=v):
  D=np.column_stack([np.ones(len(zz)),*f(zz,p)]); b=np.linalg.lstsq(D,vv,rcond=None)[0]; return D@b-vv,b
 best=min((least_squares(lambda p:fit(p)[0], [a]*np_,bounds=(.0001,12),max_nfev=120) for a in [.1,.7,1.8]),key=lambda r:np.linalg.norm(r.fun)); p=best.x; e,b=fit(p); out.append((np.linalg.norm(e),name,p,b))
for e,n,p,b in sorted(out,key=lambda row:row[0])[:32]:print('%.8f'%e,n,p.round(5),b.round(5))
