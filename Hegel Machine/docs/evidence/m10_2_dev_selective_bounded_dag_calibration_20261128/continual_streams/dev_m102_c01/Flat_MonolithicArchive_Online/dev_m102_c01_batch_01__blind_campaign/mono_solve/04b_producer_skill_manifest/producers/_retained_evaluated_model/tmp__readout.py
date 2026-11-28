exec(open('/tmp/probe.py').read().split('rows=[]')[0])
from scipy.optimize import least_squares
F=p(Q['A-'],2); rng=np.random.default_rng(601); splits=[np.arange(len(x))]+[rng.permutation(len(x)) for _ in range(5)]
print('Y extrema/std:',Y.min(0),Y.max(0),Y.std(0)); print('svd:',np.linalg.svd(t,compute_uv=False)); print('T basis:',sv[:3]); print('raw cubic',np.linalg.lstsq(p(Q['A-'],3),Y,rcond=None)[0].T)
fun={'tanh':np.tanh,'sin':np.sin,'atan':np.arctan,'rat':lambda a:a/np.sqrt(1+a*a),'log':lambda a:np.arcsinh(a),'soft':lambda a:a/(1+abs(a)),'cub':lambda a:a**3}
rows=[]
for j in (0,1):
 yy=Y[:,j]; scale=np.std(yy); yy=yy/scale
 for deg in (1,2):
  F=p(Q['A-'],deg)
  for name,f in fun.items():
   for lin in (0,1,2):
    L=p(Q['A-'],lin); L=L[:,:1] if lin==0 else L
    def predict(par,F,L):return par[0]*f(F@par[1:1+F.shape[1]])+L@par[1+F.shape[1]:]
    es=[]; ps=[]
    for i in splits:
     a,b=i[:52],i[52:]; best=None
     for gain in (.75,1.5,3):
      beta=np.linalg.lstsq(F[a],yy[a],rcond=None)[0]/gain
      r=least_squares(lambda w:predict(w,F[a],L[a])-yy[a],np.r_[gain,beta,np.zeros(L.shape[1])],max_nfev=140)
      if best is None or r.cost<best.cost:best=r
     es.append([np.mean(abs(best.fun))*scale,np.mean(abs(predict(best.x,F[b],L[b])-yy[b]))*scale]);ps.append(best.x)
    rows.append((j,deg,name,lin,np.round(es[0],5),np.round(np.mean(es,0),5),np.round(np.max(es,0),5),np.round(ps[0],4)))
for j in (0,1):
 for row in sorted([r for r in rows if r[0]==j],key=lambda r:r[5][1])[:10]:print(row)
