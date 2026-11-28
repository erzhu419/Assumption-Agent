from compose_probe import *
from scipy.optimize import least_squares
from scipy.linalg import svd
print('times',t,'telemetry',s['telemetry_vec']);print('summary match',y.mean(),y.std(),z[:,-1].mean(),z[:,-1].std());
for lab,p in [('B',s['panel_b']),('C2',{'u':s['panel_c']['u'],'v':s['panel_c']['v2']})]:
 u,v=p['u'],p['v'];
 for typ,f in [('lin',lambda u:bias(u)),('bil',bil),('poly2',lambda u:poly(u,2)),('cos',lambda u:cos(u,2.4)),('sin',lambda u:bias(np.c_[u,np.sin(2.4*u)]))]:
  X=f(u); print(lab,typ,np.std(X@fit(X,v)-v))
main={'x':x,'r':raw,'c':C(x),'d':D(x),'ox':np.cos(2.4*x),'or':np.cos(2.4*raw),'cr':C(raw),'dr':D(raw)}
for k in list(main): main[k+'2']=poly(main[k],2)[:,5:]
F={k:bias(a) for k,a in main.items()};F.update({k+l:bias(np.c_[main[k],main[l]]) for k,l in combinations(main,2)});
for k in ['cd','crdr','xd','rd']:
 a,b=(main['cr'],main['dr']) if k=='crdr' else [main[j] for j in k];F[k+'prod']=bias(np.c_[a,b,(a[:,:,None]*b[:,None,:]).reshape(len(a),-1)]);
za=z[:52];y0=y[:52];
for target in [z,y,np.c_[y,z]]:
 rows=[]
 for k,a in F.items():
  yh=a@fit(a[:52],target[:52]); rr=yh-target; rows.append((np.std(rr[:52]),np.mean(abs(rr[52:])),k))
 print('best spatial',sorted(rows)[:12]);
for l,f in [('e',lambda k:-np.expm1(-k*t)),('r',lambda k:t/(1+k*t))]:
 for k in [.4,.8,1.7,2.4]:
  h=f(k); Y=za@h/(h@h);E=za-Y[:,None]*h;print(l,k,'spatial amp',[(a,round(np.std(F[a][:52]@fit(F[a][:52],Y)-Y),4)) for a in ['x','r','c','d','cr','dr','cd','crdr']])
from itertools import product
E=lambda t,k:-np.expm1(-k*t); R=lambda t,k:t/(1+k*t)
for key1,key2 in product(['c','d','cr','dr','cd','crdr','r2','x2','cdprod'],repeat=2):
 a,b=F[key1],F[key2]
 def run(p,all=False):
  H=np.c_[(a[:,None,:]*E(t,p[0])[None,:,None]).reshape(-1,a.shape[1]),(b[:,None,:]*R(t,p[1])[None,:,None]).reshape(-1,b.shape[1])];w=fit(H[:52*8],za.ravel());return (H@w).reshape(-1,8)-z if all else H[:52*8]@w-za.ravel()
 p=least_squares(run,[1.7,.4],bounds=(.03,7),max_nfev=45).x;err=run(p,True)
 if 'scores' not in globals():scores=[]
 scores.append((np.std(err[:52]),np.mean(abs(err[52:])),key1,key2,p))
print('time-space',sorted(scores)[:15]);print('Y~z',[(k,metrics(np.c_[F[k],z],y)) for k in ['c','d','cd','cr','dr','crdr','x','r']]);print('Y~z alone',metrics(bias(z),y));print('Y~z correlations',np.corrcoef(np.c_[y,z].T)[0]);
