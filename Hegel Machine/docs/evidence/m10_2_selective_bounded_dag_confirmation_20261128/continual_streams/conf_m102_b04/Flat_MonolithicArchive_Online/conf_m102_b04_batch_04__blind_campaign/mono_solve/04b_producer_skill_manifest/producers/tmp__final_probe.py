exec(open('/tmp/stress_batch.py').read().split('for mode in')[0])
from itertools import combinations
from scipy.linalg import solve
D={}
for name,i,j in folds:
 T=m._time_basis(s['eval_times'],z[i]); p=m._profile(u[i],np.c_[y[i],z[i]@T]); q=m._quadratic(u); g=m._design(u,q,p); h=g[:,15:]; h=h/np.max(abs(h[i]),axis=0); g=np.c_[q,h]
 shapes={'cube':u**3,'fifth':u**5,'shapeSquare':h*h,'shapeDiff':(h-u)**2,'uTimesShape':u*h,'stateSq':np.c_[g@np.linalg.lstsq(g[i],z[i]@T,rcond=None)[0]]**2,'mixedCubes':np.column_stack([u[:,k]**2*u[:,l] for k in range(4) for l in range(4) if k!=l]),'triples':np.column_stack([np.prod(u[:,k],axis=1) for k in combinations(range(4),3)]),'shapeProducts':np.column_stack([h[:,k]*h[:,l] for k,l in combinations(range(4),2)])}
 for key,extra in [('base',q[:,:0]), *shapes.items(),('cube-shapeSquare',np.c_[shapes['cube'],shapes['shapeSquare']]),('cube-uTimesShape',np.c_[shapes['cube'],shapes['uTimesShape']])]:
  a=np.c_[g,extra];
  for lq in [0,.3,1,2]:
   for le in ([0] if key=='base' else [0,.05,.2,.8,3]):
    reg=np.r_[np.zeros(5),np.full(10,lq),np.zeros(4),np.full(extra.shape[1],le)]; d=np.diag(np.sqrt(reg)); co=np.linalg.lstsq(np.r_[a[i],d],np.r_[y[i],np.zeros(len(d))],rcond=None)[0]; D.setdefault((key,lq,le),[]).append(a[j]@co)
for key1,key2 in [(('base',0,0),('cube',0,0)),(('base',.3,0),('cube',0,0)),(('base',1,0),('cube',0,0))]:
 for w in [.25,.5,.75]: D[(key1,key2,w)]=[(1-w)*a+w*b for a,b in zip(D[key1],D[key2])]
R=[]
for tag,pp in D.items():
 e=np.array([abs(y[j]-p).mean() for (_,_,j),p in zip(folds,pp)]); R.append((e.max(),e.mean(),e[:9].mean(),e[0],str(tag),e.round(5)))
for r in sorted(R)[:16]:print(r)
for r in R:
 if r[4] in [str(('base',0,0)),str(('cube',0,0))]: print('ref',r)
