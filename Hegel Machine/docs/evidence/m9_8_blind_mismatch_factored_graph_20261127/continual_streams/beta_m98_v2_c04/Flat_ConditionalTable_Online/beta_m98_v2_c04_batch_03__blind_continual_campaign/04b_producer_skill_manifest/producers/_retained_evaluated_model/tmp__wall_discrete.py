exec(open('/tmp/wall_fit.py').read().split('for train,test')[0])
from itertools import product
names=['open','sealed','periodic','periodic-centered','dirichlet','reflect','outflow']
def make(k,v,L,r,kind):
 h=L/(len(k)-1); A=orig(k,v,L,r if kind=='open' else 0).tolil()
 if kind.startswith('periodic'):
  for j,i,l in [(0,-1,1),(-1,-2,0)]:
   A[j,i]=-(k[j]+k[i])/(2*h*h)-(max(v[j],0) if kind=='periodic' else v[j]/2)/h
   A[j,l]=-(k[j]+k[l])/(2*h*h)-(max(-v[j],0) if kind=='periodic' else -v[j]/2)/h
   A[j,j]=-A[j,i]-A[j,l]
 elif kind=='dirichlet':
  for j,l in [(0,1),(-1,-2)]: A[j,l]=0; A[j,j]=1e6
 elif kind=='reflect':
  for j,l in [(0,1),(-1,-2)]: A[j,l]*=2; A[j,j]*=2
 elif kind=='outflow':
  for j,l in [(0,1),(-1,-2)]: A[j,j]+=abs(v[j])/h
 return A.tocsc()
rows=[]
for a,b in product(names,repeat=2):
 m._axis=lambda k,v,L,r:make(k,v,L,r,a if k is s['kappa_x'] else b)
 F=m._forward(s)[0]@W; E=F-Y
 rows.append((np.linalg.norm(E)/np.linalg.norm(Y),a,b,np.round(np.linalg.norm(E,axis=0)/np.linalg.norm(Y,axis=0),6)))
for row in sorted(rows,key=lambda x:x[0])[:12]: print(row)
m._axis=lambda k,v,L,r:make(k,v,L,r,'sealed'); F=m._forward(s)[0]@W
for j in range(3):
 q=slice(j*32,(j+1)*32); print('shift',j,'mean obs/pred',Y[q].mean(0),F[q].mean(0),'corr',np.corrcoef(Y[q].T,F[q].T)[:2,2:]); print(np.c_[s['sensor_ij'],Y[q],F[q]][:12].round(5))
print('velocity',s['vel_x'].round(4),s['vel_y'].round(4))
