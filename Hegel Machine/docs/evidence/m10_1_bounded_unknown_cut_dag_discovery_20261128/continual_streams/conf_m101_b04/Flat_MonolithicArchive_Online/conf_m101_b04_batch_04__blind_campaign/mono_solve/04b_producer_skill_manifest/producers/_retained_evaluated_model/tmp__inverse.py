from search import *
from scipy.optimize import root, minimize
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor,cho_solve
u,v=s['panel_a'].values(); aa=fit(poly(u[keep],3),v[keep]);
def inv(f,q):return np.array([root(lambda z:f(z[None])[0]-w,w).x for w in q])
zi=inv(A,x); print('A inverse error',np.max(abs(A(zi)-x)), 'ranges',np.max(abs(zi),0))
bu,bv=s['panel_b'].values(); print('B residual',np.std(bv-B(bu),0), 'sv',np.linalg.svd(bc[1:])[1]); print('log y linear in trajectory',np.mean(abs(poly(tr[:,:2],1)@fit(poly(tr[:,:2],1),y)-y))); print('svd',np.linalg.svd(np.c_[tr,y],compute_uv=False)); print('raw rows', np.round(np.c_[x,y.tolist()+[0]*36,np.r_[tr[:,0],np.zeros(36)]][:12],3))
bi=(x-bc[0])@np.linalg.inv(bc[1:]); coords={'x':x,'i':zi,'ib':B(zi),'bi':inv(A,B(x)),'j':bi,'ij':inv(A,bi),'ji':(zi-bc[0])@np.linalg.inv(bc[1:]),'a':A(x),'b':B(x),'ba':B(A(x))}
for k,z in list(coords.items()):
 coords[k+'c']=C(z);coords[k+'d']=D(z)
rows=[]
for k,z in coords.items():
 for d in (1,2,3):
  p=poly(z,d);e=maes(p,None,0);rows.append((e.max(),k,d,'-',e[0],e.mean(0)))
  for base in ('x','i'):
   zz=np.c_[coords[base]/coords[base][:52].std(0),z/z[:52].std(0)];dist=cdist(zz,zz,'sqeuclidean');
   for g in (.008,.03,.09):
    for reg in (.0005,.006,.03):
     e=maes(p,np.exp(-g*dist),reg);rows.append((e.max(),k,d,(base,g,reg),e[0],e.mean(0)))
print('best inverse/poly',*sorted(rows,key=lambda r:r[0])[:12],sep='\n')
print('best public',*sorted(rows,key=lambda r:max(r[4]))[:7],sep='\n')
np.savez('/tmp/coords.npz',**coords)
