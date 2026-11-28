exec(open('/tmp/probe02.py').read().split('for name,Z')[0])
from scipy.linalg import svd
from itertools import product

def shapes(u):
 names=['1']; fs=[np.ones(len(u))]
 def add(n,f): names.append(n);fs.append(f)
 for i,w in enumerate(u.T):
  for j in range(1,5):add(f'x{i}^{j}',w**j)
 for i,j in combinations(range(4),2):
  v,w=u[:,i],u[:,j];add(f'x{i}x{j}',v*w)
 for i in range(4):
  j=(i+1)%4; v,w=u[:,i],u[:,j]
  for name,f in [('s',np.sin),('c',np.cos),('t',np.tanh)]:
   for k in [0.5,0.8,1,1.2,1.5,1.8,2,2.4,2.8,3]:
    add(f'{name}{k}x{i}',f(k*v));add(f'{name}{k}x{i}x{j}',f(k*v*w));add(f'{name}{k}(x{i}+x{j})',f(k*(v+w)));add(f'{name}{k}(x{i}-x{j})',f(k*(v-w)))
 return np.column_stack(fs),np.array(names)
C=s['panel_c']; u=C['u']; F,names=shapes(u)
base=F[:,[0,1,5,9,13]]; N=F.shape[1];
for key in ('v1','v2'):
 for j,yc in enumerate(C[key].T):
  print(key,j)
  keep=[]; G=base; res=yc-G@np.linalg.lstsq(G,yc,rcond=None)[0]
  for k in range(4):
   Q=np.linalg.qr(G)[0]; p=F-Q@(Q.T@F); norm=np.linalg.norm(p,axis=0); sc=np.abs(p.T@res)/np.maximum(norm,1e-9); win=np.argsort(sc)[-6:][::-1]; i=win[0]; G=np.c_[G,F[:,i]]; res=yc-G@np.linalg.lstsq(G,yc,rcond=None)[0];print(' ',names[win].tolist(), np.round(sc[win],3),round(np.linalg.norm(res),5))
for name,zz in [('z',z),('Bz',z@B[1:]+B[0])]:
 print(name)
 for j in range(3):
  f,keys=poly(zz,3); xx=np.c_[np.ones(len(f)),f]; W=np.linalg.lstsq(xx[a],Y[a,j],rcond=None)[0]; top=np.argsort(abs(W[1:]))[-10:][::-1]; print(j,[(keys[i],round(W[1+i],3)) for i in top])
print('y given measured modes and quadratic',[(m,np.mean(abs((ff[q]@np.linalg.lstsq(ff[a],y[a],rcond=None)[0])-y[q]))) for m,ff in [('modes',np.c_[np.ones(len(z)),Y[:,1:]]),('z+modes',np.c_[np.ones(len(z)),z,Y[:,1:]]),('quad+modes',np.c_[np.ones(len(z)),poly(z,2)[0],Y[:,1:]]),('quad+quadmode',np.c_[np.ones(len(z)),poly(z,2)[0],Y[:,1:],Y[:,1:]**2,Y[:,1]*Y[:,2]])]])
