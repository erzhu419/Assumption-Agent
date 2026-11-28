exec(open('/tmp/readout_probe.py').read().split('np.set_printoptions')[0])
from scipy.linalg import qr
from scipy.optimize import minimize_scalar
n=len(z); T=Y.copy(); T[:,1:]/=np.sqrt(8); A=poly(z,1); D=poly(z,2)
B=s['panel_b']; k,g=m._fit_b(B['u'],B['v']); H=m._b_design(u,k)[:,5:]; Z=H@g[5:]; C=s['panel_c']; f=lambda q:np.c_[poly(q,1),np.tanh(2.1*q)]; V=f(u)@np.linalg.lstsq(f(C['u']),C['v1'],rcond=None)[0]; W=poly(u,2)@np.linalg.lstsq(poly(C['u'],2),C['v2'],rcond=None)[0]
sets={'u':u,'z':z,'h':H,'Z':Z,'v':V,'w':W,'x':x}; base=np.c_[A]; nm=['1']+['z'+str(i) for i in range(4)]; cols=[]; names=[]
def add(name,q):
 for j,c in enumerate(np.asarray(q).reshape(n,-1).T):cols.append(c);names.append(f'{name}:{j}')
for name,q in sets.items():
 for d in [1,2,3,4]:
  if d<=2 or name in 'uzhZvw':
   add(name+str(d),poly(q,d)[:,poly(q,d-1).shape[1]:])
 for k in [.6,1.2,2.4]:
  add(name+'sin'+str(k),np.sin(k*q));add(name+'tanh'+str(k),np.tanh(k*q));add(name+'rat'+str(k),q/(1+k*q*q))
 for i,j in itertools.combinations(range(4),2):
  for sign in [-1,1]:
   h=q[:,i]+sign*q[:,j]
   for k in [.6,1.2,2.4]:
    add(f'{name}{i}{j}{sign}cos{k}',np.cos(k*h));add(f'{name}{i}{j}{sign}rat{k}',1/(1+k*h*h));add(f'{name}{i}{j}{sign}abs',abs(h))
for (n1,q),(n2,r) in itertools.combinations(sets.items(),2):
 add(n1+n2,q[:,:,None]*r[:,None,:]); add(n1+n2+'sq1',q[:,:,None]**2*r[:,None,:]);add(n1+n2+'sq2',q[:,:,None]*r[:,None,:]**2)
F=np.column_stack(cols)
def omp(a,b,base,t,maxterms):
 p=[];out=[]; q=np.linalg.qr(base[a])[0];G=F[a]-q@(q.T@F[a]);R=t[a]-q@(q.T@t[a]); avail=np.ones(F.shape[1],bool)
 norms=np.linalg.norm(F[a],axis=0)
 for j in range(maxterms):
  scales=np.linalg.norm(G,axis=0); corr=np.abs(G.T@R)/np.maximum(scales,1e-9); corr[~avail|(scales<1e-3*norms)]=-1
  r=np.argmax(corr);p.append(r);v=G[:,r]/scales[r];G-=np.outer(v,v@G);R-=v*(v@R);avail[r]=False
  U=np.c_[base,F[:,p]];ec=np.linalg.lstsq(U[a],t[a],rcond=1e-8)[0];out.append(np.mean(abs(U[b]@ec-t[b])))
 return np.array(out),[names[i] for i in p]
for desc,base0 in [('a',A),('d',D),('mix',np.c_[D,u,(z[:,:,None]*u[:,None,:]).reshape(n,16)]),('c',np.c_[D,poly(V,2)[:,1:],poly(W,2)[:,1:]])]:
 for target in [0,-1]:
  v=[omp(a,b,base0,Y[:,target],12) for a,b in fs]; E=np.array([p[0] for p in v]);print(desc,target,'official',E[0].round(4),'worst',E.max(0).round(4),v[0][1]);
for d in [A,D,np.c_[D,u,(z[:,:,None]*u[:,None,:]).reshape(n,16)],np.c_[D,poly(V,2)[:,1:],poly(W,2)[:,1:]]]:
 print('fit',np.std(Y-d@np.linalg.lstsq(d,Y,rcond=None)[0],axis=0).round(4))
