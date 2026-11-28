exec(open('/tmp/search10.py').read().split('A=panel')[0])
from scipy.optimize import least_squares,root
p=S['panel_a']; u,v=p['u'],p['v']; m=np.linalg.norm(v-u,axis=1)<4*np.median(np.linalg.norm(v-u,axis=1)); u,v=u[m],v[m];
def ar(x,k,j):
 o=(j+2)%4; return a(x[:,j],x[:,o],np.sin(k*x[:,j]),np.sin(k*x[:,o]))
def ra(k,ret=False):
 fs=[ar(u,k[0],j) for j in range(4)]; bs=[ls(f,v[:,j]) for j,f in enumerate(fs)]; return bs if ret else np.column_stack([f@b for f,b in zip(fs,bs)]).ravel()-v.ravel()
k=least_squares(ra,[2.4],bounds=(1.5,3.5)).x; bs=ra(k,True); A=lambda x:np.column_stack([ar(x,k[0],j)@b for j,b in enumerate(bs)]); Z=np.array([root(lambda p:A(p[None])[0]-x,x).x for x in X]); print('cal',k,np.sqrt(np.mean(ra(k)**2)),np.max(abs(A(Z)-X)))
p=S['panel_b']; u,v=p['u'],p['v'];
for name,g in forms.items():
 if name=='sq': continue
 def prof(p,ret=False):
  f=a(u,g(u,p[0])); b=ls(f,v); return b if ret else (f@b-v).ravel()
 r=min((least_squares(prof,[x],bounds=(.02,5)) for x in [.25,1,2.8,4]),key=lambda r:r.cost); k=r.x[0]; b=prof(r.x,True); print('B',name,round(k,4),round(np.sqrt(np.mean(r.fun**2)),5));
 if name in ('sin','rat'): print(np.round(b,4))
C=panel(S['panel_c'],'v1',forms)[0][-1]; D=panel(S['panel_c'],'v2',['sq'])[0][-1]; opts=panel(S['panel_b'],'v',forms); B=opts[0][-1]; R=next(p[-1] for p in opts if p[1]=='rat' and not p[2]); Q=B(Z)
T=S['eval_times']; basis=lambda p:-np.expm1(-T[:,None]*p); pair=lambda z:np.column_stack([z[:,i]*z[:,j] for i,j in combinations_with_replacement(range(4),2)])
def pred(f,ii,jj,l):
 w=f[ii]; c=ls(w,Y[ii]) if not l else np.linalg.solve(w.T@w+l*np.diag([0]+[1]*(w.shape[1]-1)),w.T@Y[ii]); o=f[jj]@c
 def err(p):
  f=basis(p); return (f@ls(f,Y[ii,1:].T)-Y[ii,1:].T).ravel()
 ft=basis(least_squares(err,[.4,1.8],bounds=(.01,10)).x); o[:,1:]=o[:,1:]@ft@np.linalg.pinv(ft); return abs(o-Y[jj])
fs={};
for k0,x in [('z',Z),('x',X),('b',Q),('r',R(Z)),('bx',B(X))]:
 fs[k0]=a(x); fs[k0+'2']=a(x,pair(x));
for k,x in [('z',Z),('x',X),('c',C(Z)),('d',D(Z)),('cc',C(C(Z))),('dc',C(D(Z))),('dcd',D(C(D(Z))))]:
 for kb,b in [('b',Q),('r',R(Z)),('bx',B(X))]:
  fs[kb+'+'+k]=a(b,x); fs[kb+'2+'+k]=a(b,pair(b),x); fs[kb+'+'+k+'2']=a(b,x,pair(x)); fs[kb+'*'+k]=a(b,x,pair(b),(b[:,:,None]*x[:,None,:]).reshape(-1,16))
idx=np.arange(72); splits=[(idx[:52],idx[52:])]; rng=np.random.default_rng(21)
for _ in range(5):
 i=rng.permutation(idx); splits.append((i[:52],i[52:]))
res=[]
for k,f in fs.items():
 for l in [0,1e-4,.002,.03]:
  if l==0 and np.linalg.cond(f)>1e7: continue
  E=[pred(f,i,j,l) for i,j in splits]; s=np.array([[e[:,0].mean(),e[:,1:].mean(),e[:,-1].mean(),e[:,1:].mean(0).max()] for e in E]); res.append((k,l,s[0],s.mean(0),s.max(0)))
for j in range(3):
 print('leader',j); 
 for x in sorted(res,key=lambda x:x[4][j])[:8]: print(x[:2],*[np.round(z,4) for z in x[2:]])
