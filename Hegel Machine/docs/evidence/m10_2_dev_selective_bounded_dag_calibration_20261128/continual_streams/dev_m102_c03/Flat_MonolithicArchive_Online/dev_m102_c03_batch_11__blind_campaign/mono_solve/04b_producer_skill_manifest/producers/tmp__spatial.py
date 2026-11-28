exec(open('/tmp/identify.py').read().split("p=s['panel_a']")[0]); from itertools import combinations_with_replacement
from scipy.spatial.distance import cdist
soft=lambda x:x/(1+abs(x))
def fitpanel(p,key,f,start,zz):
 u=p['u'];v=p[key]
 def e(r):
  a=f(u,r);return (a@ls(a,v)-v).ravel()
 r=least_squares(e,start,bounds=(.08,5),gtol=1e-12).x
 print(key, r,np.std(e(r)),flush=True); return f(zz,r)@ls(f(u,r),v)
t=s['train']; x=np.vstack([t['x'],s['test']['x']]); p=s['panel_a'];u=p['u'];v=p['v'];a=aff(u);keep=np.linalg.norm(v-a@ls(a,v),axis=1)<3
w=lambda z:np.roll(z,2,axis=-1)
def cal(z,r):
 return np.stack([np.ones_like(z),z,w(z),z*w(z),z*z,w(z)**2,z**3,np.sin(r*w(z))],axis=-1)
def bc(r):
 a=cal(u[keep],r); return np.array([ls(a[:,j],v[keep,j]) for j in range(4)])
def fw(z,r,b):return np.einsum('...ij,ij->...i',cal(z,r),b)
r=minimize_scalar(lambda r:np.std(fw(u[keep],r,bc(r))-v[keep]),bounds=(1.2,3.5),method='bounded').x;b=bc(r);print('cal',r,np.std(fw(u[keep],r,b)-v[keep]),b.round(3)); z=np.array([least_squares(lambda q:fw(q,r,b)-row,q,gtol=1e-12).x for row,q in zip(x,(x-ls(a[keep],v[keep])[0])@np.linalg.inv(ls(a[keep],v[keep])[1:]))]);print('ranges',np.max(abs(z),axis=0),np.max(abs(u[keep]),axis=0));
c=fitpanel(s['panel_c'],'v1',lambda u,r:np.c_[aff(u),soft(r[0]*u)], [.68],z)
c2=fitpanel(s['panel_c'],'v1',lambda u,r:np.c_[aff(u),u**3,np.sin(r[0]*u),soft(r[1]*u)], [3.17,.5],z)
d=fitpanel(s['panel_c'],'v2',lambda u,r:np.c_[np.ones(len(u)),u*u,np.cos(r[0]*u)], [2.57],z)
d2=fitpanel(s['panel_c'],'v2',lambda u,r:np.c_[np.ones(len(u)),u*u,np.exp(-(r[0]*u)**2)], [.88],z)
p=s['panel_b']; ub=p['u']; vb=p['v']; print('B linear RMS',loss(aff(ub),vb));
cyc=lambda u:u*np.roll(u,-1,axis=1)
for name,f in [('poly2',lambda u:np.c_[aff(u),*[np.prod(u[:,k],axis=1) for k in combinations_with_replacement(range(4),2)]]),('cycle',lambda u:np.c_[aff(u),cyc(u)]),('soft',lambda u:np.c_[aff(u),soft(.8*u)]),('cos',lambda u:np.c_[aff(u),np.cos(2*u)]),('sin',lambda u:np.c_[aff(u),np.sin(2*u)])]:
 print('B',name,loss(f(ub),vb));
B=aff(z)@ls(aff(ub),vb)
def cross(a,b):return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
def poly(u,k):return np.column_stack([np.prod(u[:,j],axis=1) for j in combinations_with_replacement(range(u.shape[1]),k)])
U,sv,V=np.linalg.svd(t['traj'][:52],full_matrices=False);Q=V[:2].T;Y=np.c_[t['y'],t['traj']@Q];n=52
options=[]
def report(name,F=None,K=None):
 if K is None:
  F=F.copy(); F-=F[:n].mean(0); F/=np.maximum(F[:n].std(0),1e-10);K=F@F.T/F.shape[1]
 K=K+1; k=K[:n,:n]; ev,vec=np.linalg.eigh(k);l=[]
 for a in [1e-9,1e-7,1e-5,1e-4,.001,.005,.02,.1]:
  inv=(vec/(np.maximum(ev,0)+a))@vec.T; coef=inv@Y[:n]; loo=np.mean((coef/inv.diagonal()[:,None])**2,axis=0);p=K[n:72,:n]@coef;
  e=[np.mean(abs(p[:,0]-Y[n:,0])),np.mean(abs(p[:,1:]@Q.T-t['traj'][n:]))];l.append((loo,p,e,a))
 yy=min(l,key=lambda i:i[0][0]); ss=min(l,key=lambda i:sum(i[0][1:])); options.append((name,yy[2][0],ss[2][1],np.sqrt(yy[0][0]),np.sqrt(sum(ss[0][1:])/8),yy[-1],ss[-1]));
for name,a in [('x',x),('z',z),('B',B)]:
 for k in [2,3,4]:report((name,k),np.c_[*[poly(a,l) for l in range(1,k+1)]])
 for g in [.1,.25,.5,1,2]:report((name,'rbf',g),K=np.exp(-g*cdist(a,a,'sqeuclidean')))
for i,b1 in enumerate([c,c2,B,np.tanh(z),soft(.7*z)]):
 for j,b2 in enumerate([d,d2,np.cos(2.57*z),np.exp(-.877**2*z*z),B]):
  for m,F in enumerate([np.c_[z,b1,b2,cross(b1,b2)],np.c_[z,b1,b2,cross(b1,b2),poly(z,2)],np.c_[b1,b2,cross(b1,b2),cross(z,b1),cross(z,b2),z],np.c_[z,b1,b2,poly(np.c_[b1,b2],2)]]):report((i,j,m),F)
for i in sorted(options,key=lambda r:max(r[1:3]))[:20]:print('joint',i)
for i in sorted(options,key=lambda r:r[3])[:5]:print('y loo',i)
for i in sorted(options,key=lambda r:r[4])[:5]:print('s loo',i)
