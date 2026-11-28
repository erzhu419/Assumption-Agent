exec(open('/tmp/probe.py').read().split('rows=[]')[0])
from scipy.optimize import least_squares, minimize_scalar
from scipy.linalg import solve
from itertools import combinations
u=c['u']; v=c['v1']; best=[]
for name,fn in [('tanh',np.tanh),('sin',np.sin),('atan',np.arctan)]:
 for form in range(5):
  if form<3:
   def f(z,h):
    k=np.exp(h[:1] if form==0 else h[:4]); d=h[4:8] if form==2 else 0
    return np.c_[np.ones(len(z)),fn(z*k+d)]
   def ef(h):
    F=f(u,h);return (F@np.linalg.lstsq(F,v,rcond=None)[0]-v).ravel()
   h=least_squares(ef,np.r_[np.ones(4)*.3,np.zeros(4)],max_nfev=300).x; w=np.linalg.lstsq(f(u,h),v,rcond=None)[0]; cb=lambda z,f=f,h=h,w=w:f(z,h)@w; rm=np.std(ef(h))
  else:
   L=np.c_[np.ones(len(u)),u]; W=np.linalg.lstsq(L,v,rcond=None)[0]
   def f(z,h):
    ww=h[:20].reshape(5,4); a=np.exp(h[20:24]); b=h[24:28] if form==4 else 0;return b+a*fn(np.c_[np.ones(len(z)),z]@ww)
   h=least_squares(lambda h:(f(u,h)-v).ravel(),np.r_[W.ravel(),np.zeros(8)],max_nfev=300).x; cb=lambda z,f=f,h=h:f(z,h);rm=np.std(cb(u)-v)
  best.append((rm,name,form,h,cb))
for rm,nm,form,h,cb in sorted(best,key=lambda x:x[0]):print('panel1',rm,nm,form,h[:8]);
cb=min(best,key=lambda x:x[0])[-1]; c2=maps['v2']; bmap=maps['B']; bi=maps['Bi']; z=Q['A-'];
for bname,bf in [('B',bmap),('Bi',bi),('',lambda z:z)]:
 for cn,cf in [('C1',cb),('C2',c2)]:
  Q['new-'+bname+cn]=cf(bf(z)); Q['new-'+cn+bname]=bf(cf(z))
I,J=np.triu_indices(4); rng=np.random.default_rng(702); splits=[np.arange(72)]+[rng.permutation(72) for _ in range(6)]; results=[]
def score(F):
 sd=np.sqrt(np.mean(F*F,axis=0)); F=F/np.maximum(sd,1e-8); ee=[]
 for a,b in [(ix[:52],ix[52:]) for ix in splits]:
  ff=F[a]; w=solve(ff.T@ff+.003*np.eye(F.shape[1]),ff.T@Y[a],assume_a='pos'); ee.append(np.mean(abs(Y[b]-F[b]@w),axis=0))
 return np.asarray(ee)
for key,q in Q.items():
 L=p(q,1); P=p(q,2); D=q[:,I]*q[:,J]; pairs=np.array(list(combinations(range(4),2))); D1=q[:,pairs[:,0]]+q[:,pairs[:,1]]; D2=q[:,pairs[:,0]]-q[:,pairs[:,1]]; H=np.c_[q,D1,D2];
 def add(nm,F):
  e=score(F); results.append((key,nm,e[0],e.mean(0),e.max(0)))
 add('p2',P)
 for k in (.25,.7,1.25,2):
  for name,fn in [('tanh',np.tanh),('sin',np.sin),('atan',np.arctan),('rat',lambda x:x*x/(1+x*x)),('rt',lambda x:x/(1+x*x))]:
   G=fn(k*q); H1=fn(k*H); add((name,k,'products'),np.c_[P,G,G[:,I]*G[:,J]]); add((name,k,'sums'),np.c_[P,H1]); add((name,k,'prodop'),np.c_[P,fn(k*D)]);
 for key2 in ['A-','A-B','new-BC1','new-BC2']:
  if key==key2: continue
  r=Q[key2]; add((key2,'stack'),np.c_[P,p(r,2)[:,1:]]); add((key2,'cross'),np.c_[P,r,(q[:,:,None]*r[:,None,:]).reshape(len(q),-1)])
for j in [0,1]:
 print('OUT',j)
 for row in sorted(results,key=lambda r:r[3][j])[:13]:print(row)
