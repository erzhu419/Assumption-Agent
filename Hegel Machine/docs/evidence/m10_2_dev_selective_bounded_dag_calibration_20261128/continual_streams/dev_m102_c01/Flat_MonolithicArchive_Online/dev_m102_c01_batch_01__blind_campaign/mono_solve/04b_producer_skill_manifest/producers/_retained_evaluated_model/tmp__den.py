exec(open('/tmp/probe.py').read().split('rows=[]')[0])
from scipy.optimize import least_squares
from scipy.linalg import solve
from itertools import combinations
u=c['u']; v=c['v1']; rrs=[]
for nm,fn in [('root',lambda z:z/np.sqrt(1+z*z)),('soft',lambda z:z/(1+abs(z))),('rat',lambda z:z/(1+z*z)),('erf',__import__('scipy').special.erf),('tanhlin',lambda z:np.tanh(z)),('atanlin',np.arctan)]:
 for extra in (0,1):
  def feats(z,k,fn=fn,extra=extra):return np.c_[np.ones(len(z)),z,fn(k*z)] if extra else np.c_[np.ones(len(z)),fn(k*z)]
  def fit(k):
   F=feats(u,k); W=np.linalg.lstsq(F,v,rcond=None)[0];return np.std(v-F@W),W
  k=minimize_scalar(lambda k:fit(k)[0],bounds=(.03,6),method='bounded').x; er,W=fit(k); print('C1',nm,extra,k,er); rrs.append((er,lambda z,k=k,W=W,feats=feats:feats(z,k)@W))
cb=min(rrs,key=lambda r:r[0])[1]; cb2=maps['v2']; bi=maps['Bi']; bmap=maps['B']; al=lambda z:(z-off)/sl
for nm,z in [('BiA',al(bi(x))),('BA',al(bmap(x))),('A-Bi',Q['A-Bi']),('A-B',Q['A-B']),('A-',Q['A-']),('x',x),('A+Bi',Q['A+Bi'])]:
 Q[nm]=z; Q[nm+'c']=cb(z); Q[nm+'d']=cb2(z)
print('Bmatrix',B,'C1 fit',min(rrs,key=lambda r:r[0])[0])
sp=[np.arange(72)]+[np.random.default_rng(i+254).permutation(72) for i in range(5)]
rows=[]
for key in ['A-','A-Bi','A+Bi']:
 z=Q[key]; P=p(z,2); L=P[:,:5]; PQ=P[:,5:]; T3=p(z,3)[:,15:]; ij=np.array(list(combinations(range(4),2))); H=np.c_[z,z[:,ij[:,0]]+z[:,ij[:,1]],z[:,ij[:,0]]-z[:,ij[:,1]]]; P2=H**2
 for j in (0,1):
  yy=Y[:,j]; ys=yy.std(); yy=yy/ys
  for typ in ('cubic',):
   ee=[]
   for ix in sp:
    a,b=ix[:52],ix[52:]; pp=P; ll=L; dims=5; penalty=.018
    if typ=='cubic': pp=np.c_[P,T3]; penalty=.06
    if typ in ('axis','diag','full'): dims={'axis':16,'diag':4,'full':16}[typ]; penalty=.055
    if typ=='free':dims=14; penalty=.09
    def den(h):
     if typ in ('rank1','cubic'):return 1+(ll@h)**2
     if typ=='diag':return 1+(z*z)@(h*h)
     if typ=='axis':return 1+P2@(h*h)
     if typ=='full':return 1+np.sum((z@h.reshape(4,4))**2,axis=1)
     return np.exp(np.clip(P[:,1:]@h,-3,3))
    if typ=='axis':dims=H.shape[1]
    def fit(h):
     F=pp/den(h)[:,None]; ff=F[a]; reg=np.r_[np.full(P.shape[1],.0001),np.full(pp.shape[1]-P.shape[1],.20)]; W=solve(ff.T@ff+np.diag(reg),ff.T@yy[a],assume_a='pos');return np.r_[F[a]@W-yy[a],penalty*h,np.sqrt(reg)*W],F@W
    k=None
    for sd in range(3 if typ in ('rank1','cubic') else 1):
     h=np.random.default_rng(100+sd).normal(0,.3,dims); rr=least_squares(lambda h:fit(h)[0],h,max_nfev=85)
     if k is None or rr.cost<k.cost:k=rr
    e=abs(fit(k.x)[1]-yy)*ys;ee.append([e[a].mean(),e[b].mean()])
    if len(ee)==1: print("DEN",j,key,k.x,"NUM",np.round(np.linalg.lstsq((pp/den(k.x)[:,None])[a],yy[a],rcond=None)[0],3))
   ee=np.array(ee);rows.append((j,key,typ,ee[0],ee.mean(0),ee.max(0)))
for j in (0,1):
 for r in sorted([r for r in rows if r[0]==j],key=lambda r:r[4][1])[:12]:print(r)
