from graph_probe import *
from sparse_graph import pred
from scipy.spatial.distance import cdist
import sys
sys.path.insert(0,'/holdout_task')
from batch_solver import _smooth
rng=np.random.default_rng(1124)
ss=splits.copy()
for i in range(8):
 p=rng.permutation(72); ss.append((p[:56],p[56:]))
configs=[]
for warp in ('raw','quad+','quad-','tanh','sinh','joint','hessian'):
 for ell in (.65,.8,1.,1.2,1.45):
  for nug in (1e-5,1e-4,5e-4):
   for trend in ('linear','latent','quadratic','latentquadratic'):
    configs.append((warp,ell,nug,trend))
cache={}
for a,b in ss:
 xp=x/x[a].std(0); zz=z.copy();zz[b]=_smooth(x[a],z[a],x[b],1.7,.0001).ravel();zn=zz/z[a].std();
 coeff=np.linalg.lstsq(poly(xp[a],2),y[a],rcond=None)[0]; H=np.zeros((4,4));
 for (i,j),c in zip(combinations_with_replacement(range(4),2),coeff[5:]):H[i,j]=H[j,i]=c/(1+(i!=j))
 val,vec=np.linalg.eigh(H)
 frames2={'raw':xp,'quad+':xp+.18*xp*xp,'quad-':xp-.18*xp*xp,'tanh':np.tanh(.52*xp)/.52,'sinh':np.sinh(.55*xp)/.55,'joint':np.c_[xp,.55*zn], 'hessian':(xp@vec)*np.sqrt(.6+.8*abs(val)/max(abs(val)))}
 D1=poly(xp,1); D2=poly(xp,2); ds={k:cdist(h,h,'sqeuclidean')/4 for k,h in frames2.items()}; D={'linear':D1,'latent':np.c_[D1,zn],'quadratic':D2,'latentquadratic':np.c_[D2,zn]}
 for w,l,n,t in configs:
  K=np.exp(-ds[w]/(2*l*l)); p,il,_=pred(K,D[t],y,a,b,n,True);cache.setdefault((w,l,n,t),[]).append((y[b]-p,il))
def score(v):
 m=np.array([abs(r).mean() for r in v]);return .35*m[0]+.3*m.mean()+.35*m.max(),np.round([m[0],m.mean(),m.max()],5)
for k in sorted(cache,key=lambda k:score([r for r,_ in cache[k]])[0])[:16]: print(k,score([r for r,_ in cache[k]])[1])
for n in (3,8,16):
 print('nested LOO',n,score([np.mean([cache[k][i][0] for k in sorted(cache,key=lambda k:cache[k][i][1])[:n]],axis=0) for i in range(len(ss))])[1])
keys=[('raw',1.,.0001,'latent'),('raw',1.,.0001,'linear')]; good=sorted(cache,key=lambda k:score([r for r,_ in cache[k]])[0]); keys+=good[:12]
out=[]
for i,k in enumerate(keys):
 for g in keys[i:]:
  for a in (.25,.5,.75):
   v=[(1-a)*r[0]+a*t[0] for r,t in zip(cache[k],cache[g])]; out.append((*score(v),k,g,a))
for _,sc,k,g,a in sorted(out,key=lambda r:r[0])[:12]:print('blend',sc,k,g,a)
print('by gate');
for _,sc,k,g,a in sorted(out,key=lambda r:r[1][0])[:8]:print(sc,k,g,a)
