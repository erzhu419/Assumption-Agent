exec(open('/tmp/rheo_probe.py').read().split("print('train RMSE'")[0])
from itertools import combinations
P,H,D=c
n=len(H)
X=m._features(dg,dg)[-n:]; R=np.c_[np.ones(n),X]@w[2:]
B=(-np.expm1(-t[:,None]/tau)); B=np.vstack([B,1/tau]); F=(H[:,None]*w[:2])@B.T
Y=yd-F
lab=dg['lab_id']; grids={'direct':R[:,None], 'intercept':np.c_[R,np.ones(n)], 'interaction':np.c_[R,H*R], 'history':np.c_[R,P], 'free_gain':np.c_[R,H]}
for name,Z in grids.items():
  G=np.einsum('ij,kb->ikjb',Z,B)
  def fit(which):
    return np.linalg.lstsq(G[which].reshape(-1,2*Z.shape[1]),Y[which].ravel(),rcond=None)[0]
  coef=fit(np.ones(n,dtype=bool)); pred=np.empty_like(yd)
  for g in np.unique(lab):
    v=lab==g; pred[v]=F[v]+G[v].reshape(sum(v),len(t)+1,-1)@fit(~v)
  e=pred-yd
  print(name,'coef',coef.round(5),'LOLO strain/rate',[(round(np.sqrt(np.mean(e[lab==g,:-1]**2)),5),round(np.sqrt(np.mean(e[lab==g,-1]**2)),5)) for g in np.unique(lab)],'worst horizon',np.sqrt(np.mean(e[:,:-1]**2,0)).max())
print('archived deployment:',m.solve_batch.__code__.co_names)
for name,z in [('joint train/test',b[2]@w[2:]),('test-only',np.c_[np.ones(len(b[1])),m._features(te,te)[len(b[1]):]]@w[2:]),('diagnostic-only',R)]:
 print(name,'per-lab drift mean/std',[(g,round(z[bb['lab_id']==g].mean(),3),round(z[bb['lab_id']==g].std(),3)) for bb in [dg if name=='diagnostic-only' else te] for g in np.unique(bb['lab_id'])]);
print('train counts',len(y),'strain noise MAD',np.median(abs((m._design(a,dt,t,tau)@w-y)[:,:-1])),'rate MAD',np.median(abs((m._design(a,dt,t,tau)@w-y)[:,-1])))
