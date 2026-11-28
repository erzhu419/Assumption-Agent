import pickle,numpy as np
from itertools import combinations_with_replacement as cwr
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x,y,T=(tr[k] for k in ('x','y','traj'))
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
def keys(d):return [k for n in range(d+1) for k in cwr(range(4),n)]
def poly(x,d):return np.column_stack([np.prod(x[:,k],axis=1) for k in keys(d)])
a=s['panel_a']; A=poly(a['u'],3); rng=np.random.default_rng(3); best=None
for _ in range(800):
 j=rng.choice(len(A),35,replace=False); e=np.linalg.norm(a['v']-A@ls(A[j],a['v'][j]),axis=1); score=np.median(e)
 if best is None or score<best[0]:best=score,e
ix=np.argsort(best[1])[:45]; ba=ls(A[ix],a['v'][ix]); print('panel A outliers',np.sort(np.argsort(best[1])[45:])); print('A coefficients:')
for k,b in zip(keys(3),ba):
 if max(abs(b))>0.002: print(k,b.round(5))
for j in range(4):
 b=abs(ba[:,j])>.002; ba[~b,j]=0; ba[b,j]=ls(A[ix][:,b],a['v'][ix,j]); print('A sparse inlier RMSE',j,np.std(A[ix]@ba[:,j]-a['v'][ix,j]))
def inverse(x):
 u=(x-ba[0])@np.linalg.inv(ba[1:5]); k=keys(3)
 for _ in range(20):
  e=poly(u,3)@ba-x
  J=np.stack([np.column_stack([sum(kj==j for kj in q)*np.prod(u[:,list(q[:q.index(j)]+q[q.index(j)+1:])],axis=1) if j in q else np.zeros(len(u)) for q in k])@ba for j in range(4)],axis=2)
  delta=np.linalg.solve(J,e[...,None])[...,0]; u-=delta/np.maximum(1,np.linalg.norm(delta,axis=1,keepdims=True))
 print('inverse residual',abs(poly(u,3)@ba-x).max());return u
u=inverse(x); ute=inverse(s['test']['x']); b=s['panel_b']; bb=ls(poly(b['u'],1),b['v']); z=poly(u,1)@bb; zt=poly(ute,1)@bb; print('ranges u,z',u.min(0),u.max(0),z.min(0),z.max(0)); print('test u,z',ute.min(0),ute.max(0),zt.min(0),zt.max(0))
if __name__=='__main__':
 Y=np.c_[y,T]; splits=[np.arange(72)]+[np.random.default_rng(i).permutation(72) for i in range(1,4)]
 for name,w in [('raw',x),('cal',u),('mix',z)]:
  for d in (1,2,3):
   h=poly(w,d); rms=np.sqrt(np.mean((h@ls(h,Y)-Y)**2,axis=0)); cv=[np.mean(abs(h[i[56:]]@ls(h[i[:56]],Y[i[:56]])-Y[i[56:]]),axis=0) for i in splits]; cv=np.array(cv); print(name,d,'train RMS y,T',rms[0],rms[1:].mean(),'cv y,T',np.c_[cv[:,0],cv[:,1:].mean(1)].round(5).tolist())
 for name,w in [('cal',u),('mix',z)]:
  print(name,'degree2 y, t1 and last coefficients'); print(np.c_[ls(poly(w,2),Y)[:,[0,1,-1]]].round(4))
