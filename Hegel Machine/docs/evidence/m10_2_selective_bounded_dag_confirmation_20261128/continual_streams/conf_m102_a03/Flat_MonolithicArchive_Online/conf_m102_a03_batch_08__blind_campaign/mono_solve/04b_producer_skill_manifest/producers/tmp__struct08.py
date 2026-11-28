exec(open('/tmp/explore08.py').read().split('for name in')[0])
from itertools import product
T=s['train']['traj']; h=m._warp(z,p); w=aff(h)@b; print('t/tel',s['eval_times'],s['telemetry_vec']); k=52
X=np.c_[aff(z),h,(z[:,:,None]*h[:,None,:]).reshape(-1,16),pol(h,2)[:,4:]]
C=np.linalg.lstsq(X[:k],T[:k],rcond=None)[0]; print('trajectory coefs temporal SVD'); u,sv,vh=np.linalg.svd(C,full_matrices=False); print(sv); print(np.c_[np.arange(len(C)),u[:,:3]].round(3)); print('y vs T',probe(np.c_[X,T[:,-1]]),probe(np.c_[X,T]),'corr',np.corrcoef(y,T[:,-1])[0,1]); print('corr y residual T residual',np.corrcoef(np.c_[y,T]-X@np.linalg.lstsq(X[:k],np.c_[y,T][:k],rcond=None)[0],rowvar=False)[0].round(2));
# Profile a low-rank bilinear interaction, including a rank-constrained cross term in either latent or operator coordinates.
def trial(q,which,r,extra,split):
 i,j=split; hh=m._warp(z,q); ww=aff(hh)@b if which=='b' else hh
 a=np.c_[aff(z),hh]; Q=pol(hh,2)[:,4:]; D=np.c_[a,Q] if extra else a
 D=D/np.sqrt((D[i]**2).mean(0)).clip(1e-7); U=z/z[i].std(0); V=ww/ww[i].std(0); dd=D[i]; pin=np.linalg.pinv(dd); target=y[i]; ui,vi=U[i],V[i]
 def fun(v,ret=False):
  l,rr=v.reshape(2,4,r); yz=((U@l)*(V@rr)).sum(1); c=pin@(target-yz[i]); e=dd@c+yz[i]-target
  if ret:return D@c+yz,l@rr.T
  penalty=.02 if extra else .009
  return np.r_[e,penalty*v]
 mat=np.linalg.lstsq(np.c_[dd,(ui[:,:,None]*vi[:,None,:]).reshape(-1,16)],target,rcond=None)[0][-16:].reshape(4,4); l,sv,rr=np.linalg.svd(mat); init=np.array([l[:,:r]*np.sqrt(sv[:r]),rr.T[:,:r]*np.sqrt(sv[:r])]).ravel(); found=least_squares(fun,init,max_nfev=150); yy,mat=fun(found.x,True); return np.abs(yy[i]-y[i]).mean(),np.abs(yy[j]-y[j]).mean(),mat
rng=np.random.default_rng(42); perm=rng.permutation(72); splits=[(np.arange(52),np.arange(52,72)),(perm[:52],perm[52:]),(np.arange(20,72),np.arange(20))]; out=[]
for q,r,extra in product((p,[.42,1.25],[.38,1.32]),(1,2,3),(False,True)):
 rr=[trial(q,'h',r,extra,sp) for sp in splits]; out.append(([tuple(round(x,5) for x in a[:2]) for a in rr],q,r,extra))
print('low rank',*sorted(out,key=lambda a:max(v[1] for v in a[0])),sep='\n')
