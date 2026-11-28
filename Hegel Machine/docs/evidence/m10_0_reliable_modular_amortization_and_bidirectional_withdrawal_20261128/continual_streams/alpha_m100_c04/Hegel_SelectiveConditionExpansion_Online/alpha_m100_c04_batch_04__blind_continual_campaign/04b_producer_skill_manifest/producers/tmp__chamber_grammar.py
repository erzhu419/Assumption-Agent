exec(open('/tmp/chamber_probe.py').read().split('for tt in (ta,')[0].replace("print('taus theta',ta,th);", "print('taus theta',ta,th);"))
from itertools import combinations
from collections import defaultdict

def combine(*parts):
 return {k:np.concatenate([p[k] for p in parts]) for k in ('lab_id','stratum','coupon_features_raw','ref_coupons')}
empty={k:v[:0] for k,v in combine(d).items()}
X={'train+diag':xd,'diag only':a._features(d,empty),'test+diag':a._features(te,d)[-len(h):], 'all':a._features(combine(tr,te),d)[-len(h):]}
lab=d['lab_id']; d0=len(h); raw=d['coupon_features_raw'];
for name,parts in [('train+test',(tr,te)),('train only',(tr,)),('test only',(te,))]:
 fitp=combine(*parts); xx=np.empty_like(raw)
 for l in np.unique(lab):
  panel=fitp['ref_coupons'][fitp['lab_id']==l]; xx[lab==l]=(raw[lab==l]-np.median(panel.mean(0)))/np.median(panel.std(0))
 xf=a._features(fitp,empty); _,ii,cc=np.unique(np.c_[fitp['lab_id'],fitp['stratum']],axis=0,return_inverse=True,return_counts=True); mu=np.average(xf,axis=0,weights=1/cc[ii]); X[name]=np.where(np.isfinite(xx),xx,mu)
f=F(ta); H=h[:,None]*(f[:,:2]@th[:2]); R=Y-H
print('aux summaries',[(k,np.round(C[k]['aux_chamber_telemetry'][:3],4)) for k in ('train','test')]); print('diagnostic missing rows',np.flatnonzero(np.isnan(raw).any(1)))
U,S,V=np.linalg.svd(R,full_matrices=False); print('residual singular values',S[:7]); print('leading load-subtracted mode',np.linalg.lstsq(f,V[0],rcond=None)[0]); latent=U[:,0]*S[0];
for nm,xx in X.items():
 v=np.c_[np.ones(d0),xx]@th[2:]; err=R-v[:,None]*np.linalg.lstsq(v[:,None],R,rcond=None)[0]; print(nm,'drift alignment rms',np.sqrt(np.mean(err**2)), 'lab means', [np.round(np.mean((latent/(v@latent/(v@v))-v)[lab==l]),4) for l in np.unique(lab)])
 for typ,cols in [('d',[v]),('d,d2',[v,v*v]),('d,h*d',[v,h*v]),('d,z1*d',[v,past.mean(1)*v]),('d,const,h',[v,np.ones(d0),h]),('d,features',[np.ones(d0),*xx.T])]:
  M=np.column_stack(cols); errs=[]
  for l in np.unique(lab):
   keep=lab!=l; b=np.linalg.lstsq(M[keep],R[keep],rcond=None)[0]; e=M[~keep]@b-R[~keep]; errs.append(np.sqrt(np.mean(e**2,axis=0)))
  b=np.linalg.lstsq(M,R,rcond=None)[0]; print(' ',typ,'worst lab curve/rate',np.round([max(np.max(e[:-1]) for e in errs),max(e[-1] for e in errs)],5),'physical coefs',np.round(np.linalg.lstsq(f,b.T,rcond=None)[0].T,4).tolist())
print('lab drift relationship for best coordinate hypothesis');
for nm,xx in X.items():
 v=np.c_[np.ones(d0),xx]@th[2:]; scale=v@latent/(v@v); print(nm,'latent',np.round(latent/scale-v,4))
