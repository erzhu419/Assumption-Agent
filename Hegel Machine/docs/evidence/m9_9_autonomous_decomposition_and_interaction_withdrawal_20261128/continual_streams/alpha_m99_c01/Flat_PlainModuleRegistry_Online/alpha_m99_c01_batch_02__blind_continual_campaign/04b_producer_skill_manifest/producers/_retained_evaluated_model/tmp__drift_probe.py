import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from chamber_probe import *
from itertools import combinations
D=np.c_[y,r]; F=cf.T; intrinsic=(a@np.r_[b[:2],np.zeros(5)]); ir=c@np.r_[b[:2],np.zeros(5)]; Y=D-np.c_[intrinsic,ir]; labs=diag['lab_id']; strs=diag['stratum']
full=lambda sels:{k: np.concatenate([s[k] for s in sels],axis=0) for k in ('lab_id','stratum','coupon_features_raw','ref_coupons')}
def transforms(refsets,imputesets):
 ref=full(refsets); imp=full(imputesets)
 def align(d):
  out=d['coupon_features_raw'].copy()
  for l in np.unique(d['lab_id']):
   m=d['lab_id']==l; rr=ref['ref_coupons'][ref['lab_id']==l]; out[m]=(out[m]-np.median(rr.mean(0)))/np.median(rr.std(0))
  return out
 xx=align(imp);_,v,n=np.unique(np.c_[imp['lab_id'],imp['stratum']],axis=0,return_inverse=True,return_counts=True); w=1/n[v]; mm=np.nansum(w[:,None]*xx,0)/(np.isfinite(xx).T@w); u=align(diag); return np.where(np.isfinite(u),u,mm)
parts=[tr,te,diag]; sets=list(sum((list(combinations(range(3),k)) for k in range(1,4)),[])); names=['tr','te','dg']; key=lambda s: '+'.join(names[i] for i in s)
records=[]
for ri in sets:
 for ii in sets:
  u=transforms([parts[j] for j in ri],[parts[j] for j in ii]); v=b[2]+u@b[3:]; w=np.r_[np.ones(len(t)),2.]; yy=Y*w; ff=F*w
  for typ in ['dr','dr+1','dr+hold']:
   g=v[:,None] if typ=='dr' else np.c_[v,np.ones(len(v))] if typ=='dr+1' else np.c_[v,np.ones(len(v)),h]
   M=np.einsum('ij,kt->itjk',g,ff).reshape(len(v),len(w),-1); pred=np.zeros_like(D)
   for l in np.unique(labs):
    m=labs!=l; q=np.linalg.lstsq(M[m].reshape(-1,M.shape[-1]),yy[m].ravel(),None)[0]; pred[~m]=M[~m]@q/w
   q=np.linalg.lstsq(M.reshape(-1,M.shape[-1]),yy.ravel(),None)[0]; e=pred-Y
   rms=np.sqrt(np.mean(e**2,axis=0)); records.append((np.sqrt(np.mean(e**2)),key(ri),key(ii),typ,np.sqrt(np.mean(e[:,:-1]**2)),rms[-1],np.max(abs(e[:,:-1])),q.round(5)))
for z0 in sorted(records,key=lambda q:q[0])[:20]: print(z0)
for z0 in records:
 if z0[1:3] in [('tr+te','tr+te'),('tr+dg','tr+dg'),('dg','dg'),('tr+te+dg','tr+te+dg')]: print('primary',z0)
print('reference stats by panel')
for l in range(4):
 print(l,[(np.median(s['ref_coupons'][s['lab_id']==l].mean(0)),np.median(s['ref_coupons'][s['lab_id']==l].std(0))) for s in [tr,te,diag,full([tr,te]),full([tr,diag])]])
