import contextlib,io,json
with contextlib.redirect_stdout(io.StringIO()):
 from chamber_probe import *
from scipy.optimize import least_squares
np.set_printoptions(precision=6,suppress=True)
for key in ('archive_apparatus_log.json','archive_manifest.json'):
 a=json.load(open('/tmp/hegel_continual_archive/'+key)); print(key,str(a)[:12500])
print('test telemetry quantiles',np.quantile(te['aux_chamber_telemetry'],[0,.25,.5,.75,1],axis=0)); print('test and diagnostic panels/missingness')
for s in (te,di):
 print('missing',np.isnan(s['coupon_features_raw']).sum(0));
 for lab in np.unique(s['lab_id']):
  panel=s['ref_coupons'][s['lab_id']==lab]; print(lab,len(panel),np.median(panel.mean(0)),np.median(panel.std(0)))
def normalize(s):
 x=s['coupon_features_raw'].copy(); l=s['lab_id']; z=s['stratum']; panel=s['ref_coupons']
 for j in np.unique(l):
  b=l==j; x[b]=(x[b]-np.median(panel[b].mean(0)))/np.median(panel[b].std(0))
 return x
x=normalize(di); _,inv,n=np.unique(np.c_[di['lab_id'],di['stratum']],axis=0,return_inverse=True,return_counts=True); W=1/n[inv]; present=np.isfinite(x); good=present.all(1)
base=c[:,:,:2]@theta[:2]; br=r[:,:2]@theta[:2]
for tag,fill in [('cell',np.array([np.average(v[g],weights=W[g]) for v,g in zip(x.T,present.T)])),('mean',np.nanmean(x,axis=0)),('zero',np.zeros(4)),('joint',None)]:
 v=np.where(present,x,fill) if fill is not None else xd; d=theta[2]+v@theta[3:]
 for mode in ('single','double'):
  def kern(a):
   s=a[0]; f=a[1] if mode=='double' else 0; return -(1-f)*s*np.expm1(-t/s)-f*taus[0]*np.expm1(-t/taus[0])
  fit=least_squares(lambda a: (base[good]+d[good,None]*kern(a)-y[good]).ravel(),[1.05,.09],bounds=([.1,0],[10,1]),loss='soft_l1',f_scale=.004)
  for k,a in [('fitted',fit.x),('unit',[1.,0.])]:
   ec=base+d[:,None]*kern(a)-y; er=br+d-q; print(tag,mode,k,a,'curve/rate RMSE',[(int(g.sum()),rms(ec[g]),rms(er[g])) for g in (good,~good)],'worst',rms(ec,1).max(),abs(er).max())
  if tag=='cell' and mode=='single': print('missing row drift errors',np.c_[np.where(~good)[0],(q-br)[~good],d[~good]])
