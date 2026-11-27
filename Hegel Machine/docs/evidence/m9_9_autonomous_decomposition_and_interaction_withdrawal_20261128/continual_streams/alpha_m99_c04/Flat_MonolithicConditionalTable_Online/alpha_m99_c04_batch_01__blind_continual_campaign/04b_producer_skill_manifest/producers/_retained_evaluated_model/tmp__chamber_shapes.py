import contextlib,io,json
with contextlib.redirect_stdout(io.StringIO()):
 exec(open('/tmp/chamber_cv.py').read().split('for name,x in stds.items():')[0])
from scipy.optimize import least_squares
from functools import lru_cache
x=stds['diag']; drift=th[2]+x@th[3:]; w=np.r_[np.ones(len(t)),.65]; ys=np.c_[y,v]

def fit(mask,kind,free_hold=False):
 def evaluate(p):
  ta=tau*np.exp(p[:2]) if kind=='both' else tau*np.exp([0,p[0]]) if kind=='slow' else tau
  b=np.r_[-np.expm1(-t[:,None]/ta),[1/ta]]
  features=np.column_stack([drift,np.ones(N)]) if kind=='offset' else drift[:,None]
  base=(hold[:,None]*th[:2])@b.T
  if kind=='memory': features=np.c_[features,z]
  A=[np.column_stack([features,hold]) if free_hold else features]*2+[drift[:,None]]
  D=np.concatenate([A[j][:,None,:]*b[None,:,j:j+1] for j in range(2)]+[A[2][:,None,:]*np.r_[t,1][None,:,None]],axis=2)
  X=(D[mask]*w[None,:,None]).reshape(-1,D.shape[-1]); Y=((ys-base)[mask]*w).ravel(); coef=np.linalg.lstsq(X,Y,rcond=None)[0]
  return (base+D@coef),coef,ta
 def res(p): return ((evaluate(p)[0]-ys)[mask]*w).ravel()
 p=least_squares(res,np.zeros(2 if kind=='both' else 1),bounds=(-.6,.6)).x if kind in ('both','slow') else None
 return evaluate(p)
for kind,fh in [('fixed',False),('offset',False),('slow',False),('both',False),('fixed',True),('both',True),('memory',False)]:
 e=np.zeros_like(ys); params=[]
 for l in np.unique(labs):
  b=labs==l; p,c,ta=fit(~b,kind,fh); e[b]=ys[b]-p[b]; params.append(np.r_[ta,c]);
 p,c,ta=fit(np.ones(N,dtype=bool),kind,fh)
 by_lab=np.array([[np.sqrt(np.mean(e[labs==l,:-1]**2)),np.sqrt(np.mean(e[labs==l,-1]**2))] for l in np.unique(labs)])
 print(kind,'free_hold',fh,'worst',by_lab.max(0).round(5),'mean',by_lab.mean(0).round(5),'parameters',np.r_[ta,c].round(5),'fold range',np.ptp(params,axis=0).round(5))
 if kind=='fixed' and not fh:
  print('by missingness:',[(j,np.sqrt(np.mean(e[np.isnan(g['coupon_features_raw']).any(1)==j]**2,axis=0))[[0,5,-2,-1]].round(5)) for j in (False,True)])
  print('diag drift ranges by lab',[(l, drift[labs==l].mean(),drift[labs==l].std()) for l in np.unique(labs)])
for entry in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 a=np.array(entry['facility_telemetry_trace']);print(entry['completed_campaign_short_name'],'first',a[0].round(4),'sd',a.std(0).round(4),'channel1',a[::4,1].round(4),'channel5',a[::4,5].round(4))
