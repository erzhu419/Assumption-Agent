import sys
sys.path.insert(0,'/holdout_task')
import batch_solver as b
from compare_walls import *
import compare_walls as cw
np.set_printoptions(precision=5,suppress=True)
raw=s['y_obs_raw']; print('raw min/max',raw.min(),raw.max(),'banks',s['row_bank_id'][:16]); X,Y=a['ref_rosette'],a['raw_rosette']; keep=abs(Y)<5
for degree in (4,6,8,10,12,14):
 errs=np.zeros_like(Y)
 for idx in np.array_split(np.random.default_rng(20).permutation(len(X)),8): errs[idx]=b.raw_map(X[idx],*b.calibration_map(a,idx,degree=degree))-Y[idx]
 print('degree',degree,'cv inlier error',np.quantile(abs(errs[keep]),[.5,.95,1]))
y=cw.y=b.calibrate(s); print('y range',y.min(),y.max(),'invert residual',abs(b.raw_map(y.reshape(-1,4),*b.calibration_map(a)).ravel()-raw).max())
opts=['p',(0,0),(1,1),(1,0),(0,1)]; rows=[]; tr=np.ones(len(y),bool)
folds=[np.tile(np.arange(ns)//4==i,len(s['mu_shifts'])) for i in range(ns//4)]
for p,q in itertools.product(opts,repeat=2):
 S,U=forward(axis('x',p),axis('y',q)); cv=np.array([np.mean(fit(S,~t,0.0000001)[1][t]**2)**.5 for t in folds]); full=np.mean(fit(S,tr,0.0000001)[1]**2)**.5; mandated=max(np.mean(fit(S,~t)[1][t]**2)**.5 for t in folds); rows.append((np.mean(cv**2)**.5, cv.max(),full,mandated,p,q))
for row in sorted(rows,key=lambda x:x[0]): print(row)
print('telemetry end',[(r['completed_campaign_short_name'],np.array(r['facility_telemetry_trace']).mean(0)[[1,7]]) for r in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))]); print('aux means',s['aux_manifold_telemetry'].mean(0))
