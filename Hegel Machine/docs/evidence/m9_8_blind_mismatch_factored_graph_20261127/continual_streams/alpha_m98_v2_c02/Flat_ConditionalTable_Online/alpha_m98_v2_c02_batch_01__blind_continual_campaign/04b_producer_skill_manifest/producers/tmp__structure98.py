exec(open('/tmp/law98.py').read().split('for mode in range')[0])
import json
L=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))
for e in L:
 print('apparatus', [(k,v) for k,v in e.items() if k!='facility_telemetry_trace']); z=np.array(e['facility_telemetry_trace']);print('trace avg/std',z.mean(0),z.std(0))
print('dataset keys',list(d), 'diag keys',list(a),'test keys',list(d['test']))
for key in d:
 if key not in ('train','test','calibration_load_cells'): print('extra',key,str(d[key])[:500])
cal=d['calibration_load_cells']; print('sweep', {k:np.round(v,4).tolist() for k,v in cal['single_channel_sweeps'].items()})
P,C,p=fit(np.arange(len(h)),3); E=Y-P; V=np.linalg.svd(E,full_matrices=False); print('residual svd',V[1]);
for name,X in [('x',x),('raw',a['coupon_features_raw']),('aux',a.get('aux_chamber_telemetry',np.zeros((len(h),1)))),('hold/past',np.c_[h,s]),('signed-square',np.c_[h*np.abs(h),s*np.abs(s)]),('stress diff square',np.diff(np.c_[s,h],axis=1)**2),('lab/stratum',np.c_[a['lab_id'],a['stratum']]),('refs',a['ref_coupons'])]:
 X=np.nan_to_num(X);print(name,'top residual correlations',np.corrcoef(np.c_[V[0][:,:2],X].T)[:2,2:])
# Identify whether separately projected modal amplitudes support a simpler kernel.
for times in [u,np.array([.5784,2.5911]),np.array([.587,2.2]),np.array([.587,2.7])]:
 H=np.stack([expb(v) for v in times]); amp=np.linalg.lstsq(H.T,Y.T,rcond=None)[0].T
 for name,X in [('stress+drift',np.c_[h,s,q]),('stress+intrinsic features',np.c_[h,s,f]),('stress+drift+aux',np.c_[h,s,q,a['aux_chamber_telemetry']])]:
  inv=np.linalg.pinv(X); C=inv@amp; D0=amp-X@C; loo=D0/(1-np.einsum('ij,ji->i',X,inv))[:,None]; print('times',times,name,'C.T',C.T,'fit',rm(Y-X@C@H),'loo_amp',rm(loo@H))
print('feature standardization sensitivity')
for nm, (left,right) in {'train_diag':(d['train'],a),'all_diag':(d['train'],xx),'test_diag':(d['test'],a),'diag_only':(a,a)}.items():
 q1=np.c_[np.ones(len(h)),m['_standardize'](left,right)[1][-len(h):]]@b[2:];print(nm,q1)
