import contextlib,io,runpy,numpy as np
from scipy.optimize import least_squares
with contextlib.redirect_stdout(io.StringIO()): v=runpy.run_path('/tmp/chamber_structure.py')
p,l,a,h,t,dt,Y,g,tr,te,align=(v[k] for k in ('p','l','a','h','t','dt','Y','g','tr','te','align'))
np.set_printoptions(precision=5,suppress=True,linewidth=150)
b=v['b']; d=v['s']; cal=v['v']['d']['calibration_load_cells']; raw=cal['bridge_raw']; ce=raw@b[1:]+b[0]-cal['true_channel_loads'].sum(1); print('cal residual RMS/max',np.std(ce),max(abs(ce))); print('aux summaries',[(x['aux_chamber_telemetry'].mean(0),x['aux_chamber_telemetry'].std(0)) for x in (tr,te)])
for name,X in [('diag',align([g])),('joint',align([tr,te,g])[-len(l):]),('train_test',np.load('/tmp/chamber_train_test.npz')['X'])]:
 r=h[2]+X@h[3:]; labs=np.eye(4)[g['lab_id']]; missing=np.isnan(g['coupon_features_raw']); U=np.c_[np.ones(len(l)),l,p,r];
 for mode,D in [('base',U),('lab',np.c_[U,labs[:,1:]]),('lab+missing',np.c_[U,labs[:,1:],missing[:,missing.any(0)]]),('lab+r2',np.c_[U,labs[:,1:],r*r])]:
  C=np.linalg.lstsq(D,Y,rcond=None)[0]; E=Y-D@C; Q,S,_=np.linalg.svd(D,full_matrices=False); hat=np.sum(Q[:,S>1e-9]**2,axis=1); ok=hat<.99999; L=E[ok]/(1-hat[ok,None]); print(name,mode,'RMSE',np.sqrt(np.mean(E**2)),np.sqrt(np.mean(L**2)),np.max(abs(L)),'rate',np.sqrt(np.mean(L[:,-1]**2)),'valid',sum(ok),'last coefficients',C[:,-2]);
print('diagnostic missing locations',np.argwhere(np.isnan(g['coupon_features_raw'])))
