import pickle,json,numpy as np
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import least_squares
np.set_printoptions(precision=5,suppress=True)
d=pickle.load(open('public_batch.pkl','rb'))[0]['dataset']; c=d['calibration_load_cells']; a,b=c['bridge_raw'],c['true_channel_loads']; good=np.max(np.abs(a),axis=1)<8
p=PolynomialFeatures(3); D=p.fit_transform(b); print('excluded',np.where(~good)[0]); M=np.linalg.lstsq(D[good],a[good],rcond=None)[0]; print('cubic coefficients',[(n,m.round(5)) for n,m in zip(p.get_feature_names_out(),M) if np.max(abs(m))>0.001]); print('residual',np.linalg.norm((a-D@M)[good],axis=0))
for e in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')): print('metadata',{k:v for k,v in e.items() if k!='facility_telemetry_trace'})
for t in ['train','test']:
 q=d[t]; print(t,'strain_0',q['strain_0'].min(),q['strain_0'].max()); print('history first/last mean raw',q['past_channel_raw'].mean(0)[[0,-1]])
