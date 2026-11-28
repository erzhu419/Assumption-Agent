import sys,pickle,numpy as np
from sklearn.preprocessing import PolynomialFeatures
sys.path.insert(0,'/holdout_task'); from batch_solver import calibrate,temporal
np.set_printoptions(precision=4,suppress=True,linewidth=200)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; off,gain=calibrate(s); z=(tr['x']-off)/gain
q,a=temporal(s['eval_times'],tr['traj']); Y=np.c_[tr['y'],a]
for pkey,vkey,deg in [('panel_b','v',1),('panel_c','v1',2),('panel_c','v2',2)]:
 p=s[pkey]; f=PolynomialFeatures(deg); F=f.fit_transform(p['u']); c=np.linalg.lstsq(F,p[vkey],rcond=None)[0]; print(pkey,vkey); print(list(zip(f.get_feature_names_out(),c))); print('fit to outputs',np.linalg.lstsq(np.c_[np.ones(len(z)),f.transform(z)@c],Y,rcond=None)[0]);
for deg in (2,3,4):
 f=PolynomialFeatures(deg); F=f.fit_transform(z); c=np.linalg.lstsq(F[:52],Y[:52],rcond=None)[0]; print('degree',deg,'CV',np.mean(abs(F[52:]@c-Y[52:]),0));
 if deg==3:
  for j in range(3): print(sorted(zip(f.get_feature_names_out(),c[:,j]),key=lambda a:-abs(a[1]))[:22])
