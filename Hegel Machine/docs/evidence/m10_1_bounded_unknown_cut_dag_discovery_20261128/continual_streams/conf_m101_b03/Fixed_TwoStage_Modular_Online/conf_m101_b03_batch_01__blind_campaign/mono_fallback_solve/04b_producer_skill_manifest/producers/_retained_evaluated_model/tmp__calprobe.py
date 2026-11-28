import pickle,numpy as np
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; t=s['eval_times']; q=tr['traj']; n=52
np.set_printoptions(precision=5,suppress=True,linewidth=150)
pa=s['panel_a']; u,v=pa['u'],pa['v']; f=np.c_[np.ones(len(u)),u]
ix=np.arange(len(u))
for i in range(6):
 c=np.linalg.lstsq(f[ix],v[ix],rcond=None)[0]; ix=np.argsort(np.linalg.norm(f@c-v,axis=1))[:45]
for panel,key,ids in [('panel_a','v',ix),('panel_c','v1',np.arange(24)),('panel_c','v2',np.arange(24))]:
 p=s[panel]; uu,vv=p['u'][ids],p[key][ids]
 for degree in [1,2,3,4]:
  P=PolynomialFeatures(degree); F=P.fit_transform(uu); C=np.linalg.lstsq(F,vv,rcond=None)[0]; E=F@C-vv; lev=np.sum(F*np.linalg.pinv(F).T,axis=1); cv=np.mean(abs(E/np.maximum(1-lev[:,None],1e-9))) if len(F)>F.shape[1] else np.nan
  print(panel,key,degree,'MAE',np.mean(abs(E)),'LOO',cv)
  if panel=='panel_a' and degree in [2,3]:
   for j in range(4):
    i=np.argsort(-abs(C[:,j]))[:10]; print(j,list(zip(P.get_feature_names_out()[i],np.round(C[i,j],5))))
