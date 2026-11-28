import numpy as np,pickle
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True,linewidth=140)
p=PolynomialFeatures(2); c=s['panel_c']; u=c['u']; a=p.fit_transform(u); print('panel c features',p.get_feature_names_out()); print(np.linalg.lstsq(a,c['v2'],rcond=None)[0]); print('v1 cubic-identifiable subset')
for features in [np.c_[np.ones(len(u)),u,np.sin(u),np.cos(u)],np.c_[a,np.sin(u),np.cos(u)],np.c_[np.ones(len(u)),u,u**2,u**3,np.sin(u)]]:
 r=c['v1']-features@np.linalg.lstsq(features,c['v1'],rcond=None)[0]; print(np.std(r,axis=0))
x=s['train']['x']; y=s['train']['y']; T=s['train']['traj']; a=p.fit_transform(x)
for n,v in [('y',y),('early',T[:,0]),('late',T[:,-1])]:
 b=np.linalg.lstsq(a,v,rcond=None)[0]; print(n,'coef',b,'residual std',np.std(v-a@b)); Q=np.zeros((4,4)); Q[np.triu_indices(4)]=b[5:]; Q=(Q+Q.T)/2; print('Hessian eigenvalues',np.linalg.eigvalsh(Q))
