import pickle,numpy as np
from calibrated import *
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoLars
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; q=tr['traj']; n=52
np.set_printoptions(precision=5,suppress=True)
c=calibrate(s['panel_a']); a=inverse(x,c); h,m=time_modes(s['eval_times'],q,n); Y=np.c_[tr['y'],m]
b=s['panel_b']; B=np.linalg.lstsq(design(b['u']),b['v'],rcond=None)[0]; print('latent bounds',a.min(0),a.max(0))
for name,z in [('raw',x),('a',a),('aB',design(a)@B),('ab',(a-B[0])@np.linalg.inv(B[1:]))]:
    for degree in [2,3]:
        pf=PolynomialFeatures(degree); F=pf.fit_transform(z); terms=pf.get_feature_names_out(); sd=np.maximum(.001,F[:n].std(0)); sd[0]=1; F/=sd
        for reg in [1e-5,.01,.1,1]:
            W=np.linalg.solve(F[:n].T@F[:n]+np.eye(F.shape[1])*reg,F[:n].T@Y[:n]); P=F[n:]@W
            print(name,degree,reg,'MAE',np.round(np.mean(abs(P-Y[n:]),0),4),round(np.mean(abs(P[:,1:]@h.T-q[n:])),4))
        if name=='a':
            for j in range(3):
                model=LassoLars(alpha=.012).fit(F[:n,1:],Y[:n,j]); ix=np.argsort(-np.abs(model.coef_))[:9]; print(j, list(zip(terms[ix+1],np.round(model.coef_[ix]/sd[ix+1],4))))
    F=np.c_[design(z), z*z,z**3,np.sin(2*z),np.cos(2*z),np.tanh(2*z),np.column_stack([z[:,i]*z[:,j] for i in range(4) for j in range(i)])]; sd=np.maximum(.001,F[:n].std(0)); sd[0]=1; F/=sd
    for reg in [.01,.1,1]:
        W=np.linalg.solve(F[:n].T@F[:n]+np.eye(F.shape[1])*reg,F[:n].T@Y[:n]); P=F[n:]@W; print(name,'add',reg,np.round(np.mean(abs(P-Y[n:]),0),4),round(np.mean(abs(P[:,1:]@h.T-q[n:])),4))
