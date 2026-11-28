import pickle,sys,runpy
import numpy as np
from scipy.optimize import least_squares
from sklearn.preprocessing import PolynomialFeatures
from scipy.linalg import qr
p='/tmp/hegel_continual_archive/'
a=runpy.run_path(p+'conf_m102_a04_batch_00_solver.py')
b=runpy.run_path(p+'conf_m102_a04_batch_01_solver.py')
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; d=s['train']; t=s['eval_times']; X=d['x']; Y=d['y']; T=d['traj']; N=len(X)
lin=lambda x,y:np.linalg.lstsq(x,y,rcond=1e-9)[0]
one=lambda x:np.c_[np.ones(len(x)),x]
o,k=a['_align'](s['panel_a']); Z=(X-o)/k
pb=s['panel_b']; W=one(Z)@lin(one(pb['u']),pb['v']); C=a['_panel_maps'](s['panel_c'],Z)
print('times, tele, align',t,s['telemetry_vec'],o,k)
def basis(r):return -np.expm1(-t[:,None]*r)
def res(r):
 f=basis(r); return (f@lin(f,T.T)-T.T).ravel()
rr=least_squares(res,[.4,1.8],bounds=([.01,.5],[2,10]),gtol=1e-13).x
F=basis(rr); A=lin(F,T.T).T
print('rates',rr,'noise',np.std(res(rr)),'amp',A.std(0),'y correlation',np.corrcoef(np.c_[Y,A].T))
print('range W C',W.min(0),W.max(0),C.min(0),C.max(0))
for archived in [a,b]:
 sc=dict(s,train={k:v[:52] if k!='summary' else v for k,v in d.items()},test={'x':X[52:]}); out=archived['solve_batch'](sc); print('archived',np.mean(abs(out['y_hat']-Y[52:])),np.mean(abs(out['state_pred']-T[52:])))
if __name__=='__main__':
 for name,q in [('Z',Z),('W',W),('C1',C[:,:4]),('C2',C[:,4:]),('C',C),('ZC',np.c_[Z,C]),('WC',np.c_[W,C])]:
  for degree in range(1,4):
   P=PolynomialFeatures(degree).fit_transform(q); scale=np.maximum(P[:52].std(0),.1); scale[0]=1;P=P/scale
   for reg in [0.01,1.]:
    K=P@P.T; D=np.c_[Y,T]; fit=K[:, :52]@np.linalg.solve(K[:52,:52]+reg*np.eye(52),D[:52]); print(name,degree,reg,'fit',np.round(np.mean(abs(fit[:52]-D[:52]),axis=0)[[0,1,8]],4),'val',np.round([np.mean(abs(fit[52:,0]-Y[52:])),np.mean(abs(fit[52:,1:]-T[52:]))],4))
 print('linear coefficients for Y and amplitudes');print(lin(one(Z),np.c_[Y,A]));print(lin(one(C),np.c_[Y,A]));print(lin(one(W),np.c_[Y,A]));print(np.c_[Z,W,Y,A][:12])
