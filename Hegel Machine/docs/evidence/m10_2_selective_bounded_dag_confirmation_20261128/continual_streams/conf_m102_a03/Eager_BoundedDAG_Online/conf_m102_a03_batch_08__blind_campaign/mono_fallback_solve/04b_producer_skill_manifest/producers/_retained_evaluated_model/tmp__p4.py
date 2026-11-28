import pickle,numpy as np
from sklearn.preprocessing import PolynomialFeatures
from scipy.linalg import lstsq
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];np.set_printoptions(precision=4,suppress=True,linewidth=180)
def fit(A,Y,rob=False):
 c=lstsq(A,Y)[0]
 if rob:
  for _ in range(60):
   er=np.linalg.norm(A@c-Y,axis=1); w=np.minimum(1,(1.5*np.median(er)/np.maximum(er,1e-12))**2);c=lstsq(A*w[:,None],Y*w[:,None])[0]
 return c
for k in ['panel_a','panel_b','panel_c']:
 d=s[k]; u=d['u']; print(k)
 for name,v in d.items():
  if name=='u':continue
  for deg in (1,2,3):
   p=PolynomialFeatures(deg).fit(u); F=p.transform(u); c=fit(F,v,k=='panel_a'); er=np.linalg.norm(F@c-v,axis=1); print(name,deg,np.quantile(er,[0,.25,.5,.75,.8,.9,1]));
   if deg==1 or (deg==2 and k=='panel_a'):
    for j in range(4):print(j,[(p.get_feature_names_out()[i],round(c[i,j],4)) for i in np.argsort(-abs(c[:,j]))[:10]])
   if deg==1:
    keep=er<2*np.median(er); E=v-F@c; print('res correlation with quadratic/cubic/trig'); p=PolynomialFeatures(3).fit(u); A=p.transform(u)[:,5:]; A=np.c_[A,np.sin(2*u),np.cos(2*u)]; names=list(p.get_feature_names_out()[5:])+[f'{h}{i}' for h in ['sin','cos'] for i in range(4)]; ar=A[keep]-F[keep]@fit(F[keep],A[keep]); ee=E[keep]-F[keep]@fit(F[keep],E[keep]); C=ar.T@ee/np.linalg.norm(ar,axis=0)[:,None]/np.linalg.norm(ee,axis=0);print([[(names[i],round(C[i,j],3)) for i in np.argsort(-abs(C[:,j]))[:5]] for j in range(4)])
x=s['train']['x']; Y=np.c_[s['train']['y'],s['train']['traj']];
for dg in [1,2,3]:
 p=PolynomialFeatures(dg).fit(x);F=p.transform(x); c=fit(F[:52],Y[:52]); E=abs(F[52:]@c-Y[52:]); print('direct',dg,E.mean(0),E[:,1:].mean());print(np.c_[p.get_feature_names_out(),c[:,[0,1,8]]][:15])
