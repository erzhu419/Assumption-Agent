exec(open('/tmp/readout_probe.py').read().split('res=[]')[0])
from scipy.optimize import least_squares
from sklearn.linear_model import ARDRegression, OrthogonalMatchingPursuit
from itertools import combinations
qtime=st['eval_times']; T=tr['traj'];o=least_squares(lambda r:(lambda q:(q@np.linalg.lstsq(q,T.T,rcond=None)[0]-T.T).ravel())(-np.expm1(-qtime[:,None]*np.exp(r))),np.log([.39,1.86]));rt=np.exp(o.x);Q=-np.expm1(-qtime[:,None]*rt); A=np.c_[Y[:,0],np.linalg.lstsq(Q,T.T,rcond=None)[0].T];print('rates',rt,'rms',np.sqrt(np.mean(o.fun**2)))
def tst(f,kind):
 a0=[]
 for a,b,sp in folds:
  u=(f-f[a].mean(0))/np.maximum(f[a].std(0),1e-9); pred=[]
  for j in range(3):
   m=ARDRegression(threshold_lambda=kind) if kind>99 else OrthogonalMatchingPursuit(n_nonzero_coefs=kind,tol=None); m.fit(u[a],A[a,j]);pred.append(m.predict(u[b]))
  pred=np.array(pred).T;p=np.c_[pred[:,0],pred[:,1:]@Q.T];v=abs(p-Y[b]);a0.append([v[:,0].mean(),v[:,1:].mean()]);
 return np.array(a0)
p=st['panel_a'];u=p['u'];v=p['v'];h=np.c_[np.ones(len(u)),u];keep=np.linalg.norm(v-h@np.linalg.lstsq(h,v,rcond=None)[0],axis=1)<3; aff=np.linalg.lstsq(h[keep],v[keep],rcond=None)[0]; z0=(x-aff[0])@np.linalg.inv(aff[1:]);z1=(x-aff[0])/aff[1:].diagonal()
for z in (x,z0,z1):
 for z in (z,):
  f,g=arc['_features'](z); f=np.column_stack(f[1:]+g+[pol(z,3)]); fs=[f,pol(z,3)]
  for idx,f in enumerate(fs):
   for j in (6,12,1e3,1e4):
    s=tst(f,j);print(idx,j,s[0].round(4),s.max(0).round(4),s.mean(0).round(4))
