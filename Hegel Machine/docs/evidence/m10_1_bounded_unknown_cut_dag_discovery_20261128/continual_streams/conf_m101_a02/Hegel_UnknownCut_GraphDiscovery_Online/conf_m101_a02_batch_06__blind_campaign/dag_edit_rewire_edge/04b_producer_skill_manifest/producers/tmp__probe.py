import pickle, numpy as np, runpy, itertools
from scipy.optimize import least_squares
from sklearn.linear_model import Ridge, OrthogonalMatchingPursuit
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; A=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py'); B=runpy.run_path('/holdout_task/_frozen_nodes/node_01.py'); P=A['run_node']({'__input__':s}); Z=B['run_node']({'node_00':P}); C=s['panel_c']; t=s['eval_times']; T=s['train']['traj']; y=s['train']['y']; x=P['cal_tr']; z=Z['z_tr']; u=C['u']; uz=B['run_node']({'node_00':{'cal_tr':u,'cal_te':u,'panel_b':s['panel_b']}})['z_tr']
def feats(x,z):
 W=np.c_[x,z]; names=['x'+str(i) for i in range(4)]+['z'+str(i) for i in range(4)]; out=list(W.T); keys=names.copy()
 for i,j in itertools.combinations_with_replacement(range(8),2): out.append(W[:,i]*W[:,j]); keys.append(names[i]+'*'+names[j])
 return np.array(out).T,keys
if __name__=='__main__':
 for mode,a,b in [('direct',u,uz),('inverse',A['_invert'](u,A['_fit_forward'](**s['panel_a'])),u)]:
  for k in ('v1','v2'):
   for name,F in [('x',a),('z',b),('xz',np.c_[a,b]),('x diag2',np.c_[a,a*a]),('z diag2',np.c_[b,b*b]),('xz diag2',np.c_[a,b,a*a,b*b,a*b]),('x+quadratic',np.c_[a,np.array([a[:,i]*a[:,j] for i,j in itertools.combinations_with_replacement(range(4),2)]).T])]:
    F=np.c_[np.ones(len(F)),F]; v=C[k]; ai=np.linalg.pinv(F); err=(F@ai@v-v)/(1-np.sum(F*ai.T,axis=1))[:,None]; print(mode,k,name,np.mean(abs(err)).round(5))
 for basis in [lambda k:np.array([-np.expm1(-k*t),t]),lambda k:np.array([-np.expm1(-k*t),np.log1p(t)]),lambda k:np.array([-np.expm1(-k*t),t,np.ones(len(t))]),lambda k:np.array([-np.expm1(-k*t),t,t*t])]:
  f=lambda k:T@np.linalg.pinv(basis(k[0]))@basis(k[0]); r=least_squares(lambda k:(f(k)-T).ravel(),[.7],bounds=(.02,8)); print('time',r.x,np.mean(abs(r.fun)),np.linalg.svd(T-f(r.x),compute_uv=False).round(4))
 F,keys=feats(x,z); R=np.c_[y,T[:,0],T[:,-1]]; scale=F.std(0); G=(F-F.mean(0))/scale
 for alpha in [0,.00001,.001,.01,.1,1,10]:
  m=Ridge(alpha=alpha).fit(G[:52],R[:52]); print('ridge',alpha,np.mean(abs(m.predict(G[52:])-R[52:]),axis=0).round(4));
  if alpha in [.01,1]:
   for c in m.coef_: print([(keys[i],round(c[i]/scale[i],3)) for i in np.argsort(abs(c))[-10:][::-1]])
 for k in [4,6,8,10,13,16,20,25]:
  m=OrthogonalMatchingPursuit(n_nonzero_coefs=k).fit(G[:52],R[:52]); print('omp',k,np.mean(abs(m.predict(G[52:])-R[52:]),axis=0).round(4));
  if k in [8,16]:
   for c in m.coef_: print([(keys[i],round(c[i]/scale[i],3)) for i in np.flatnonzero(c)])
