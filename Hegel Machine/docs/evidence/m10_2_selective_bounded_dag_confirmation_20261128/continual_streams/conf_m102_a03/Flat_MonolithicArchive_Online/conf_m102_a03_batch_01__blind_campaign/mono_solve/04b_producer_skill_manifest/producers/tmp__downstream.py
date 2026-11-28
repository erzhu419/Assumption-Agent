import numpy as np,pickle,importlib.util,itertools
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];tr=s['train'];p=s['panel_a'];b=s['panel_b'];A=m._calibration(p); z=m._decode(tr['x'],A); Z=m._affine(z)@m._fit(m._affine(b['u']),b['v']); Y=tr['y']; T=tr['traj']; print('calibration',A);print('test latent range',np.quantile(m._decode(s['test']['x'],A),[0,1],axis=0));
for name,W in [('latent',z),('panel B',Z)]:
 D=m._quadratic(W); c=m._fit(D[:52],Y[:52]);print(name,'quad coeff',c); print('linear trajectory coeff',m._fit(m._affine(W[:52]),T[:52])[:,[0,-1]])
 for k in [1,2,3]:
  losses=[]
  for js in itertools.combinations(range(5,15),k):
   d=D[:,list(range(5))+list(js)]; cc=m._fit(d[:52],Y[:52]);losses.append((np.mean((Y[:52]-d[:52]@cc)**2),np.mean(abs(Y[52:]-d[52:]@cc)),js))
  print('sparse',k,sorted(losses)[:3])
 for a,b in itertools.combinations(range(4),2):
  pass
 d=np.c_[m._affine(W),W[:,0]*W[:,2]+W[:,1]*W[:,3]];c=m._fit(d[:52],Y[:52]); print('archived template MAE',np.mean(abs(Y[52:]-d[52:]@c)))
t=s['eval_times']; X=m._affine(z); q,_=np.linalg.qr(X[:52]);QT=q.T@T[:52];
for kind in ['exp2','exp+texp','exp+t','exp+constant']:
 def ts(p):
  f=-np.expm1(-p[0]*t);return np.array([f,{'exp2':lambda:-np.expm1(-p[1]*t),'exp+texp':lambda:t*np.exp(-p[0]*t),'exp+t':lambda:t,'exp+constant':lambda:np.ones(len(t))}[kind]()]);
 def res(p):
  F=ts(p);return (m._fit(F.T,QT.T).T@F-QT).ravel()
 p=least_squares(res,[.5,2] if kind=='exp2' else [1.],bounds=(.00001,20)).x; F=ts(p); c=m._fit(X[:52],m._fit(F.T,T[:52].T).T);r=X[52:]@c@F-T[52:];print(kind,p,'fit',np.std(res(p)),'cv',np.mean(abs(r),axis=0))
