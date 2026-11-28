import pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True)
def I(x):return np.c_[np.ones(len(x)),x]
def ls(x,y):return np.linalg.lstsq(x,y,rcond=1e-9)[0]
def fit(d,y):return d@ls(d,y)-y
fns={'sin':np.sin,'cos':np.cos,'tanh':np.tanh,'asinh':np.arcsinh,'satr':lambda x:x/(1+x*x),'sabs':lambda x:x/(1+abs(x)),'gauss':lambda x:x*np.exp(-x*x),'fold':lambda x:abs(x),'cube':lambda x:x**3,'sq':lambda x:x*x}
for k in ['panel_a','panel_b','panel_c']:
 p=s[k]; u=p['u'];
 for j in p.keys()-{'u'}:
  v=p[j]; keep=np.ones(len(u),bool) if k!='panel_a' else np.linalg.norm(v,axis=1)<4; a=u[keep]; b=v[keep]; print(k,j,'ranges',a.min(0),a.max(0),b.min(0),b.max(0),'lin RMS',np.std(fit(I(a),b)))
  for name,fn in fns.items():
   def d(q): return I(np.c_[a,fn(q[0]*a)])
   opt=min((least_squares(lambda q:fit(d(q),b).ravel(),[q],bounds=(.15,14),max_nfev=70) for q in [.8,2.15,3.5,5,7,10,13]),key=lambda r:r.cost)
   print(name,round(opt.x[0],4),round(np.std(opt.fun),6),end='; ')
  print();
  if k=='panel_a':
   rr=[np.roll(a,j,axis=1) for j in range(4)]; d=I(np.c_[a,a*a,a**3,*[rr[0]*r for r in rr[1:]]]); print('poly rms',np.std(fit(d,b)));print('coeff lin clean',ls(I(a),b)); print('first rows',np.c_[a,b][:9])
  elif k=='panel_b':print('first rows',np.c_[a,b][:10]);print('SVD',np.linalg.svd(b,compute_uv=False))
t=s['train']; _,_,vt=np.linalg.svd(t['traj'][:52],full_matrices=False); print('modes',vt[:2]); y=np.c_[t['y'],t['traj']@vt[:2].T];print('mode-to-y',ls(I(y[:52,1:]),y[:52,0]),np.std(fit(I(y[:52,1:]),y[:52,0]))); print('train fit polynomial',*[np.std(fit(I(np.c_[t['x'][:52],t['x'][:52]**2,t['x'][:52]**3]),y[:52]),axis=0)])
