import pickle,runpy,numpy as np
from scipy.optimize import least_squares
M=runpy.run_path('/holdout_task/batch_solver.py'); s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; d=s['train']; y=d['y']; H=d['traj']; T=s['eval_times']; fit=M['_fit']; z=M['_decode'](d['x'],M['_calibration'](s['panel_a'])); w=M['_operator'](s['panel_b'])(z); A=M['_affine'](w); g=w[:,0]*w[:,2]+w[:,1]*w[:,3]; B=np.c_[A,g]; C=M['_affine'](M['_quadratic'](w));
splits=[('52/20',np.arange(52),np.arange(52,72))]
for k in range(12):
 r=np.random.default_rng(k).permutation(72); splits.append((f'random{k}',r[:52],r[52:]))
for metric,sgn in [(np.max(abs(z),axis=1),1),(np.linalg.norm(w,axis=1),1),(np.linalg.norm(w,axis=1),-1)]:
 r=np.argsort(sgn*metric); splits.append(('boundary',r[:52],r[52:]))
forms={'affine':A,'coupled':B,'full':C}; stats={}; scalar={}; temporal={
 'twoexp':(lambda p:-np.expm1(-T[:,None]*p),[.25,1.25],([.01,.05],[5.,7.])),
 'exp_sin':(lambda p:np.c_[-np.expm1(-T*p[0]),np.sin(T*p[1])],[1.25,2.15],([.01,.01],[7.,8.])),
 'exp_lin':(lambda p:np.c_[-np.expm1(-T*p[0]),T],[1.25],([.01],[7.]))}
for nm,i,j in splits:
 for n,X in forms.items():
  v=X[j]@fit(X[i],y[i]); scalar.setdefault(n,[]).append(np.mean(abs(v-y[j])));
  _,_,V=np.linalg.svd(H[i],full_matrices=False)
  for k,P in [('raw',np.eye(len(T))),('rank2',V[:2].T@V[:2])]:
   pred=X[j]@fit(X[i],H[i]@P); er=abs(pred-H[j]).mean(0); stats.setdefault((n,k),[]).append([er.mean(),er.max(),er[-1]])
  if n!='full':
   for k,(f,start,bounds) in temporal.items():
    def residual(p):
     b=f(p); return (H[i]-(b@fit(b,H[i].T)).T).ravel()
    p=least_squares(residual,start,bounds=bounds).x; b=f(p); pred=X[j]@fit(X[i],fit(b,H[i].T).T)@b.T; er=abs(pred-H[j]).mean(0); stats.setdefault((n,k),[]).append([er.mean(),er.max(),er[-1]])
    if nm=='52/20' and n=='affine': print(k,p,'time reconstruction train MAE',abs(residual(p)).mean()); print('coeffs',fit(X[i],fit(b,H[i].T).T))
 p=M['solve_batch'](dict(s,train={k:v[i] for k,v in d.items() if k!='summary'},test={'x':d['x'][j]})); er=abs(p['state_pred']-H[j]).mean(0); stats.setdefault(('deployed','ridge'),[]).append([er.mean(),er.max(),er[-1]]); scalar.setdefault('deployed',[]).append(np.mean(abs(p['y_hat']-y[j])))
for n,l in scalar.items():print('scalar',n,'fixed/mean/worst',l[0],np.mean(l),max(l))
for n,l in stats.items():print('trajectory',n,'fixed / mean / worst (average, max timestamp, final)',*[np.round(v,6) for v in [l[0],np.mean(l,axis=0),np.max(l,axis=0)]])
