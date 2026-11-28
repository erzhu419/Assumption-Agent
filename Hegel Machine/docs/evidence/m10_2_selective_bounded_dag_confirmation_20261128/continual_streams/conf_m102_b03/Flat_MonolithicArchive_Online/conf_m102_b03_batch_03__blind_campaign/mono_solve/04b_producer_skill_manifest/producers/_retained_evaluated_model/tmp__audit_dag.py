import importlib.util,itertools,pickle,numpy as np
from scipy.optimize import least_squares
p=importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(p);p.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=len(tr['x']); panel=s['panel_c']; u=panel['u']; z=m._calibrate(s['panel_a'],tr['x']); c,d=m._branches(panel,z); M,S=m._features(s,tr['x']); print('study/train keys, telemetry shape',list(s),list(tr),np.shape(s['telemetry_vec'])); print('times',s['eval_times']); print('no rowwise batch or relay labels in this schema')
# Compare calibration inverses with the same downstream model.
a=s['panel_a']; A=m._design(a['u']); p0=m._ls(A,a['v']); keep=np.linalg.norm(A@p0-a['v'],axis=1)<3; p0=m._ls(A[keep],a['v'][keep]); e=a['v'][keep]-A[keep]@p0; print('reference inliers and affine versus cyclic inverse RMS',keep.sum(),np.sqrt(np.mean(e**2)),np.sqrt(np.mean((m._calibrate(a,a['v'][keep])-a['u'][keep])**2)))
# Candidate single-coordinate branch nonlinearities, compared by panel leave-one-out.
for key,fs in [('v1',{'softsign':lambda x:x/(1+abs(x)), 'rational':lambda x:x/(1+x*x), 'tanh':np.tanh,'atan':np.arctan}), ('v2',{'silu':lambda x:x/(1+np.exp(-x)), 'rational':lambda x:x/(1+x*x)})]:
 v=panel[key]
 for name,f in fs.items():
  for shifted in [False,True]:
   def design(x,h): return m._design(np.c_[x,f(x*h[0]+(h[1:] if shifted else 0))])
   def res(h,ids):
    B=design(u[ids],h); return (B@m._ls(B,v[ids])-v[ids]).ravel()
   def fit(ids):
    lo=[.12]+([-1.5]*4 if shifted else []); hi=[4.]+([1.5]*4 if shifted else []); opts=[]
    for q in [.6,1.4]:
     opts.append(least_squares(res,[q]+([0.]*4 if shifted else []),bounds=(lo,hi),args=(ids,),max_nfev=75))
    return min(opts,key=lambda a:a.cost).x
   I=np.arange(len(u)); full=fit(I); err=[]
   for i in I:
    ids=I[I!=i]; h=fit(ids); err.extend((design(u[i:i+1],h)@m._ls(design(u[ids],h),v[ids])-v[i]).ravel())
   print('panel',key,name,shifted,'rms full / loo',np.sqrt(np.mean(res(full,I)**2)),np.sqrt(np.mean(np.square(err))), 'params',np.round(full,5))
# Extend the identified response cut by rational and pairwise-difference features, controlling model size with ridge.
D=np.column_stack([f[:,i]+k*f[:,j] for f in (z,c,d) for i,j in itertools.combinations(range(4),2) for k in [-1,1]])
E=np.c_[D**2, D/(1+D**2), 1/(1+D**2)]; E=(E-E.mean(0))/np.maximum(E.std(0),1e-8)
order=np.argsort(np.linalg.norm(z,axis=1)); rng=np.random.default_rng(512); splits=[('52/20',np.arange(52,n)),('large norm',order[-20:]),('small norm',order[:20])];splits += [(f'random{i}',rng.choice(n,20,False)) for i in range(12)]
for j in range(4):
 o=np.argsort(z[:,j]); splits.extend([(f'low{j}',o[:12]),(f'high{j}',o[-12:])])
for reg in [np.inf,1000,100,10,1]:
 scores=[]
 for name,iv in splits:
  it=np.setdiff1d(np.arange(n),iv); H=m._time_basis(s['eval_times'],tr['traj'][it]); amps=m._ls(H,tr['traj'][it].T).T
  def predict(F,y):
   if np.isinf(reg): return F[iv]@m._ls(F[it],y)
   R=E-F@m._ls(F[it],E[it]); B=np.linalg.solve(R[it].T@R[it]+reg*np.eye(R.shape[1]),R[it].T@(y-F[it]@m._ls(F[it],y)))
   return F[iv]@m._ls(F[it],y)+R[iv]@B
  yp=predict(M,tr['y'][it]); tp=predict(S,amps)@H.T; scores.append([np.abs(yp-tr['y'][iv]).mean(),np.abs(tp-tr['traj'][iv]).mean(),np.abs(tp-tr['traj'][iv]).mean(0).max()]);
 print('basis extension',reg,'required',np.round(scores[0],6),'worst',np.round(np.max(scores,0),6))
