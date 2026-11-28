import runpy,pickle,numpy as np
q=runpy.run_path('/holdout_task/batch_solver.py'); s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; x=t['x']; y=t['y']; z=t['traj']; times=s['eval_times']; n=len(x)
a,b=q['_branches'](s,x); mix=a[:,0]*b[:,1]; dot=np.sum(a*b,axis=1); P=np.c_[a,dot]; Q=np.c_[b,mix]; one=np.ones(n); H=np.c_[P,Q]; fit=q['_fit']; predict=q['_predict_state']
from scipy.optimize import least_squares
splits=[np.arange(56,72),*np.array_split(np.random.default_rng(51).permutation(n),5),np.argsort(np.linalg.norm(x-x.mean(0),axis=1))[-16:]]
for name,D,E in [('sparse',Q,H),('with constants',np.c_[one,Q],np.c_[one,H]),('full',np.c_[one,H],np.c_[one,H]),('no fast mix',Q,H[:,:-1]),('no fast B',Q,P)]:
 errors=[]
 for te in splits:
  tr=np.setdiff1d(np.arange(n),te); obs=z[tr].ravel()
  def design(r,ix):
   f,g=-np.expm1(-r[:,None]*times); return np.concatenate((D[ix,None,:]*f[None,:,None],E[ix,None,:]*g[None,:,None]),axis=2).reshape(len(ix)*len(times),-1)
  def res(r):
   M=design(r,tr); return M@fit(M,obs)-obs
  o=least_squares(res,[.41,1.81],bounds=([.05,.9],[.85,4])); C=fit(design(o.x,tr),obs); e=abs((design(o.x,te)@C).reshape(len(te),len(times))-z[te]); errors.append([e.mean(),e.mean(0).max()]);
  if len(errors)==1: print(name,'rates',o.x,'coef',C.round(5),'train rmse',np.mean(o.fun**2)**.5)
 print(name,'trajectory MAE and worst time by split',np.round(errors,6).tolist())
for name,D in [('no intercept',H[:,:-1]),('intercept',np.c_[one,H[:,:-1]])]:
 print(name,'y MAE',np.round([np.mean(abs(D[te]@fit(D[tr:=np.setdiff1d(np.arange(n),te)],y[tr])-y[te])) for te in splits],6))
