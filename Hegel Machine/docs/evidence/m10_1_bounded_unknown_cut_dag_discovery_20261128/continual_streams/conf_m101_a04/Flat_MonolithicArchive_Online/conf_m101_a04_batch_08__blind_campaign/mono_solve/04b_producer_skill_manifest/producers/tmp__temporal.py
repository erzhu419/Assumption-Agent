import sys,pickle,numpy as np,importlib.util,itertools
from scipy.optimize import least_squares
p='/tmp/hegel_continual_archive/conf_m101_a04_batch_06_solver.py';sp=importlib.util.spec_from_file_location('archive',p);ar=importlib.util.module_from_spec(sp);sp.loader.exec_module(ar)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tt=s['eval_times'];T=s['train']['traj']; y=s['train']['y']; raw=np.r_[s['train']['x'],s['test']['x']];x=ar._calibrate(s['panel_a'],raw);c,d=ar._branches(s['panel_c'],x);bp=s['panel_b'];b=ar._bias(x)@ar._fit(ar._bias(bp['u']),bp['v']); n=52
bias=ar._bias;fit=ar._fit
np.set_printoptions(precision=5,suppress=True)
if __name__=='__main__':
 print('times',tt);print('ranges',x.min(0),x.max(0));print('raw recalibration spread');a=s['panel_a'];X=ar._calibration_basis(a['u']); mask=np.linalg.norm(a['v'],axis=1)<5;W=np.stack([fit(X[mask,i],a['v'][mask,i]) for i in range(4)]); print(W);print('residuals',np.sort(np.linalg.norm((X*W.T[None,:,:].transpose(0,2,1)).sum(2)-a['v'],axis=1))[-6:]);
 u,sv,vt=np.linalg.svd(T[:n],full_matrices=False); print('temporal modes',vt[:4]);
 fun={'exp':lambda k:-np.expm1(-k*tt),'rat':lambda k:tt/(1+k*tt),'log':lambda k:np.log1p(k*tt),'sin':lambda k:np.sin(k*tt),'sat':lambda k:np.tanh(k*tt),'damp':lambda k:tt*np.exp(-k*tt),'quad':lambda k:tt**2/(1+k*tt),'lin':lambda k:tt}
 def run(names):
  def design(p):return np.stack([fun[f](k) for f,k in zip(names,p)],axis=1)
  def resid(p):
   Z=design(p);return (T[:n]-(Z@fit(Z,T[:n].T)).T).ravel()
  out=min((least_squares(resid,guess,bounds=(.005,8),max_nfev=80) for guess in ([.5,1.8,3.][:len(names)],[2.,.3,1.][:len(names)])),key=lambda o:o.cost)
  Z=design(out.x);amps=fit(Z,T.T).T
  return np.sqrt(np.mean(out.fun**2)),names,out.x,amps
 results=sorted([run(k) for q in (2,3) for k in itertools.combinations_with_replacement(fun,q)],key=lambda r:r[0]);
 for e,k,p,A in results[:18]: print('temporal',e,k,p,'ampstd',A.std(0))
 for e,k,p,A in results[:6]:
  print('features',k)
  for name,f in [('c,d',np.c_[c,d]),('x,c,d',np.c_[x,c,d]),('x,b,c,d',np.c_[x,b,c,d]),('c',c),('d',d),('x',x)]:
   G=bias(f);tar=np.c_[y,A];w=fit(G[:n],tar[:n]);print(name,np.mean(abs(G[n:72]@w-tar[n:]),0))
