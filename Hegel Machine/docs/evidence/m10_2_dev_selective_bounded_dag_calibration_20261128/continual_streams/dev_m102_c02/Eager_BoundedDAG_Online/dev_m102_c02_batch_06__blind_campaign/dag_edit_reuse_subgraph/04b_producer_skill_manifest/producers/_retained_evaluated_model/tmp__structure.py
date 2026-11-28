exec(open('/tmp/probe.py').read().split("print('A row error'")[0]); from scipy.optimize import least_squares
print('telemetry',s.get('telemetry_vec'),s['eval_times'])
N=52
for name,z in [('c',c),('u',u)]:
 L=np.c_[np.ones(len(z)),z]; pairs=list(combinations_with_replacement(range(4),2)); Q=np.c_[L,*[z[:,i]*z[:,j] for i,j in pairs]]; B=np.linalg.lstsq(Q[:N],Y[:N],rcond=None)[0]; E=Q@B-Y; print(name,'quad',abs(E[:N]).mean(0).round(4),abs(E[N:]).mean(0).round(4)); print(np.c_[B[:,0],B[:,-1]].round(3)); print('resid sv',np.linalg.svd(E[:N],compute_uv=False).round(3)); print('target links');
 for w in [L,Q]:
  F=np.c_[w,y]; e=F[N:]@np.linalg.lstsq(F[:N],t[:N],rcond=None)[0]-t[N:]; print('trajectory with measured y',abs(e).mean())
  F=np.c_[w,t[:,-1]]; e=F[N:]@np.linalg.lstsq(F[:N],y[:N],rcond=None)[0]-y[N:]; print('scalar with measured final',abs(e).mean())
 print('quadratic residual rows',np.round(np.c_[z,y,t[:,-1],E[:,0],E[:,-1]][np.argsort(abs(E[:,0]))[-8:]],3))
