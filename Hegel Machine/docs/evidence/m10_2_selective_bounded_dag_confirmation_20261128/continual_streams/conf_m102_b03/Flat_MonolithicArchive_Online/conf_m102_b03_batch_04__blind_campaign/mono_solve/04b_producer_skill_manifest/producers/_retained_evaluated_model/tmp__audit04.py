exec(open('/tmp/validate_batch04.py').read().split('records=[]')[0])
np.set_printoptions(precision=5,suppress=True,linewidth=150)
print('available fields:',list(s),list(train),'telemetry:',s['telemetry_vec'],'times:',s['eval_times']);print('centered time singular values',np.linalg.svd(t,compute_uv=False))
z=m._calibrate(s['panel_a'],x); u=s['panel_c']['u']; pv=s['panel_c']['v1']; qv=s['panel_c']['v2']; p,q=m._operators(s['panel_c'],z)
from scipy.optimize import minimize_scalar,least_squares
basis={'coordinatewise quadratic':lambda z:np.c_[z,z*z], 'cyclic products':lambda z:z*np.roll(z,-1,axis=1),'cyclic differences plus squares':lambda z:np.c_[z*z,(z-np.roll(z,-1,axis=1))**2], 'all products':lambda z:np.c_[z,np.array([z[:,i]*z[:,j] for i in range(4) for j in range(i,4)]).T]}
for name,f in basis.items():
 A=m._design(f(u)); residual=A@m._ls(A,pv)-pv; loo=residual/np.maximum(1e-8,1-np.diag(A@np.linalg.pinv(A)))[:,None]; print('first panel LOO MAE',name,abs(loo).mean())
for name,fn in [('softplus',lambda z,k:np.logaddexp(0,k*z)),('rational',lambda z,k:1/(1+k*z*z)),('rational cyclic differences',lambda z,k:np.c_[1/(1+k*z*z),1/(1+k*(z-np.roll(z,-1,axis=1))**2)])]:
 e=[]
 for j in range(len(u)):
  i=np.arange(len(u))!=j
  def A(z,k):return m._design(fn(z,k))
  def loss(k):
   d=A(u[i],k);return np.sum((d@m._ls(d,qv[i])-qv[i])**2)
  k=minimize_scalar(loss,bounds=(.05,6),method='bounded').x;e.append(abs(A(u[j:j+1],k)@m._ls(A(u[i],k),qv[i])-qv[j]).mean())
 print('second panel nested LOO MAE',name,np.mean(e))
h=m._basis(np.asarray(s['eval_times']),t); theta=m._ls(b,t); print('per-feature time coefficients (intercept, P, Q, P-by-Q):\n',theta)
r=(t-(b@theta)); print('unrestricted residual svs',np.linalg.svd(r,compute_uv=False));print('relaxation residual svs',np.linalg.svd(t-(m._ls(h,t.T).T@h.T),compute_uv=False));print('train/test max abs calibrated',abs(z).max(0),abs(m._calibrate(s['panel_a'],s['test']['x'])).max(0))
