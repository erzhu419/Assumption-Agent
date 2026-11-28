exec(open('/tmp/fit.py').read().split("print('B linear'")[0])
from scipy.optimize import least_squares
np.set_printoptions(precision=5,suppress=True,linewidth=130)
u,v=b['u'],b['v']; A=aff(u); vu=v
for kind in ['linear','cubic','tanh','sin','sinh','atan','softsign','tanh+cubic','outputcube','outputatan','cubic-input','tanh-input']:
 def pred(p,A):
  L=A@p[:20].reshape(5,4); g=p[20:24]; h=p[24:28]; j=p[28:32]
  if kind=='linear': return L
  if kind=='cubic': return L+g*L**3
  if kind=='tanh': return g*np.tanh(L)+h
  if kind=='sin': return g*np.sin(L)+h
  if kind=='sinh': return g*np.sinh(np.clip(L,-6,6))+h
  if kind=='atan': return g*np.arctan(L)+h
  if kind=='softsign': return g*L/np.sqrt(1+L*L)+h
  if kind=='tanh+cubic': return g*np.tanh(L)+h*L**3+j
  if kind=='outputcube': return np.sinh(np.arcsinh(g*L)/3)*h+j
  if kind=='outputatan': return np.arcsinh(g*L)*h+j
  U=A[:,1:]; U=(U+U**3*g) if kind=='cubic-input' else np.tanh(U*g); return aff(U)@p[:20].reshape(5,4)
 p=np.r_[ls(A,v).ravel(),np.ones(4),np.zeros(4),np.zeros(4)];
 if kind in ('outputcube','outputatan'): p[24:28]=1
 sol=least_squares(lambda p:(pred(p,A)-v).ravel(),p,max_nfev=350); r=pred(sol.x,A)-v; print('B',kind,np.mean(abs(r),0),np.max(abs(r)),sol.x[20:]);
 if np.mean(abs(r))<.02: print(sol.x[:20].reshape(5,4)); np.savez('/tmp/B_'+kind,p=sol.x,bx=pred(sol.x,aff(x)),lin=aff(x)@sol.x[:20].reshape(5,4))
print('B samples',np.c_[u,v][:14]);
u,v=c['u'],c['v2'];
for name,z in [('poly2sep',np.c_[aff(u),u*u]),('poly3sep',np.c_[aff(u),u*u,u**3]),('poly4sep',np.c_[aff(u),u*u,u**3,u**4]),('exp',aff(np.exp(.88*u))),('inputsin',np.c_[aff(u+.44*u*u),np.sin(1.25*u)]),('exp+linear',np.c_[aff(np.exp(.88*u)),u])]:
 r=v-z@ls(z,v); h=np.sum(z*np.linalg.pinv(z).T,axis=1); print('C2',name,np.mean(abs(r),0),np.mean(abs(r/(1-h[:,None])),0)); print(ls(z,v) if name=='poly3sep' else '')
Y=np.c_[tr['y'],tr['traj']]; U,S,V=np.linalg.svd(tr['traj'],full_matrices=False); print('time vectors',V[:2]); print('targets',np.c_[x[:24],Y[:24,0],U[:24,:2]*S[:2]])
