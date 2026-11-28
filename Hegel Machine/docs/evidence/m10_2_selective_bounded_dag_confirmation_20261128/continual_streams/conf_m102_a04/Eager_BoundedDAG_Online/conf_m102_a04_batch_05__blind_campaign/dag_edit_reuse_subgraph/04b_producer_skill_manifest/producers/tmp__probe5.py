import pickle,numpy as np
from scipy.optimize import least_squares, minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=5,suppress=True,linewidth=170)
def lin(x,y):return np.linalg.lstsq(x,y,rcond=1e-10)[0]
def quad(u):return np.c_[np.ones(len(u)),u,u*u,*[u[:,i]*u[:,j] for i in range(4) for j in range(i)]]
c=s['panel_c'];u=c['u'];print('v1 coefficients\n',lin(quad(u),c['v1']))
for kind in range(4):
 def dm(u,q):
  w=u*u; return [np.c_[np.ones(len(u)),u,w,w/(1+q[0]*w)],np.c_[np.ones(len(u)),w,w/(1+q[0]*w)],np.c_[np.ones(len(u)),u,w/(1+q[0]*w)],np.c_[np.ones(len(u)),w,w/(1+q*w)]][kind]
 def rr(q):
  x=dm(u,q);return (x@lin(x,c['v2'])-c['v2']).ravel()
 q=least_squares(rr,[.4]*(4 if kind==3 else 1),bounds=(.001,10),gtol=1e-13).x
 print('v2',kind,q,np.sqrt(np.mean(rr(q)**2)),lin(dm(u,q),c['v2']))
for k in ('panel_a','panel_b'):
 a=s[k];print(k)
 for j in range(4):
  u=a['u'][:,j];v=a['v'][:,j]; ss=np.argsort(u);print(j,np.c_[u,v][ss].T)
