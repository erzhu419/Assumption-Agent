import numpy as np,pickle
from scipy.optimize import least_squares,minimize_scalar
from itertools import combinations_with_replacement as cwr
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];tr=s['train'];t=s['eval_times']; fit=lambda a,b: np.linalg.lstsq(a,b,rcond=1e-10)[0]
def aff(*xs):return np.column_stack([np.ones(len(xs[0])),*xs])
def quad(x):return np.column_stack([x[:,i]*x[:,j] for i,j in cwr(range(x.shape[1]),2)])
def cal(x):
 j=np.roll(x,-1,axis=1);return np.stack((np.ones_like(x),x,j,x*j,x**3),axis=-1)
def coords(s,x,report=False):
 u,v=(s['panel_a'][k] for k in ('u','v'));a=cal(u);mask=np.max(np.abs(v),axis=1)<3
 for _ in range(4):
  c=np.stack([fit(a[mask,i],v[mask,i]) for i in range(4)]);r=np.linalg.norm(np.einsum('nif,if->ni',a,c)-v,axis=1);mask=r<.03 if np.sum(r<.03)>25 else r<.3
 if report: print('calibration',c,'sorted errors',np.sort(r).round(5))
 z=(x-c[:,0])/c[:,1];j=np.arange(4);J=np.zeros((len(x),4,4))
 for _ in range(15):
  zz=np.roll(z,-1,axis=1);J[:,j,j]=c[:,1]+c[:,3]*zz+3*c[:,4]*z*z;J[:,j,(j+1)%4]=c[:,2]+c[:,3]*z
  r=np.einsum('nif,if->ni',cal(z),c)-x;z-=np.clip(np.linalg.solve(J,r[...,None])[...,0],-.5,.5)
 if report: print('inverse err',np.max(abs(np.einsum('nif,if->ni',cal(z),c)-x)), 'z min max',z.min(),z.max())
 return z
def times(t,ys,report=False):
 def b(r):return -np.expm1(-t[:,None]*r)
 def res(r):
  h=b(r);return (h@fit(h,ys.T)-ys.T).ravel()
 r=least_squares(res,[.4,1.8],bounds=([.05,.65],[.9,6])).x;h=b(r)
 if report: print('rates',r,'rms',np.std(res(r)))
 return h
if __name__=='__main__':
 np.set_printoptions(precision=5,suppress=True);z=coords(s,np.r_[tr['x'],s['test']['x']],True);h=times(t,tr['traj'],True);m=fit(h,tr['traj'].T).T;print('MODE std',m.std(0));p=s['panel_c'];u=p['u'];cc=fit(aff(u,quad(u)),p['v1']);print('v1 coefs',cc);v=p['v2'];
 for fun in [lambda u,k:aff(u,np.sin(k*u),np.cos(k*u)),lambda u,k:aff(np.sin(k*u),np.cos(k*u)),lambda u,k:aff(u,np.tanh(k*u)),lambda u,k:aff(u,1/(1+k*u*u)),lambda u,k:aff(np.sin(k*u)),lambda u,k:aff(np.cos(k*u)),lambda u,k:aff(u,np.sin(k*u)),lambda u,k:aff(u,np.cos(k*u)),lambda u,k:aff(u,u*u,np.sin(k*u)),lambda u,k:aff(u,u*u,1/(1+k*u*u))]:
  def loss(k):a=fun(u,k); return np.std(a@fit(a,v)-v)
  g=np.linspace(.1,8,160);i=np.argmin([loss(k) for k in g]);o=minimize_scalar(loss,bounds=(g[max(0,i-1)],g[min(len(g)-1,i+1)]),method='bounded');print('v2 fit',o.x,o.fun)
 for fam,xx in [('raw',tr['x']),('cal',z[:72])]:
  for a in [aff(xx),aff(xx,quad(xx)),aff(xx,quad(xx),xx**3),aff(xx,quad(xx),xx**3,np.sin(2*xx),np.cos(2*xx))]:
   yy=np.c_[tr['y'],m];r=a@fit(a[:52],yy[:52])-yy;print(fam,a.shape[1],np.mean(abs(r[:52]),axis=0),np.mean(abs(r[52:]),axis=0),np.mean(abs(r[52:,1:]@h.T)))
