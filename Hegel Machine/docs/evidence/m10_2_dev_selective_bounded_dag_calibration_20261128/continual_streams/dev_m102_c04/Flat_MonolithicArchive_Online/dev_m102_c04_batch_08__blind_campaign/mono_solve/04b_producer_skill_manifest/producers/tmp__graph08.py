import numpy as np,pickle,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
ls=lambda A,Y:np.linalg.lstsq(A,Y,rcond=1e-10)[0]
def aff(*x):return np.column_stack((np.ones(len(x[0])),*x))
def quad(x):return np.column_stack([x[:,i]*x[:,j] for i in range(4) for j in range(i,4)])
def setup(s):
 a,b,c=[s['panel_'+q] for q in 'abc']; u,v=a['u'],a['v']; keep=np.ones(len(u),bool)
 for _ in range(8):
  w=np.array([ls(aff(u[keep,j]),v[keep,j]) for j in range(4)]).T
  e=np.linalg.norm(v-w[0]-u*w[1],axis=1); keep=e<4*np.median(e)
 wb=ls(aff(b['u']),b['v']); uc=c['u']; w1=ls(aff(uc,quad(uc)),c['v1']); w1[abs(w1)<.001]=0
 def cos(x):return aff(np.cos(2.4*x))
 w2=ls(cos(uc),c['v2']); print('panel residuals',np.std(aff(uc,quad(uc))@w1-c['v1']),np.std(cos(uc)@w2-c['v2']))
 def nodes(x):
  z=(x-w[0])/w[1]; b=aff(z)@wb
  return {'z':z,'b':b,'p':aff(z,quad(z))@w1,'r':cos(z)@w2,'P':aff(b,quad(b))@w1,'R':cos(b)@w2}
 return nodes
if __name__=='__main__':
 d=setup(s)(s['train']['x']); y=s['train']['y']; T=s['train']['traj']; t=s['eval_times']; N=52
 def fit(A):
  yp=A@ls(A[:N],np.c_[y,T][:N]); return np.mean(abs(yp[N:]-np.c_[y,T][N:]),axis=0)
 out=[]
 for j,k in itertools.combinations(d,2):
  p,r=d[j],d[k]; dot=np.sum(p*r,axis=1); cp=np.column_stack([p[:,i]*r[:,j] for i in range(4) for j in range(4)])
  for a,A in [('linear',aff(p,r)),('dot',aff(p,r,dot)),('dot+cross',aff(p,r,dot,p[:,0]*r[:,1])),('all cross',aff(p,r,cp)),('plus b',aff(p,r,dot,p[:,0]*r[:,1],d['b']))]:
   e=fit(A);out.append((e[0],e[1:].mean(),j+k,a))
 for i in range(2):
  print('best',i,sorted(out,key=lambda a:a[i])[:20])
 for K in [2,3,4]:
  def basis(r):return -np.expm1(-t[:,None]*r)
  def err(r):
   A=basis(r);return (A@ls(A,T[:N].T)-T[:N].T).ravel()
  rates=least_squares(err,[1.8,.42,3.5, .12][:K],bounds=(.005,20),max_nfev=1000).x
  print('time rates',rates,'fit rmse',np.std(err(rates)))
