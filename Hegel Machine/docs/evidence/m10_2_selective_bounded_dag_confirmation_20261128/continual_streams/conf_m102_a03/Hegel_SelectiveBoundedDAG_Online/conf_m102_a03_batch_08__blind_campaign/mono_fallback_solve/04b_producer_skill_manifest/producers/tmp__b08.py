import numpy as np,pickle
from scipy.optimize import least_squares
from scipy.special import erf
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u,v=[s['panel_b'][k] for k in ('u','v')]; z=np.load('/tmp/feat08.npz')['z']; fit=lambda A,Y: np.linalg.lstsq(A,Y,rcond=1e-11)[0]; aff=lambda u:np.c_[np.ones(len(u)),u]
ops={'sin':np.sin,'tanh':np.tanh,'atan':np.arctan,'rat':lambda x:x/(1+x*x),'erf':erf,'arcsinh':np.arcsinh,'soft':lambda x:x/np.sqrt(1+x*x),'eg':lambda x:x*np.exp(-x*x),'sinh':np.sinh,'rad':lambda x:x*np.sqrt(1+x*x),'log':lambda x:x*np.log1p(x*x),'cube':lambda x:x**3}
for name,fn in ops.items():
 def calc(u,h,mode):
  p=fn(h[0]*u)
  if mode==0:return np.c_[aff(u),p]
  if mode==1:return aff(u+h[1]*p)
  if mode==2:return np.c_[aff(u),p,u**2]
  if mode==3:return aff(u+h[1]*fn(h[0]*u+h[2]))
  if mode==4:return aff(u+h[1]*fn(h[0]*u+np.array(h[2:6])))
  if mode==5:return np.c_[aff(u),p,u*np.roll(u,-1,axis=1)]
 for mode,k in [(0,1),(1,2),(2,1),(3,3),(4,6),(5,1)]:
  def res(h):
   A=calc(u,h,mode);return (A@fit(A,v)-v).ravel()
  best=min([least_squares(res,[h]+[-.4]+[0]*(k-2) if k>=2 else [h],bounds=([.08]+[-4]*(k-1),[9]+[4]*(k-1)),max_nfev=160) for h in (0.7,1.5,2.6,4.0)],key=lambda a:a.cost)
  r=np.linalg.norm(best.fun)/np.sqrt(v.size)
  if mode<2 or r<.003:
   print(name,mode,round(r,8),np.round(best.x,5));
  if r<.001:
   A=calc(u,best.x,mode);np.save('/tmp/Bexact08.npy',calc(z,best.x,mode)@fit(A,v))
A=aff(u); w=fit(A,v); print('v mapped back\n',np.c_[u,(v-w[0])@np.linalg.inv(w[1:])].round(4));print('y r',np.load('/tmp/feat08.npz')['y'].min(),np.load('/tmp/feat08.npz')['y'].max())
