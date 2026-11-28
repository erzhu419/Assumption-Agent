import sys
sys.path.insert(0,'/holdout_task')
from batch_solver import _spatial,_fit,_time_basis,solve_batch
from probe import s,tr,t,np
from scipy.optimize import least_squares
from sklearn.model_selection import KFold
x,y,traj=(tr[k] for k in ('x','y','traj')); a=_spatial(s,x); one,f,g=a[:,0],a[:,5:9],a[:,9:13]; dot=a[:,23:27].sum(1); cross=a[:,31]; dy=np.c_[one,f,g,dot]; slow=np.c_[one,g,cross]; fast=np.c_[dy,cross];
order=np.argsort(np.max(abs((x-np.median(x,axis=0))/x.std(0)),axis=1)); splits=[('52/20',np.arange(52),np.arange(52,72)),('56/16',np.arange(56),np.arange(56,72)),('boundary',order[:-16],order[-16:])]+[(f'fold{i}',j,k) for i,(j,k) in enumerate(KFold(5,shuffle=True,random_state=104).split(x))];
for tag,j,k in splits:
 h=_time_basis(t,traj[j]); m=_fit(h,traj[j].T).T
 def pred(d,b):return d[k]@_fit(d[j],b)
 yc=pred(dy,y[j]); st=np.c_[pred(slow,m[:,0]),pred(fast,m[:,1])]@h.T
 # Jointly profile the readout under the compact amplitude subspaces.
 def joint(r,test=False):
  h=-np.expm1(-t[:,None]*r)
  def design(j):return np.concatenate([d[j,None,:]*h[None,:,i,None] for i,d in enumerate((slow,fast))],axis=2).reshape(len(j)*len(t),-1)
  D=design(j); b=_fit(D,traj[j].ravel()); return (design(k)@b).reshape(len(k),-1) if test else D@b-traj[j].ravel()
 rates=least_squares(joint,[.4,1.8],bounds=([.05,.65],[.9,6])).x; sj=joint(rates,True)
 def err(p): e=abs(p-traj[k]); return np.round([e.mean(),e.mean(0).max()],6)
 print(tag,'y',round(abs(yc-y[k]).mean(),6),'two-stage',err(st),'joint',err(sj),'rates',rates.round(4),'y columns',_fit(dy[j],y[j]).round(3) if tag=='52/20' else '')
