exec(open('/tmp/baseline.py').read().split('for name,b,xx')[0])
from scipy.optimize import least_squares
n=len(hd); y=np.c_[yd,rd]; r=th[2]+xd@th[3:]; later=np.r_[np.cumsum(dt[:0:-1])[::-1],0.]
def mem(u): return pd@(-np.expm1(-dt[:,None]/u)*np.exp(-later[:,None]/u))
def basis(u): return np.vstack((np.c_[-np.expm1(-t[:,None]/u),t],np.r_[1/u,1.]))
b=basis(np.array([tau[0],.55*tau[1]])); a=np.linalg.lstsq(b,y.T,rcond=None)[0].T
z=mem(tau); dz=mem(np.array([tau[0],.55*tau[1]])); hc=np.c_[np.ones(n),di['hold_channel_raw']]@br; pc=di['past_channel_raw']@br[1:]+br[0]; prev=pd[:,-1]; revz=pd[:,::-1]@(-np.expm1(-dt[:,None]/tau)*np.exp(-later[:,None]/tau)); np.set_printoptions(precision=4,suppress=True,linewidth=140)
print('hold, last, drift, A0/theta0-hold, A1/theta1, slope',np.c_[hd,prev,r,a[:,0]/th[0]-hd,a[:,1]/th[1],a[:,2]])
base=np.c_[np.ones(n),hd,r]; choices={'base':[],'bench memory':[z], 'chamber memory':[dz], 'reverse memory':[revz], 'all past':[pd], 'all features':[xd], 'hold channels':[hc[:,:3]], 'mean and last':[pd.mean(1),prev], 'hold-drift products':[hd*r,r*r,hd*hd], 'memory and feature products':[z,z[:,0]*r,z[:,1]*r,hd*r]}
for j,target in enumerate((a[:,0]/th[0]-hd,a[:,1]/th[1],a[:,2])):
 print('component',j)
 for name,extra in choices.items():
  A=np.column_stack([base,*extra]); p=np.linalg.lstsq(A,target,rcond=None)[0]; e=target-A@p; cv=np.array([target[k]-A[k]@np.linalg.lstsq(np.delete(A,k,0),np.delete(target,k),rcond=None)[0] for k in range(n)]); print(name,'rms/cv',np.sqrt(np.mean(e**2)).round(4),np.sqrt(np.mean(cv**2)).round(4),'coef',p)
 e=target-base@np.linalg.lstsq(base,target,rcond=None)[0]; pool = {'past':pd,'past channels':pc.reshape(n,-1),'hold channels':hc,'features':xd,'bench memory':z,'reverse memory':revz,'hold*drift':(hd*r)[:,None],'stress^2':(hd*hd)[:,None]};
 for name,D in pool.items():
  D=D-base@np.linalg.lstsq(base,D,rcond=None)[0]; corr=e@D/(np.linalg.norm(e)*np.maximum(np.linalg.norm(D,axis=0),1e-9)); print('partial',name,corr.round(2))
