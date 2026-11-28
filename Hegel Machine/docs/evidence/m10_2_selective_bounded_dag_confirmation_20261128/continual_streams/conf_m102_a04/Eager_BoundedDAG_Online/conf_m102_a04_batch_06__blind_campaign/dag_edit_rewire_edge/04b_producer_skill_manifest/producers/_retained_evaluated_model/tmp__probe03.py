exec(open('/tmp/explore03.py').read().split('for n in (48')[0])
np.set_printoptions(precision=3,suppress=True,linewidth=180)
P=s['panel_a']; Z=a._calibrate(P,P['v']); print('calibration errors',np.quantile(abs(Z-P['u']),[0,.5,.8,.9,.95,.99,1],axis=0));print('panelB');print(np.c_[s['panel_b']['u'],s['panel_b']['v']][:12]); print('samples z,y,traj');print(np.c_[z[:N],y[:,0],y[:,1],y[:,-1]]);print('time correlations',np.corrcoef(y.T)[0]);
d=a.run_node({'__input__':s});w=d['features'];print('feature singulars',np.linalg.svd(w[:N],compute_uv=False));print('interaction dim affine',np.linalg.lstsq(a._interactions(z),w[:,:4],rcond=None)[1]);
for v in (z[:N],abs(z[:N]),z[:N]**2,z[:N]*np.roll(z[:N],-1,axis=1),w[:N], np.c_[tr['x']]):
 print('corr',np.round(np.corrcoef(np.c_[v,y[:,0],y[:,1],y[:,-1]].T)[:-3,-3:],2).T)
from scipy.spatial.distance import cdist
for v in (z[:N],z[:N]*np.sign(z[:N,:1]),w[:N]):
 m=cdist(v,v);np.fill_diagonal(m,np.inf);k=m.argmin(1);i=np.argsort(m.min(1))[:10];print('nearest', list(zip(i,k[i],m[i,k[i]].round(3),y[i,0].round(3),y[k[i],0].round(3),y[i,-1].round(3),y[k[i],-1].round(3))))
