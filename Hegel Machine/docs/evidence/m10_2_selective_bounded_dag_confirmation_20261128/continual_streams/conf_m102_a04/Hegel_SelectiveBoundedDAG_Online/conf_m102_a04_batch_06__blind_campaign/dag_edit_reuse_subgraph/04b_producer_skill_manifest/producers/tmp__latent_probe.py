from dag_probe import *
P=s['panel_a']; z=mods[0]._align(P,s['train']['x']); c=z*np.roll(z,-1,axis=1); d=np.cos(2.2*z); p=s['panel_c']; u=p['u']; C=np.c_[np.ones(len(u)),u*np.roll(u,-1,axis=1)]; q=mods[0].least_squares(lambda q:(np.c_[np.ones(len(u)),np.cos(q*u)]@lst(np.c_[np.ones(len(u)),np.cos(q*u)],p['v2'])-p['v2']).ravel(),[2.2],bounds=(.5,4)).x
for name,A,B in [('pair',C,p['v1']),('cos',np.c_[np.ones(len(u)),np.cos(q*u)],p['v2']),('panel_b linear',np.c_[np.ones(24),s['panel_b']['u']],s['panel_b']['v'])]:
 print(name,'fit rmse',np.sqrt(np.mean((A@lst(A,B)-B)**2)), 'coef',lst(A,B).round(4))
print('q',q,'z max',np.max(abs(z),axis=0),'z correlations',np.corrcoef(np.c_[z,abs(z),c,T[:,0],T[:,-1]].T)[-2:].round(3))
for name,a in [('z',z),('abs',abs(z)),('pair',c),('z2',z*z),('raw',s['train']['x']),('all',np.c_[z,c,z*z]),('z_odd_even',np.c_[z,c,z*z,np.sin(2*z),np.cos(2*z)]),('signcanon',z*np.sign(z[:,0:1])),('W',W)]:
 F=np.c_[np.ones(n),a]; pred=F[52:]@lst(F[:52],T[:52]); er=abs(pred-T[52:]); print(name,np.mean(er[:,0]),np.mean(er[:,1:]),'train',np.mean(abs(F[:52]@lst(F[:52],T[:52])-T[:52]),axis=0)[[0,-1]])
print(np.c_[z[:30],T[:30,0],T[:30,-1]].round(3))
print('target svd',np.linalg.svd(T,compute_uv=False)); print('y~traj',np.mean(abs(T[:,1:]@lst(T[:,1:],T[:,0])-T[:,0])))
