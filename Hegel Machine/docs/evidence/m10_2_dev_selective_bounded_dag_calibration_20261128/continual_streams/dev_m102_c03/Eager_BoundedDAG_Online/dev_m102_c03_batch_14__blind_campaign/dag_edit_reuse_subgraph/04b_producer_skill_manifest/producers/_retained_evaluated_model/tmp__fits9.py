import pickle,runpy,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; m=runpy.run_path('/holdout_task/node_01.py'); t=s['train']; bias=m['_bias']; ls=m['_ls']; z=m['_coordinates'](s,t['x']); a,b=m['_panel'](s,z); p=s['panel_b']; w=bias(z)@ls(bias(p['u']),p['v']); Y=np.c_[t['y'],t['traj']]; c=s['panel_c']; u,v=c['u'],c['v2']; h=m['_second'];
def er(p):
 f=h(u,p);return (f@ls(f,v)-v).ravel()
r=least_squares(er,[1.2,.45,.05]);print('second',r.x,np.std(r.fun));
for base in [np.c_[a,b],np.c_[a,b,w],np.c_[a,b,z],np.c_[a,b,w,z]]:
 for feat in [np.sum(a*b,1)[:,None],np.c_[np.sum(a*b,1),a[:,0]*b[:,1]],a*b,(a[:,:,None]*b[:,None,:]).reshape(-1,16),np.c_[(a[:,:,None]*b[:,None,:]).reshape(-1,16),a*a,b*b]]:
  f=bias(np.c_[base,feat]); print(base.shape[1],feat.shape[1],np.abs(f[52:]@ls(f[:52],Y[:52])-Y[52:]).mean(0).round(4),np.abs(f@ls(f,Y)-Y).mean(0)[[0,-1]].round(4))
