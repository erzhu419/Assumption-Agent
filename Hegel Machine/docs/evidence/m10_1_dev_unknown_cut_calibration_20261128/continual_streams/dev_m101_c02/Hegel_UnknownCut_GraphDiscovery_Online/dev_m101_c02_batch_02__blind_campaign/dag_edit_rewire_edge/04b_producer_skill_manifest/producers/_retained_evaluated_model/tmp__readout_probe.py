import numpy as np,pickle,runpy
from itertools import combinations_with_replacement
p='/holdout_task/'
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']; a=runpy.run_path(p+'_frozen_nodes/node_00.py')['run_node']({'__input__':s}); b=runpy.run_path(p+'_frozen_nodes/node_01.py')['run_node']({'node_00':a}); u,z=a['u_tr'],b['z_tr']; y,t=s['train']['y'],s['train']['traj']; U,V=s['panel_b'].values(); B=np.linalg.lstsq(np.c_[np.ones(len(U)),U],V,rcond=None)[0]; C=s['panel_c']; print('panel B affine residual',np.max(abs(np.c_[np.ones(len(U)),U]@B-V))); print('panel C vs B',np.mean(abs(np.c_[np.ones(len(C['u'])),C['u']]@B-C['v1']),axis=0)); print('B',B); print('trajectory singular values',np.linalg.svd(t,compute_uv=False)); print('target scales',np.std(y),np.std(t,axis=0));
x=np.c_[u,z]; names=['u'+str(i) for i in range(4)]+['z'+str(i) for i in range(4)]; idx=np.split(np.random.default_rng(982).permutation(len(y)),6); idx.append(np.arange(52,72));
def fit(X,Y):
 scores=[]
 for te in idx:
  tr=np.setdiff1d(np.arange(len(y)),te); scores.append(np.mean(abs(X[te]@np.linalg.lstsq(X[tr],Y[tr],rcond=1e-9)[0]-Y[te])))
 return np.round([np.mean(scores[:-1]), max(scores[:-1]),scores[-1]],5)
def report(name,X):
 X=np.c_[np.ones(len(y)),X]; print(name,'rank',np.linalg.matrix_rank(X), 'y / traj',fit(X,y),fit(X,t))
poly=lambda x,d: np.column_stack([np.prod(x[:,ij],axis=1) for ij in combinations_with_replacement(range(x.shape[1]),d)])
for k in range(1,5): report('u poly '+str(k),np.c_[tuple(poly(u,j) for j in range(1,k+1))])
funs={'sin':np.sin,'tanh':np.tanh,'softsign':lambda x:x/(1+abs(x)), 'rat':lambda x:x/(1+x*x),'abs':abs,'cos':np.cos}
P=np.c_[u,poly(u,2)];
for name,f in funs.items():
 report(name+' + linear',np.c_[u,f(x)]); report(name+' + quad',np.c_[P,f(x)]); report(name+' + quad & interaction',np.c_[P,f(x),(x[:,:,None]*f(x)[:,None,:]).reshape(len(y),-1)])
for i in range(4): print('z%d paired'%i,fit(np.c_[np.ones(len(y)),P,u*z[:,i,None]],y));
for j in range(8): print('t',j,fit(np.c_[np.ones(len(y)),P],t[:,j]),'scale relation',np.std(t[:,j]-t[:,-1]*np.dot(t[:,j],t[:,-1])/np.dot(t[:,-1],t[:,-1])))
