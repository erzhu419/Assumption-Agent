import numpy as np,pickle,itertools,runpy
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
f=runpy.run_path('/holdout_task/node_00.py'); tr=s['train']; n=len(tr['y']); x=tr['x']; o,a=f['_calibrate'](s['panel_a']); z=np.linalg.solve(a.T,(x-o).T).T
D=f['run_node']({'__input__':s}); b=D['fd'][:n]; c,d=np.hsplit(D['cd'][:n],2)
Y=np.c_[tr['y'],tr['traj']]; rng=np.random.default_rng(38); folds=[np.arange(n-20,n),*np.array_split(rng.permutation(n),5)];
for name, blocks in [('z',[z]),('all panels',[z,b,c,d]),('panels + additive',[z,b,c,d,z*z,np.sin(z),np.cos(z)]),('panels + pair',[z,b,c,d,*[z[:,i]*z[:,j] for i,j in itertools.combinations(range(4),2)]]),('panels + mixed trig',[z,b,c,d,*[F(z[:,i]+sg*z[:,j]) for i,j in itertools.combinations(range(4),2) for sg in [-1,1] for F in [np.sin,np.cos]]])]:
 G=np.column_stack([np.ones(n),*blocks]); E=[]
 for val in folds:
  ix=np.setdiff1d(np.arange(n),val); P=np.eye(G.shape[1])*.0003;P[0,0]=0
  e=abs(Y[val]-G[val]@np.linalg.solve(G[ix].T@G[ix]+P,G[ix].T@Y[ix]));E.append((e[:,0].mean(),e[:,1:].mean()))
 print(name,'official split',np.round(E[0],5),'worst',np.round(np.max(E,axis=0),5))
for pk,vk in [('panel_b','v'),('panel_c','v1')]:
 p=s[pk];u,v=p['u'],p[vk]; lin=np.c_[np.ones(len(u)),u];
 for label,fn,k in [('sin',np.sin,2.815),('tanh',np.tanh,2.25 if vk=='v1' else 1.38)]:
  g=np.c_[lin,fn(k*u)]; coef=np.linalg.lstsq(g,v,rcond=None)[0]; E=v-g@coef;print(pk,label,'coef',np.round(coef[5:],3),'error singulars',np.round(np.linalg.svd(E,compute_uv=False),5))
