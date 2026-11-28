import copy, itertools, pickle, runpy
import numpy as np
from scipy.optimize import least_squares, minimize_scalar
p='/holdout_task/'
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']
base=runpy.run_path(p+'node_00.py'); frozen=[runpy.run_path(p+f'_frozen_nodes/node_0{i}.py')['run_node'] for i in range(1,4)]
u=s['panel_c']['u']; y=s['panel_c']['v2']; pairs=[(0,1),(1,2),(2,3),(3,0)]
def terms(x): return np.array([x[:,i]*x[:,j] for i,j in pairs]).T
families={'tanh':np.tanh,'atan':np.arctan,'rational':lambda z:z/(1+z*z),'softsign':lambda z:z/(1+abs(z))}
def fit(x,y,kind):
 g=families[kind]
 def f(z,q): return np.c_[np.ones(len(z)),z,terms(z),g(q*z)]
 def res(q):
  X=f(x,q[0]); return (X@np.linalg.lstsq(X,y,rcond=1e-10)[0]-y).ravel()
 q=least_squares(res,[1.1],bounds=(.15,4),gtol=1e-13).x[0]; X=f(x,q); c=np.linalg.lstsq(X,y,rcond=1e-10)[0]
 return q,c,lambda z:f(z,q)@c
for k in families:
 q,c,pred=fit(u,y,k); e=[]
 for i in range(len(u)):
  m=np.arange(len(u))!=i; e.extend((fit(u[m],y[m],k)[2](u[~m])-y[~m]).ravel())
 print('panel',k,'q',q,'fit',np.sqrt(np.mean((pred(u)-y)**2)),'loo',np.sqrt(np.mean(np.square(e))),'max',max(np.abs(e)),'linear/odd ratios',np.sum(c[1:5]*c[-4:],1)/np.sum(c[-4:]**2,1))
print('panel v1 RMSE',np.sqrt(np.mean((base['_panel_maps'](s['panel_c'],u)[:,:y.shape[1]]-s['panel_c']['v1'])**2)))

def candidate(st,kind):
 if kind=='baseline': return base['run_node']({'__input__':st})
 n=len(st['train']['x']); off,scale=base['_align'](st['panel_a']); z=(np.vstack((st['train']['x'],st['test']['x']))-off)/scale; panel=st['panel_c']; a=base['_panel_maps'](panel,z)[:,:y.shape[1]]; b=fit(panel['u'],panel['v2'],kind)[2](z); w=np.c_[a,b]; return {'features':(w-w[:n].mean(0))/np.maximum(w[:n].std(0),.1),'n_train':n}
T=s['train']; X=T['x']; off,scale=base['_align'](s['panel_a']); Z=(X-off)/scale
slices=[('public',np.arange(len(X))[-20:])]; slices += [(f'random{i}',np.random.default_rng(i).permutation(len(X))[:20]) for i in range(12)]; slices += [(f'axis{j}_{sg}',np.argsort(sg*Z[:,j])[-20:]) for j in range(Z.shape[1]) for sg in [-1,1]]; slices += [('radius',np.argsort(np.linalg.norm(Z,axis=1))[-20:])]
for kind in ['baseline','tanh','atan','rational']:
 out=[]
 for name,te in slices:
  tr=np.setdiff1d(np.arange(len(X)),te); st=copy.deepcopy(s); st['train']={k:T[k][tr] for k in ('x','y','traj')}; st['test']={'x':X[te]}; a=candidate(st,kind); b=frozen[0]({'__input__':st}); c=frozen[1]({'node_00':a,'node_01':b}); d=frozen[2]({'node_01':b,'node_02':c}); ey=np.abs(d['y_hat']-T['y'][te]); et=np.abs(d['state_pred']-T['traj'][te]); out.append((name,ey.mean(),et.mean(),et.mean(0).max()))
 print('dag',kind,'public',out[0],'worst-random',tuple(max(row[j] for row in out[:13]) for j in (1,2,3)),'worst-y',max(out,key=lambda r:r[1]),'worst-traj',max(out,key=lambda r:r[2]),'worst-horizon',max(out,key=lambda r:r[3]))
