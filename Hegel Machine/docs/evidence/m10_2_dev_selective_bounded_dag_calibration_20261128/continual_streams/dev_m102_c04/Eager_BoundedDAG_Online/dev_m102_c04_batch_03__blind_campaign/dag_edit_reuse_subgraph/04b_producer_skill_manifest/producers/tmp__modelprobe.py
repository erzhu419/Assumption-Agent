import numpy as np,pickle,runpy
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; m=runpy.run_path('/tmp/hegel_continual_archive/dev_m102_c04_batch_00_solver.py'); np.set_printoptions(precision=5,suppress=True)
a,ls=m['_affine'],m['_ls']; ridge=m['_ridge']; A,B,C=(s['panel_'+k] for k in 'abc'); tr=s['train']; x=tr['x']; y=tr['y']; traj=tr['traj']; n=52
frozen=runpy.run_path('/holdout_task/_frozen_nodes/node_02.py')['run_node']; temporal=runpy.run_path('/holdout_task/_frozen_nodes/node_01.py')['run_node']({'__input__':{'eval_times':s['eval_times'],'train':{'traj':traj[:n]}}});
o,M=m['_calibration'](A); bo,*bm=ls(a(B['u']),B['v']); bm=np.array(bm); print('A',o,M,'err',np.sort(np.linalg.norm(A['u']@M+o-A['v'],axis=1))); print('B sv',np.linalg.svd(bm)[1]); q=np.linalg.solve(M.T,(x-o).T).T
funcs={'satabs':lambda x,k:x/(1+k*abs(x)), 'satsq':lambda x,k:x*x/(1+k*x*x), 'ratodd':lambda x,k:x/(1+k*x*x), 'tanh':lambda x,k:np.tanh(k*x),'sin':lambda x,k:np.sin(k*x),'atan':lambda x,k:np.arctan(k*x),'square':lambda x,k:x*x,'abs':lambda x,k:abs(x)}
ops=[]
for j in [1,2]:
 u=C['u'];v=C['v'+str(j)]; fits=[]
 for name,g in funcs.items():
  for aff in [False,True]:
   def d(z,k):return np.c_[a(z) if aff else np.ones(len(z)),g(z,k)]
   def e(k):
    f=d(u,k[0]);return (f@ls(f,v)-v).ravel()
   k=least_squares(e,[.4],bounds=(.01,5),ftol=1e-11,gtol=1e-11).x[0]; w=ls(d(u,k),v); fits.append((np.sqrt(np.mean(e([k])**2)),name,aff,k,w))
 fits.sort(key=lambda z:z[0]); print('C',j,[(r,name,aff,k) for r,name,aff,k,w in fits[:12]]); ops.append(fits)
pop,rop=m['_panel_operators'](C)
def metric(common,offdiag):
 yy=ridge(common[:n],y[:n],common[n:],1e-6); out=frozen({'node_00':dict(common=common,offdiag=offdiag,y_hat=yy),'node_01':temporal});return np.mean(abs(yy-y[n:])),np.mean(abs(out['state_pred']-traj[n:]))
coords={'x':x,'q':q,'qB':q@bm+bo,'qBinv':np.linalg.solve(bm.T,(q-bo).T).T,'xB':x@bm+bo,'xBinv':np.linalg.solve(bm.T,(x-bo).T).T}
for pn,pq in coords.items():
 for rn,rq in coords.items():
  print(pn,rn,*np.round(metric(*m['_features'](pop(pq),rop(rq))),5))
print('poly baselines')
for name,z in coords.items():
 i,j=np.triu_indices(4); c=np.c_[z,z[:,i]*z[:,j]]; d=c[:,4:]; print(name,metric(c,d))
