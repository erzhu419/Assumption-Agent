import runpy,pickle,numpy as np
from itertools import combinations
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a=runpy.run_path('/tmp/hegel_continual_archive/dev_m102_c04_batch_00_solver.py'); ls=a['_ls']; aff=a['_affine']; np.set_printoptions(precision=5,suppress=True,linewidth=140)
for k in ['panel_a','panel_b']:
 p=s[k]; u,v=p['u'],p['v']; w=ls(aff(u),v); e=np.linalg.norm(aff(u)@w-v,axis=1); b,m=a['_calibration'](p); print(k,'OLS\n',w,'errors',np.quantile(e,[0,.25,.5,.75,.9,1]),'robust\n',np.vstack([b,m]),'err',np.sort(np.linalg.norm(u@m+b-v,axis=1)))
pc=s['panel_c']; p,r=a['_panel_operators'](pc)
for j,f in enumerate([p,r],1):
 e=f(pc['u'])-pc[f'v{j}']; print('operator',j,'fit error',np.linalg.norm(e),np.max(abs(e)))
x=s['train']['x']; y=np.column_stack([s['train']['y'],s['train']['traj']]); n=52
trans={'x':x}
for k in ('panel_a','panel_b'):
 b,m=a['_calibration'](s[k]);
 for name,q in [(k+' inv',np.linalg.solve(m.T,(x-b).T).T),(k+' fwd',x@m+b)]:trans[name]=q
b,m=a['_calibration'](s['panel_a']); c,d=a['_calibration'](s['panel_b']); q=trans['panel_a inv']; trans['Ainv Bfwd']=q@d+c; trans['Ainv Binv']=np.linalg.solve(d.T,(q-c).T).T; trans['Binv Ainv']=np.linalg.solve(m.T,(trans['panel_b inv']-b).T).T
for k,q in trans.items():
 P,R=p(q),r(q); core,off=a['_features'](P,R); lscores=[]
 for f in [aff(q),aff(core),aff(np.column_stack([P,R,(P[:,:,None]*R[:,None,:]).reshape(len(q),-1)])),aff(np.column_stack([core,off,q]))]:
  w=ls(f[:n],y[:n]); e=abs(f[n:]@w-y[n:]); z=abs(f[:n]@w-y[:n]); lscores.append([z[:,0].mean(),z[:,1:].mean(),e[:,0].mean(),e[:,1:].mean()]);
 print(k,np.array(lscores))
print('traj singular',np.linalg.svd(y[:,1:],compute_uv=False))
