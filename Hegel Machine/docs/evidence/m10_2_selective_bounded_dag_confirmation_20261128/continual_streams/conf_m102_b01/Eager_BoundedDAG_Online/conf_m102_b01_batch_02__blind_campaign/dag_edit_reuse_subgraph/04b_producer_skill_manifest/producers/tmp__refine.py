import pickle,numpy as np,runpy,itertools
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; w=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')['run_node']({'__input__':s})['w'][:72]; y=s['train']['y']; tr=s['train']['traj']; t=s['eval_times']; ar=runpy.run_path('/tmp/hegel_continual_archive/conf_m102_b01_batch_00_solver.py')
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
aff=lambda x:np.c_[np.ones(len(x)),x]
np.set_printoptions(precision=3,suppress=True); p=s['panel_c']; print('panel_c',np.c_[p['u'],p['v1'],p['v2']][:10]); print('panel_c ranges',[(k,z.min(0),z.max(0)) for k,z in p.items()]); print('panel correlations',np.corrcoef(np.c_[p['u'],p['v1'],p['v2']].T)[:4,4:]); print('b max',np.max(abs(s['panel_b']['u']),0));
print('panel c sample',np.round(np.c_[s['panel_c']['u'],s['panel_c']['v1'],s['panel_c']['v2']][:5],3)); print('archive 54',*[np.mean(abs(ar['_predict'](w,z[:54],i,k)-z[54:])) for z,i,k in [(y,True,'asinh'),(y,True,'atan'),(tr,False,'inv3')]])
T=np.c_[y,tr]; D=np.arange(len(w)); folds=list(np.array_split(np.random.default_rng(11).permutation(len(w)),6))+[D[54:],D[57:],D[60:]]
_,_,vt=np.linalg.svd(tr,full_matrices=False)
r=minimize_scalar(lambda k: np.mean((np.c_[-np.expm1(-k*t),t]@ls(np.c_[-np.expm1(-k*t),t],tr.T)-tr.T)**2),bounds=(.05,6),method='bounded'); print('exp+linear',r.x,r.fun**.5,'svd',np.linalg.svd(tr,compute_uv=False))
qs=list(itertools.combinations_with_replacement(range(4),2)); ps=[(i,j) for i,j in qs if i<j]; quart=list(itertools.combinations_with_replacement(range(4),4)); cubic=list(itertools.combinations_with_replacement(range(4),3));
def terms(x,ix): return np.column_stack([np.prod(x[:,ii],axis=1) for ii in ix])
def report(name,a):
 er=[]
 for vi in folds:
  ti=np.setdiff1d(D,vi); r=np.abs(a[vi]@ls(a[ti],T[ti])-T[vi]); er.append([r[:,0].mean(),r[:,1:].mean()]);
 print(name,'6fold avg',np.round(np.mean(er[:6],0),5),'worst',np.round(np.max(er[:6],0),5),'tails',np.round(er[6:],5).tolist(),'fit',np.round(np.mean(abs(a@ls(a,T)-T),axis=0)[[0,-1]],5))
base=np.c_[aff(w),terms(w,qs)]; report('quad',base); report('cross only',np.c_[aff(w),terms(w,ps)]); report('quad+cubic',np.c_[base,terms(w,cubic)]); report('quad+quartic',np.c_[base,terms(w,quart)]); print('quad coef: intercept, linear, squares and cross',np.round(ls(base,np.c_[y,tr@vt[:2].T]),4));
for j in [.1,.35,.5,.75,1.,1.25]:
 for nm,fn in [('asinh',np.arcsinh),('atan',np.arctan),('inv3',lambda u:np.sinh(np.arcsinh(u)/3))]:
  z=fn(j*w)/j; rem=z-w if nm!='inv3' else 3*z-w
  report((nm,j,'cross only'),np.c_[base,rem,*[(w[:,i]*rem[:,k]) for i,k in itertools.permutations(range(4),2)]]);
  report((nm,j,'full mixed'),np.c_[aff(w),z,terms(z,qs),*(w[:,i]*z[:,k] for i,k in itertools.product(range(4),repeat=2))]);
