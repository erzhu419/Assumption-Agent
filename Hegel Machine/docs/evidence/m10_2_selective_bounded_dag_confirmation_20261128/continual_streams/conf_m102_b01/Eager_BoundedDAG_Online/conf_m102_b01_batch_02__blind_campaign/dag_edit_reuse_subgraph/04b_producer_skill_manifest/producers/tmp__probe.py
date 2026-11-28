import pickle,numpy as np,runpy,itertools
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; w=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')['run_node']({'__input__':s})['w'][:72]; y=s['train']['y']; tr=s['train']['traj']; t=s['eval_times']; ar=runpy.run_path('/tmp/hegel_continual_archive/conf_m102_b01_batch_00_solver.py')
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
aff=lambda x:np.c_[np.ones(len(x)),x]
np.set_printoptions(precision=3,suppress=True); p=s['panel_c']; print('panel_c',np.c_[p['u'],p['v1'],p['v2']][:10]); print('panel_c ranges',[(k,z.min(0),z.max(0)) for k,z in p.items()]); print('panel correlations',np.corrcoef(np.c_[p['u'],p['v1'],p['v2']].T)[:4,4:]); print('b max',np.max(abs(s['panel_b']['u']),0));
_,_,vt=np.linalg.svd(tr,full_matrices=False); print('temporal modes',vt[:2]);
def tl(k):
 d=np.c_[-np.expm1(-k*t),t]; return np.mean((d@ls(d,tr.T)-tr.T)**2)
r=minimize_scalar(tl,bounds=(.05,6),method='bounded'); print('exp plus linear fit',r.x,r.fun**.5); print('rank2 noise',np.mean((tr-tr@vt[:2].T@vt[:2])**2)**.5)
for n in [48,54,57,58,60]:
 print('archive',n,*[np.mean(abs(ar['_predict'](w,z[:n],i,k)-z[n:])) for z,i,k in [(y,True,'asinh'),(y,True,'atan'),(tr,False,'inv3')]])
folds=np.array_split(np.random.default_rng(11).permutation(len(w)),6)
funcs={'id':lambda x:x,'asinh':np.arcsinh,'atan':np.arctan,'inv3':lambda x:np.sinh(np.arcsinh(x)/3),'tanh':np.tanh,'sin':np.sin,'rational':lambda x:x/np.sqrt(1+x*x),'cubic':lambda x:x**3}
for kind,f in funcs.items():
 for gram in ['add','square','pair','mixed','cubic']:
  scores=[]
  for g in ([1] if kind in ['id','cubic'] else [.75,1.25,2,3,4,6]):
   z=f(g*w); terms=[w,z];
   if gram in ('square','pair','mixed','cubic'): terms.extend([z[:,i:i+1]*z[:,j:j+1] for i,j in itertools.combinations_with_replacement(range(4),2) if gram!='square' or i==j]);
   if gram=='mixed': terms += [w[:,i:i+1]*z[:,j:j+1] for i in range(4) for j in range(4)];
   if gram=='cubic': terms += [np.prod(z[:,ix],axis=1)[:,None] for ix in itertools.combinations_with_replacement(range(4),3)];
   a=aff(np.column_stack(terms)); errors=[]
   for vi in folds:
    ti=np.setdiff1d(np.arange(len(w)),vi); pred=a[vi]@ls(a[ti],np.c_[y,tr][ti]); errors.append(np.mean(abs(pred-np.c_[y,tr][vi]),axis=0));
   e=np.mean(errors,axis=0); scores.append([g,e[0],np.mean(e[1:])]);
  print(kind,gram,'best y',min(scores,key=lambda v:v[1]),'best tr',min(scores,key=lambda v:v[2]))
