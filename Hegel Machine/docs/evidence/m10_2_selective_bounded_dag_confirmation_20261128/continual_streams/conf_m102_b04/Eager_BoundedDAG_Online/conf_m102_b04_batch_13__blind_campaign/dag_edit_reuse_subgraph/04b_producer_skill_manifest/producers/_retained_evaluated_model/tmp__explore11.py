import pickle,numpy as np
from scipy.optimize import minimize_scalar,least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=3,suppress=True,linewidth=130)
def aff(u):return np.c_[np.ones(len(u)),u]
def proj(f,v):return np.linalg.lstsq(f,v,rcond=None)[0]
for p,key in [('panel_a','v'),('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u,v=s[p]['u'],s[p][key]; keep=np.linalg.norm(v-aff(u)@proj(aff(u),v),axis=1)<4; u,v=u[keep],v[keep]; print(p,key)
 opts=[]
 for mode in ('s','c','sc','sq_s','sq_c','sq_sc','cub_sc','sq_t','sq_rat','sq_tan'):
  def feat(w):
   f=[aff(u)]; h=mode.split('_')[-1]
   if '_' in mode:f+=[u*u]
   if mode.startswith('cub'):f+=[u**3]
   if h in ('s','sc'):f+=[np.sin(w*u)]
   if h in ('c','sc'):f+=[np.cos(w*u)-1]
   if h=='t':f+=[np.tanh(w*u)]
   if h=='rat':f+=[u/np.sqrt(1+w*u*u)]
   if h=='tan':f+=[np.arctan(w*u)]
   return np.column_stack(f)
  def score(w):
   f=feat(w); q=np.linalg.qr(f)[0]; return np.mean(abs((v-q@(q.T@v))/np.maximum(1e-8,1-np.sum(q*q,axis=1)[:,None])))
  grid=np.linspace(.2,7,140); j=np.argmin([score(w) for w in grid]); r=minimize_scalar(score,bounds=(grid[max(0,j-1)],grid[min(len(grid)-1,j+1)]),method='bounded'); opts.append((r.fun,mode,r.x,proj(feat(r.x),v)))
 for err,mode,w,b in sorted(opts)[:4]:print(mode,round(w,4),round(err,5),'coeff',b if mode==sorted(opts)[0][1] else '')
print('scalar vs traj corr',np.corrcoef(np.c_[s['train']['y'],s['train']['traj']].T)[0]); print('trajectory modes',np.linalg.svd(s['train']['traj'],full_matrices=False)[2][:2]); print('telemetry',s['telemetry_vec'])
