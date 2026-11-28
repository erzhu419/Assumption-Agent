import numpy as np, pickle, runpy
from scipy.optimize import least_squares
p='/holdout_task/'
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']; a=s['train']; c=s['panel_c']; u,v=c['u'],c['v1']; n=len(a['x']); idx=np.arange(n)
m=runpy.run_path(p+'node_00.py'); aff=m['_affine_design']; x=m['_calibrate'](s['panel_a'],a['x']); b=s['panel_b']; z=aff(x)@np.linalg.lstsq(aff(b['u']),b['v'],rcond=None)[0]
random=[te for seed in range(8) for te in np.array_split(np.random.default_rng(seed).permutation(n),4)]; tails=[np.argsort(sign*a['x'][:,j])[-12:] for j in range(4) for sign in (1,-1)]; rad=[np.argsort(np.linalg.norm(a['x'],axis=1))[-14:]]; target=np.c_[a['y'],a['traj']]
def design(q,params,typ):
 k=np.exp(params[:4]) if len(params)>=4 else np.exp(params[0]); t=q-(params[4:] if len(params)==8 else 0)
 h=t**3/(1+k*t*t) if typ=='rational' else np.tanh(k*t)
 return np.c_[np.ones(len(q)),q,h]
for typ in ['rational','tanh']:
 for d in [1,4,8]:
  def err(params,J):
   M=design(u,params,typ); A=np.linalg.lstsq(M[J],v[J],rcond=None)[0]; return M@A-v
  bounds=([-3]*min(d,4)+[-.6]*max(0,d-4),[3]*min(d,4)+[.6]*max(0,d-4)); init=np.zeros(d)
  def fit(J): return least_squares(lambda par:err(par,J)[J].ravel(),init,bounds=bounds,max_nfev=150).x
  J=np.arange(len(u)); P=fit(J); loo=np.sqrt(np.mean([err(fit(J[J!=i]),J[J!=i])[i]**2 for i in J])); M=design(u,P,typ); w=design(z,P,typ)@np.linalg.lstsq(M,v,rcond=None)[0]; D=np.c_[aff(x),w,(x[:,:,None]*w[:,None,:]).reshape(n,-1)]
  def score(te):
   tr=np.setdiff1d(idx,te); e=D[te]@np.linalg.lstsq(D[tr],target[tr],rcond=None)[0]; modes=np.linalg.svd(target[tr,1:],full_matrices=False)[2][:2]; e[:,1:]=e[:,1:]@modes.T@modes; e-=target[te];return [abs(e[:,0]).mean(),abs(e[:,1:]).mean()]
  print(typ,d,'params',np.round(P,4),'panel LOO',round(loo,6),'public',np.round(score(np.arange(52,72)),6),'worst', [np.round(np.max([score(te) for te in t],axis=0),6).tolist() for t in [random,rad,tails]])
