import pickle,runpy,numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; n=runpy.run_path('/holdout_task/node_00.py'); aff=n['_affine']; ls=n['_ls']; t=s['train']; o,w=n['_calibration'](s['panel_a']); q=np.linalg.solve(w.T,(t['x']-o).T).T; Y=np.c_[t['y'],t['traj']]; c=s['panel_c']; U,V=c['u'],c['v1'];
def poly(x,d):return np.column_stack([np.prod(x[:,i],axis=1) for j in range(d+1) for i in combinations_with_replacement(range(x.shape[1]),j)])
def diag(x):return np.c_[np.ones(len(x)),x]
for k in ['panel_b','panel_c']:
 u=s[k]['u']; print(k)
 for key,v in s[k].items():
  if key=='u':continue
  print(key,'aff coefs',np.round(ls(aff(u),v),3),'std',v.std(axis=0))
  for d in [1,2,3]:
   a=poly(u,d); h=np.sum(a*np.linalg.pinv(a).T,axis=1); print('poly',d,np.sqrt(np.mean((a@ls(a,v)-v)**2)),np.sqrt(np.mean(((a@ls(a,v)-v)/np.maximum(1-h[:,None],1e-12))**2)))
  print('row fit residual',np.round(np.linalg.norm(aff(u)@ls(aff(u),v)-v,axis=1),3))
F={'absrat':lambda x,k:x/(1+k*np.abs(x)), 'sqrat':lambda x,k:x/(1+k*x*x),'evenrat':lambda x,k:x*x/(1+k*x*x),'tanh':lambda x,k:np.tanh(k*x), 'sin':lambda x,k:np.sin(k*x),'cos':lambda x,k:np.cos(k*x),'cube':lambda x,k:x**3,'square':lambda x,k:x*x,'abssq':lambda x,k:x*np.abs(x),'abs':lambda x,k:np.abs(x),'exp':lambda x,k:np.exp(np.clip(k*x,-30,30))}
P,R=n['_panel_operators'](c)
def downstream(p,r):
 cross=(p[:,:,None]*r[:,None,:]).reshape(len(q),-1); common=n['_features'](p,r)[0]; out=[]
 for a in [aff(common),aff(np.c_[common,cross]),aff(np.c_[q,p,r,cross])]:
  e=np.abs(a@ls(a[:52],Y[:52])-Y);out.append(np.round([e[52:,0].mean(),e[52:,1:].mean()],5).tolist())
 return out
for name,fn in F.items():
 def design(x,k):return np.c_[aff(x),fn(x,k)]
 def err(z):
  a=design(U,z[0]);return (a@ls(a,V)-V).ravel()
 opt=min([least_squares(err,[i],bounds=(.001,25)) for i in [.1,1.,3.,6.,10.,18.]],key=lambda j:j.cost); k=opt.x[0]; w2=ls(design(U,k),V); print(name,round(k,4),round(np.sqrt(2*opt.cost/V.size),6),downstream(design(q,k)@w2,R(q)))
# Test whether the nonlinear directions are affine-panel B projections.
b=s['panel_b']; wb=ls(aff(b['u']),b['v']); print('B self y/traj',downstream(aff(q)@wb,R(q))); print('q poly CV')
for d in [1,2,3,4]:
 a=poly(q,d); e=np.abs(a@ls(a[:52],Y[:52])-Y); print(d,e[52:,0].mean(),e[52:,1:].mean())
