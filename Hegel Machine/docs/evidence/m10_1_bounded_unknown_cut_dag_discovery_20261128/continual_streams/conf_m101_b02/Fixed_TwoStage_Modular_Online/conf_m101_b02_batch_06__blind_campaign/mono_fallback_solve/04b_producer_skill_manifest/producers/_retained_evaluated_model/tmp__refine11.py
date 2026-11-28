import numpy as np, pickle, runpy
from scipy.optimize import least_squares
m=runpy.run_path('/holdout_task/batch_solver.py'); s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; x=t['x']; y=np.c_[t['y'],t['traj']]; read=m['_readout']; base=list(m['_features'](s,x)); a=base[0][:,:4]; u,v=s['panel_b']['u'],s['panel_b']['v']; ls=m['_ls']; pc=s['panel_c']; cp=m['_pairs']; coeff=ls(cp(pc['u']),pc['v1']); block=lambda a,c:np.c_[a,c,(a[:,:,None]*c[:,None,:]).reshape(len(a),16)]
funcs={'poly5':lambda z,p:z*(p[0]+z*z*(1+p[1]*z*z)), 'poly7':lambda z,p:z*(p[0]+z*z*(1+z*z*(p[1]+p[2]*z*z))), 'rational':lambda z,p:z*(p[0]+z*z)/(1+p[1]*z*z), 'rational2':lambda z,p:z*(p[0]+z*z*(1+p[2]*z*z))/(1+p[1]*z*z), 'cubic':lambda z,p:z*(p[0]+z*z)}
folds=[(np.arange(52),np.arange(52,72)),(np.arange(56),np.arange(56,72))];
for seed in (35,12,81):
 for te in np.array_split(np.random.default_rng(seed).permutation(52),4):folds.append((np.setdiff1d(np.arange(52),te),te))
def evaluate(name,fs,r):
 out=[]
 for tr,te in folds:
  z=np.r_[tr,te]; pred=np.mean([read(f[z],y[tr],len(tr),r) for f in fs],axis=0); e=np.abs(pred-y[te]);out.append([e[:,0].mean(),e[:,1:].mean()])
 o=np.array(out); print(name,r,'52/20 56/16',o[:2].round(5).tolist(),'inner mean/worst',o[2:].mean(0).round(5),o[2:].max(0).round(5))
for r in (.001,.003,.01,.03):evaluate('baseline',base,r)
for name,f in funcs.items():
 n={'cubic':1,'poly7':3,'rational2':3}.get(name,2); init=np.array([-.12, .1, 0.][:n]);
 def design(z,p):return np.c_[np.ones(len(z)),f(z,p)]
 def resid(p):
  z=design(u,p);return (z@ls(z,v)-v).ravel()
 lim=([-3,0,-5][:n],[3,6,5][:n]) if 'rational' in name else (-5,5)
 p=least_squares(resid,init,bounds=lim).x; z=design(u,p); loo=resid(p).reshape(v.shape)/(1-np.sum(z*np.linalg.pinv(z).T,axis=1))[:,None]; print('panel B',name,p,'LOO RMSE', np.std(loo)); c=cp(design(a,p)@ls(z,v))@coeff; candidate=block(a,c)
 for r in (.001,.01):evaluate(name,[candidate],r)
