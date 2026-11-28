import sys,pickle,numpy as np
from scipy.optimize import least_squares, minimize_scalar
sys.path.insert(0,'/tmp/hegel_continual_archive')
import conf_m102_b03_batch_11_solver as b, conf_m102_b03_batch_00_solver as a
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; ls=b._ols; D=b._design
z=b._calibration(s['panel_a'])(tr['x']); n=52; u=s['panel_c']['u']; vs=[s['panel_c'][k] for k in ['v1','v2']]; c,d=[f(z) for f in b._branches(s['panel_c'])]; ca,da=[a._branch(s['panel_c'],z,k) for k in (1,2)]
def score(x,y):
 e=x[n:]@ls(x[:n],y[:n])-y[n:];return np.mean(abs(e))
for name, aa,bb in [('archive11',c,d),('archive00',ca,da)]:
 ff=D(np.c_[aa,bb,np.sum(aa*bb,axis=1)]); gg=D(np.c_[aa,bb,(aa[:,:,None]*bb[:,None,:]).reshape(len(z),16)]); print(name,score(ff,tr['y']),score(gg,tr['traj']), 'panel RMSE', [(np.mean((f(u)-v)**2))**.5 for f,v in zip(b._branches(s['panel_c']),vs)] if name=='archive11' else [np.mean((a._branch(s['panel_c'],u,k)-v)**2)**.5 for k,v in enumerate(vs,1)])
from itertools import combinations_with_replacement
for deg in [1,2,3]:
 X=D(np.column_stack([np.prod(z[:,co],axis=1) for k in range(1,deg+1) for co in combinations_with_replacement(range(4),k)])); print('poly',deg,score(X,tr['y']),score(X,tr['traj']))
trans={'soft':b._soft,'tanh':np.tanh,'sin':np.sin,'cos':np.cos,'lorentz':lambda x:1/(1+x*x),'atan':np.arctan,'sqrt':lambda x:np.sqrt(1+x*x),'exp':lambda x:np.exp(-x*x),'rational':lambda x:x/(1+x*x),'softabs':lambda x:abs(x)/(1+abs(x))}
for v in vs:
 out=[]
 for name,fn in trans.items():
  for phase in [False,True]:
   def obj(p):
    f=D(np.c_[u,fn(p[0]*u+(p[1:] if phase else 0))]);return (f@ls(f,v)-v).ravel()
   sols=[least_squares(obj,[f0]+([p0]*4 if phase else []),bounds=([.04]+([-6]*4 if phase else []),[8]+([6]*4 if phase else [])),max_nfev=120) for f0 in [.5,1.25,2.15,4.] for p0 in ([0.,.7] if phase else [0.])]; best=min(sols,key=lambda r:r.cost);out.append((np.sqrt(np.mean(best.fun**2)),name,phase,np.round(best.x,3)))
 print('branch fits:',sorted(out,key=lambda r:r[0])[:10])
print('panel b coefficients',ls(D(s['panel_b']['u']),s['panel_b']['v']))
