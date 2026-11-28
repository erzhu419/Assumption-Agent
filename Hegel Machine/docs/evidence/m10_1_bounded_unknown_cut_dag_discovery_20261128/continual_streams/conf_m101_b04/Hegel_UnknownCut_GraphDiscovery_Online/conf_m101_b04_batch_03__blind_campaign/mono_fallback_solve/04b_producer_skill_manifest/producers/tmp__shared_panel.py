import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
spec=importlib.util.spec_from_file_location('m','/holdout_task/batch_solver.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];x=s['train']['x'];y=s['train']['y'];Y=s['train']['traj'];o,g=m._calibration(s['panel_a']);w=(x-o)/g;u,v=(s['panel_b'][k] for k in ('u','v'))
funcs={'sin':np.sin,'tanh':np.tanh,'rational':lambda t:t/(1+t*t),'atan':np.arctan,'erf':__import__('scipy.special',fromlist=['erf']).erf,'cube':lambda t:t**3,'exp':lambda t:t*np.exp(-t*t),'softsign':lambda t:t/np.sqrt(1+t*t)}
all_i=np.arange(len(x)); folds=[all_i[52:]]+[np.random.default_rng(k).permutation(len(x))[-20:] for k in range(8)]+[np.argsort(sign*x[:,k])[-16:] for k in range(4) for sign in (1,-1)]+[np.argsort(np.linalg.norm(x-x.mean(0),axis=1))[-16:]]
def fit(f):
 def d(a,p):return m._affine(a+p[0]*f(p[1]*a))
 def residual(p):
  a=d(u,p);return (a@np.linalg.lstsq(a,v,rcond=None)[0]-v).ravel()
 bounds=([-5,.15],[5,5]);opts=[least_squares(residual,[a,b],bounds=bounds,max_nfev=300) for a in [-.75,.5] for b in [.65,1.5,2.8,4.3]];p=min(opts,key=lambda fit:fit.cost).x
 a=d(u,p);coef=np.linalg.lstsq(a,v,rcond=None)[0];q=np.linalg.qr(a)[0];e=v-a@coef
 return p, np.sqrt(np.mean(e**2)),np.sqrt(np.mean((e/(1-(q*q).sum(1))[:,None])**2)),d(w,p)@coef
ops={'baseline':m._operator(s['panel_b'])(w)}
for name,f in funcs.items():
 p,rmse,loo,ops[name]=fit(f);print(name,p.round(5),'panel RMSE/conditional LOO',round(rmse,7),round(loo,7))
for name,z in ops.items():
 a=m._affine(z);q=m._quadratic(z); pairs=list(combinations_with_replacement(range(4),2)); print(name,'scalar coefficients',np.linalg.lstsq(q,y,rcond=None)[0].round(3));
 for tag,d,p in [('linear',a,.001),('quadratic',q,.001),('ridge_quad',q,.01),('squares',np.c_[a,z*z],.001),('pairs',np.c_[a,q[:,[5+k for k,(i,j) in enumerate(pairs) if i!=j]]],.001)]:
  ee=[]
  for j in folds:
   i=np.setdiff1d(all_i,j)
   pred=d[j]@m._ridge(d[i],np.c_[y,Y][i],p); ee.append(np.mean(abs(pred-np.c_[y,Y][j]),axis=0))
  e=np.asarray(ee); print(tag,'public',e[0,[0,8]].round(5),'random',e[1:9,[0,8]].mean(0).round(5),'stress',e[9:,[0,8]].max(0).round(5),'scalar worst fold',e[:,0].argmax())
