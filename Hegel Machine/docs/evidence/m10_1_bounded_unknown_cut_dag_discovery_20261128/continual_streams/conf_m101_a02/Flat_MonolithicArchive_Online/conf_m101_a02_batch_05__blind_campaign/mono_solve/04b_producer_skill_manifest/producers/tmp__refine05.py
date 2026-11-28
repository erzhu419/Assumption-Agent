import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
from batch_solver import _affine as A, _fit as fit, _calibrate, _operator, _operator_design as D, _label_design as L
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['eval_times']; freq=s['telemetry_vec'][0]; train=s['train'];x=_calibrate(s['panel_a'],train['x']);y=train['y'];T=train['traj'];u,v=s['panel_b'].values();q,m=_operator(s['panel_b']); z=D(x,q)@m; d=L(z,freq);print('panel',q,m.round(5));print('full label coeff',fit(d,y).round(5));print('state coeff',fit(A(z),T).round(5));
for name, ii in [('52/20',np.arange(52)),('56/16',np.arange(56))]+[(f'random{j}',np.random.default_rng(j).permutation(72)[:52]) for j in range(6)]:
 jj=np.setdiff1d(np.arange(72),ii);print(name,end=' ')
 for mode in ['base','fixed','joint_rates','joint_mixer']:
  def evaluate(p,output=False):
   q0=p[:2];mp=fit(D(u,q0),v) if len(p)==2 else p[2:].reshape(5,4);f=D(x,q0);zz=f@mp; dd=L(zz,freq);tt=A(zz);w=fit(dd[ii],y[ii]);k=fit(tt[ii],T[ii]);yp=dd@w;tp=tt@k
   if output:return np.mean(abs(yp[jj]-y[jj])),np.mean(abs(tp[jj]-T[jj]))
   return np.r_[(D(u,q0)@mp-v).ravel()/.007,(yp[ii]-y[ii])/.014,(tp[ii]-T[ii]).ravel()/.006]
  if mode=='base':p=q
  elif mode=='fixed':p=np.array([1.,1.1])
  else:
   p0=q if mode=='joint_rates' else np.r_[q,m.ravel()];p=least_squares(evaluate,p0,max_nfev=100).x
  print(mode, np.round(evaluate(p,True),5),np.round(p[:2],4),end='; ')
 print()
