import sys,pickle,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
from batch_solver import _affine as A,_fit as fit,_calibrate,_operator,_operator_design as D,_label_design as L
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train'];t=s['eval_times'];f=s['telemetry_vec'][0];q,m=_operator(s['panel_b']); z=D(_calibrate(s['panel_a'],tr['x']),q)@m;d=L(z,f);a=A(z);y=tr['y'];T=tr['traj'];c=z@np.array([1,1,-1,-1]);print('times and telemetry',t,s['telemetry_vec']);print('largest full fit scalar residuals',sorted(zip(abs(d@fit(d,y)-y),range(len(y))),reverse=True)[:8]);
variants={'base':d,'contrast':np.c_[d[:,0],c,d[:,5:]],'no_tanh':np.delete(d,7,axis=1),'no_sin':np.delete(d,6,axis=1)}
for seed in [-2,-1,*range(12)]:
 i=np.arange(52 if seed==-2 else 56) if seed<0 else np.random.default_rng(seed).permutation(72)[:52];j=np.setdiff1d(np.arange(72),i)
 label=[np.mean(abs(b[j]@fit(b[i],y[i])-y[j])) for b in variants.values()];coef=fit(a[i],T[i]);pred=a[j]@coef
 def prof(p):
  b=-np.expm1(-np.exp(p[:,None])*t);return (fit(b.T,coef.T).T@b)
 rates=least_squares(lambda p:(a[i]@(prof(p)-coef)).ravel(),np.log([.4,1.8]),bounds=(-4,3)).x
 pred2=a[j]@prof(rates);err=np.mean(abs(pred-T[j]),axis=0);err2=np.mean(abs(pred2-T[j]),axis=0)
 print(seed,np.round(label,5),'state mean',np.round([err.mean(),err2.mean()],5),'worst time',np.round([err.max(),err2.max()],5),'final',np.round([err[-1],err2[-1]],5),'rates',np.round(np.exp(rates),3))
