import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task');from batch_solver import *
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; h=s['train']; times=s['eval_times']; n=52; z=cal(s['panel_a'],h['x']); b,p,q=ops(s,z); Y=np.c_[h['y'],h['traj']];
def er(pr):return np.mean(abs(pr[52:]-Y[52:]),axis=0)[[0,1,8]],np.mean(abs(pr[52:,1:]-Y[52:,1:]))
ss=dict(s,train={k:v[:52] if k!='summary' else v for k,v in h.items()},test={'x':h['x'][52:]}); pp=solve_batch(ss);print('baseline',er(np.r_[Y[:52],np.c_[pp['y_hat'],pp['state_pred']]]));print('times',times,'z',z.min(0),z.max(0));
for kind in ['z','zp','zq','zb','zpq']:
 X=np.c_[tuple(dict(z=z,p=p,q=q,b=b)[k] for k in kind)]; A=poly(X); results=[]
 for f in [A,np.c_[A,np.tanh(4*X)],np.c_[A,np.tanh(4*X),np.sqrt(.12+X*X)],np.c_[A,np.sin(2.5*X),np.sqrt(.25+X*X)],np.c_[A,X**3],np.c_[A,poly(np.tanh(1.25*X)),poly(np.sqrt(1+X*X))]]:
  row=[]
  for lam in [.0001,.003,.03,.3,1,3]:
   d,tt=er(ridge(f,Y,n,lam));row.append((round(d[0],4),round(tt,4)))
  results.append(row)
 print(kind,results)
