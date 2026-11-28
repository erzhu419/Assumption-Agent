import runpy,pickle,numpy as np
from scipy.optimize import minimize_scalar
from itertools import combinations
np.set_printoptions(precision=4,suppress=True,linewidth=160)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; z=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')['run_node']({'__input__':s})['z_tr']; c=s['panel_c']; u=c['u']; y=s['train']['y']; tr=s['train']['traj']; t=s['eval_times']; tel=s['telemetry_vec']; Y=np.column_stack((y,tr));
def stack(*a):return np.column_stack(a)
def one(x):return np.ones(len(x))
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def press(a,b):
 q,r=np.linalg.qr(a); return np.mean(np.abs((b-q@(q.T@b))/(1-(q*q).sum(1)[:,None])))
def fit(a,b):return np.sqrt(np.mean((a@ls(a,b)-b)**2))
def t1(x,k,lin):return stack(one(x),*( [x] if lin else []),np.tanh(k*x))
def t2(x):return stack(one(x),x,x*x)
for lin in [False,True]:
 for k in [tel[0],minimize_scalar(lambda k:press(t1(u,k,lin),c['v1']),bounds=(.5,4),method='bounded').x]:
  d=t1(u,k,lin); print('C1',lin,k,press(d,c['v1']),fit(d,c['v1']),ls(d,c['v1']).T)
for nm,d in [('sq only',stack(one(u),u*u)),('aff sq',t2(u)),('aff sq pair',stack(t2(u),*[u[:,i]*u[:,j] for i,j in combinations(range(4),2)]))]:print('C2',nm,press(d,c['v2']),fit(d,c['v2']),ls(d,c['v2']).T)
