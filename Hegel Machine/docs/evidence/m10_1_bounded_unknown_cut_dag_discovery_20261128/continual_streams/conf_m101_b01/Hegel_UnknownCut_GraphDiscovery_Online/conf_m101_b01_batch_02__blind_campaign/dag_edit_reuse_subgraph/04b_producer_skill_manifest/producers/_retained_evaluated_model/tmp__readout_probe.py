import pickle,numpy as np,itertools,runpy
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; p=s['panel_c']; n=len(t['y']); y=t['y']; tr=t['traj']; times=s['eval_times']
z=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')['run_node']({'__input__':s}); z=runpy.run_path('/holdout_task/_frozen_nodes/node_01.py')['run_node']({'node_00':z})['x'][:n]; feat=runpy.run_path('/holdout_task/node_02.py')['_panel_features']; c=feat(z)@np.linalg.lstsq(feat(p['u']),p['v1'],rcond=None)[0]; g=feat(z)[:,1:]; u=p['u']; pow2=lambda a:np.c_[np.ones(len(a)),a,a*a]; b=pow2(z)@np.linalg.lstsq(pow2(u),p['v2'],rcond=None)[0]; f=np.c_[np.ones(n),g]; base=lambda f,t:np.linalg.lstsq(f,t,rcond=None)[0]; residual=lambda t:t-f@base(f,t); r=residual(y)
def score(q):
 w=base(q,y); e=y-q@w; lev=(q*np.linalg.pinv(q).T).sum(1); return np.sqrt(np.mean(e*e)),np.mean(abs(e/(1-lev))),np.mean(abs(y[-20:]-q[-20:]@base(q[:-20],y[:-20])))
print('y baseline RMSE/LOO MAE/tail MAE',score(f)); print('traj coeff',np.round(base(f,tr),4))
names=[]; cols=[]
def add(name,x): names.append(name); cols.append(x)
for tag,x in [('z',z),('b',b),('c',c),('g',g)]:
 for i,a in enumerate(x.T):
  for k,fn in [('lin',lambda a:a),('sq',np.square),('cub',lambda a:a**3),('tanh',np.tanh),('sin',np.sin),('cos',np.cos),('sat',lambda a:a/(1+a*a)),('abs',abs),('exp',np.exp)]: add(f'{tag}{i}_{k}',fn(a))
 for i,j in itertools.combinations(range(x.shape[1]),2):
  a,bb=x[:,i],x[:,j]
  for k,v in [('prod',a*bb),('sinprod',np.sin(a*bb)),('cosprod',np.cos(a*bb)),('sin+',np.sin(a+bb)),('sin-',np.sin(a-bb)),('cos+',np.cos(a+bb)),('cos-',np.cos(a-bb)),('abs+',abs(a+bb)),('abs-',abs(a-bb)),('quot',a/(1+bb*bb)),('antiquot',bb/(1+a*a))]: add(f'{tag}{i}_{j}_{k}',v)
print('top single residual features'); print(*sorted((score(np.c_[f,v]),k) for k,v in zip(names,cols) if np.linalg.norm(residual(v))>1e-8)[:24],sep='\n')
