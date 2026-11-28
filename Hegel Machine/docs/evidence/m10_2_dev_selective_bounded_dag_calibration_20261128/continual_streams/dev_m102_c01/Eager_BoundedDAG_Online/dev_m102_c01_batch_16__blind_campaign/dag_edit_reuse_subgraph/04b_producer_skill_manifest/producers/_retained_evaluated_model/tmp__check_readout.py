import importlib.util, pickle, numpy as np
from scipy.linalg import solve

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
p=load('prep','/holdout_task/_frozen_nodes/node_00.py');base=load('baseline','/holdout_task/node_01.py');s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];x=s['train']['x'];y=s['train']['y'];idx=np.arange(len(y))
def prep(tr,te):
 return p.run_node({'__input__':dict(s,train={**s['train'],**{k:s['train'][k][tr] for k in ('x','y','traj')}},test={'x':x[te]})})
def predict(p,w,d,r,profile):
 K,H=p['K'],p['Kq'];S,T=p['sy'],p['qy'];scale=np.median(np.diag(K));G=K+.3*(S@S.T)+w*(K/scale)**d+r*np.eye(len(K));C=H+.3*(T@S.T)+w*(H/scale)**d;z=p['y_centered'];mu=p['y_mean']
 if profile:
  u,v=solve(G,np.column_stack((z,np.ones(len(z)))),assume_a='pos').T;off=u.sum()/v.sum();return mu+off+C@(u-off*v)
 return mu+C@solve(G,z,assume_a='pos')
rng=np.random.default_rng(27041);splits=[('random',rng.choice(idx,15,replace=False)) for _ in range(30)]
for j in range(x.shape[1]):
 order=np.argsort(x[:,j]);splits.extend([('axis',order[:m]) for m in (8,12,16)]);splits.extend([('axis',order[-m:]) for m in (8,12,16)])
for _ in range(20):
 direction=rng.normal(size=x.shape[1]);splits.append(('oblique',np.argsort(x@direction)[-12:]))
splits.extend([('radius',np.argsort((x*x).sum(1))[-m:]) for m in (8,12,16)])
opts=[(1.,2,.008,True),(.25,3,.008,True),(.25,3,.001,False),(.25,2,.008,True),(1.,2,.001,True),(2.,3,.008,True)];errs=[[] for _ in range(len(opts)+1)]
for label,te in splits:
 tr=np.setdiff1d(idx,te);q=prep(tr,te);pred=[base.run_node({'node_00':q})['y_hat']]+[predict(q,*o) for o in opts]
 for e,z in zip(errs,pred):e.append(np.abs(z-y[te]).mean())
for opt,e in zip(['baseline']+opts,errs):
 print(opt,{k:tuple(np.round([np.mean(a),max(a)],6)) for k in ('random','axis','oblique','radius') if (a:=[z for z,(l,_) in zip(e,splits) if l==k])})
print('prepared shapes:',{k:np.shape(v) for k,v in q.items()})
