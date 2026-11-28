import numpy as np,pickle,sys,itertools
sys.path.insert(0,'/holdout_task/_frozen_nodes')
import node_00 as n
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; b=s['panel_b']; c=s['panel_c']; f=s['telemetry_vec'][0]; A=n._ls(n._aff(b['u']),b['v']); W=n.run_node({'__input__':s})['w']; U=(W-A[0])@np.linalg.pinv(A[1:]); print('B affine',A,'rmse',np.std(n._aff(b['u'])@A-b['v'],axis=0),'condition',np.linalg.cond(A[1:])); print('cal',n._calibrate(s['panel_a'],f)); T=s['train']['traj']; Y=s['train']['y']; print('trajectory singular',np.linalg.svd(T,compute_uv=False)); print('panel c cross correlations',np.round(np.corrcoef(np.column_stack([c['u'],c['v1'],c['v2']]).T)[:4,4:],3)); print('Y & T corr',np.corrcoef(np.c_[Y,T].T)[0]);
def feat(x,kind):
 q=[]
 for k in kind.split('+'):
  if k=='l': q.append(x)
  if k=='q': q.append(np.column_stack([x[:,i]*x[:,j] for i,j in itertools.combinations_with_replacement(range(4),2)]))
  if k=='t': q.append(np.column_stack([x[:,i]*x[:,j]*x[:,k] for i,j,k in itertools.combinations_with_replacement(range(4),3)]))
  for name,fn in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('atan',np.arctan),('asinh',np.arcsinh)]:
   if k.startswith(name): q.append(fn(float(k[len(name):])*x))
 return np.column_stack([np.ones(len(x)),*q]);
def test(x,y,k):
 d=feat(x,k); n=len(x); ix=n*3//4; P=np.linalg.lstsq(d[:ix],y[:ix],rcond=1e-10)[0]; r=d@P-y; return np.round([np.mean(abs(r[:ix])),np.mean(abs(r[ix:]))],5)
ks=['l','l+q','l+q+t','sin2.15','l+sin2.15','l+sin2.15+cos2.15','l+sin2.15+q','tanh2.15','l+tanh2.15','l+atan2.15','l+asinh2.15']
for name,x,v in [('C v1',c['u'],c['v1']),('C v2',c['u'],c['v2']),('C v1 from B',n._aff(c['u'])@A,c['v1']),('C v2 from B',n._aff(c['u'])@A,c['v2']),('Y U',U[:72],Y),('Y W',W[:72],Y),('T U',U[:72],T),('T W',W[:72],T)]:
 print(name, [(k,test(x,v,k).tolist()) for k in ks]);
