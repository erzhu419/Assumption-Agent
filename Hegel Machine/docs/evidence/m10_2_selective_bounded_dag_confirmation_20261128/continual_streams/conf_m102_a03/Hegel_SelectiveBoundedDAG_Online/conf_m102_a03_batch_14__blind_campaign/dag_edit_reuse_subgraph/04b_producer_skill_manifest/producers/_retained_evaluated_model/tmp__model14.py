import pickle, numpy as np, itertools,sys
sys.path.insert(0,'/holdout_task')
from node_00 import _calibrate,_panel,_fit,_affine
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; print('telemetry',s['telemetry_vec'],'times',s['eval_times']); x=_calibrate(s['panel_a'],s['train']['x']); b=_panel(s['panel_b'],'v',x); c=_panel(s['panel_c'],'v1',x); d=_panel(s['panel_c'],'v2',x); n=len(x)
print('ranges',[(a.min(),a.max()) for a in [x,b,c,d]])
y=s['train']['y']; t=s['train']['traj']; q=np.linalg.svd(t,full_matrices=False)[2][:3]; v=t@q.T
print('state temporal basis',q.round(3)); print('first rows',np.c_[x,b,c,d,y,v][:4].round(3))
def pol(a,degree=2):return np.column_stack([np.prod(a[:,j],axis=1) for j in itertools.combinations_with_replacement(range(a.shape[1]),degree)])
def summary(a,target):
 p=a[52:]@_fit(a[:52],target[:52]); rm=np.mean(abs(p-target[52:]),axis=0); r=target-a@_fit(a,target); return np.round(rm,5),np.round(np.mean(abs(r),axis=0),5)
for name,a in [('x',x),('b',b),('c',c),('d',d),('c+d',c+d),('cd',np.c_[c,d]),('xbcd',np.c_[x,b,c,d])]:
 for k in [1,2,3]:
  if a.shape[1]>4 and k>2:continue
  A=np.c_[_affine(a),*[pol(a,j) for j in range(2,k+1)]]
  print(name,k,summary(A,y),summary(A,v))
for name,a in [('b',b),('c',c),('d',d)]:
 A=np.c_[_affine(a),pol(a)]
 for target in (y,v):
  print('coefs',name,_fit(A,target).round(3).T)
