import pickle,numpy as np,itertools,sys
sys.path.insert(0,'/holdout_task')
from _frozen_nodes.node_00 import run_node,_inverse,_fit_sensor
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; p=run_node({'__input__':s}); z=p['z']; u=_inverse(s['train']['x'],_fit_sensor(s['panel_a'])); y=s['train']['y']; T=s['train']['traj']; n=len(y); I=np.arange(n); nte=20; tr=I[:-nte]; te=I[-nte:]; np.set_printoptions(precision=5,suppress=True,linewidth=140)
def poly(x,k): return np.column_stack([np.prod(x[:,j],axis=1) for l in range(k+1) for j in itertools.combinations_with_replacement(range(x.shape[1]),l)])
def reg(X,Y): return np.linalg.lstsq(X,Y,rcond=None)[0]
def mse(X,Y): return np.mean(abs(X@reg(X,Y)-Y))
def scores(X,Y):
 C=reg(X[tr],Y[tr]); return np.round([np.mean(abs(X[tr]@C-Y[tr])),np.mean(abs(X[te]@C-Y[te]))],6)
for k,panel in [('b',s['panel_b']),('c',s['panel_c'])]:
 x=panel['u']; print('panel',k,'u bounds',x.min(0),x.max(0))
 for key,v in panel.items():
  if key=='u':continue
  A=reg(poly(x,1),v); E=v-poly(x,1)@A; print(key,'affine',A,'resid',np.linalg.svd(E,compute_uv=False),'poly residual',[mse(poly(x,j),v) for j in (0,1,2,3)],'samples',np.c_[x[:4],v[:4]])
print('sensor',_fit_sensor(s['panel_a'])); print('traj sv',np.linalg.svd(T,compute_uv=False)); U,S,V=np.linalg.svd(T,full_matrices=False); print('traj time modes',V[:3]); print('traj correlations',np.corrcoef(np.c_[u,z,y,U[:,:2]].T)[-3:,:8]); print('traj first amplitudes',np.c_[u[:12],z[:12],y[:12],(U*S)[:12,:2]])
for name,q in [('raw',s['train']['x']),('u',u),('z',z)]:
 for j in range(1,4):
  X=poly(q,j); print('scores',name,j,scores(X,y),scores(X,T));
  if j==3 and name!='raw':
   powers=[l for k in range(4) for l in itertools.combinations_with_replacement(range(4),k)]; A=reg(X,np.c_[y,U[:,:2]*S[:2]]); print('large cubic fit terms',[sorted(zip(abs(c),powers,c),reverse=True)[:10] for c in A.T]);
