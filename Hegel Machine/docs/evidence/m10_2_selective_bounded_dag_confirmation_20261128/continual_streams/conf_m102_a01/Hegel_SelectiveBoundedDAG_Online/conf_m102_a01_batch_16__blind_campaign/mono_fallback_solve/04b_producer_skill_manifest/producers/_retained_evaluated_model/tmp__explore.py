import numpy as np,pickle,importlib.util
from itertools import combinations_with_replacement
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; X=S['train']['x']; Y=S['train']['y']; T=S['train']['traj']; n=len(X)
p='/tmp/hegel_continual_archive/conf_m102_a01_batch_03_solver.py'; spec=importlib.util.spec_from_file_location('old',p); old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
A=lambda x: np.column_stack([np.ones(len(x)),x]); mae=lambda x: np.mean(abs(x),axis=0)
np.set_printoptions(precision=5,suppress=True,linewidth=150)
print('times',S['eval_times']); print('svd traj',np.linalg.svd(T,compute_uv=False)); print('trajectory vectors',np.linalg.svd(T,full_matrices=False)[2][:4]);
for k in ['panel_a','panel_b','panel_c']:
 p=S[k]; u=p['u']; q=A(u)
 for key,v in p.items():
  if key=='u':continue
  b=np.linalg.lstsq(q,v,rcond=None)[0]; r=q@b-v; print(k,key,'affine',b,'RMSE',np.sqrt(np.mean(r*r,axis=0)),'median',np.median(abs(r),axis=0)); print('rows residual',np.linalg.norm(r,axis=1))
z,h=old._coordinates(S,X); p=S['panel_b']; b=np.linalg.lstsq(A(p['u']),p['v'],rcond=None)[0]
C={'x':X,'z':z,'h':h,'bz':A(z)@b,'bx':A(X)@b,'ih':np.linalg.solve(b[1:].T,(X-b[0]).T).T}
for key in ['v1','v2']:
 p=S['panel_c']; c=np.linalg.lstsq(A(p['u']),p[key],rcond=None)[0]
 for k in ['x','z','h','bz']: C[key+k]=A(C[k])@c

def terms(g,d): return np.column_stack([np.prod(g[:,idx],axis=1) for idx in combinations_with_replacement(range(g.shape[1]),d)])
for k,g in C.items():
 l=A(g); q=terms(g,2); r=terms(g,3); F=np.column_stack((l,q,r)); F/=np.sqrt((F[:52]**2).mean(0));
 for mode,D in [('aff',l),('quadr',F[:,:15]),('cubic',F),('q+tr',np.c_[F[:,:15],T[:,-1]]),('nonlin',np.c_[l,np.tanh(1.25*g),g/(1+.22*g*g),q])]:
  pen= np.r_[np.zeros(5),np.full(min(10,D.shape[1]-5),.025),np.full(max(D.shape[1]-15,0),.6)]
  B=np.linalg.solve(D[:52].T@D[:52]+np.diag(pen+1e-8),D[:52].T@np.c_[Y[:52],T[:52,-1]])
  e=D@B-np.c_[Y,T[:,-1]]; print(k,mode,mae(e[:52]),mae(e[52:]),end=' | ')
 print()
print('rows: x, y, last, y affine residual, traj aff residual'); D=A(X); R=np.c_[Y,T[:,-1]]-D@np.linalg.lstsq(D[:52],np.c_[Y[:52],T[:52,-1]],rcond=None)[0]; print(np.c_[X,Y,T[:,-1],R])
