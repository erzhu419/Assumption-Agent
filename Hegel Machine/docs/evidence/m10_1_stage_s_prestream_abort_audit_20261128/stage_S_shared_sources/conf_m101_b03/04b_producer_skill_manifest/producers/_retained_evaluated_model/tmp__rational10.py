import sys,pickle,itertools as it,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
from batch_solver import _coordinates,_mode_fit,_predict,_ls
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; y=tr['y']; t=tr['traj']; time=s['eval_times']; d=_coordinates(s,x); a=s['panel_a']; u,v=a['u'],a['v']; aff=lambda u:np.c_[np.ones(len(u)),u]; beta=_ls(aff(u),v); keep=np.linalg.norm(aff(u)@beta-v,axis=1)<2*np.median(np.linalg.norm(aff(u)@beta-v,axis=1)); beta=_ls(aff(u[keep]),v[keep]); z=(x-beta[0])/beta[1:].diagonal(); B=s['panel_b']; b=_ls(aff(B['u']),B['v']); g=aff(d)@b
amp,F=_mode_fit(time,t); Y=np.c_[y,amp]; obs=np.c_[y,t]; Q=np.c_[aff(d),*[d[:,i]*d[:,j] for i in range(4) for j in range(i,4)]]; R=Y-Q@_ls(Q,Y); print('quadratic residual correlation',np.corrcoef(R.T).round(3)); print('quadratic residual spectrum',np.linalg.svd(R,compute_uv=False).round(4))
C=s['panel_c']; u,v=C['u'],C['v2']; H=lambda z,k:np.c_[aff(z),np.hypot(1,k*z)-1]; f=least_squares(lambda p:(H(u,p[0])@_ls(H(u,p[0]),v)-v).ravel(),[1]); coef=_ls(H(u,f.x[0]),v); print('v2 scale ratios',(coef[5:]/coef[1:5]).round(3)); print('v2 affine singular values',np.linalg.svd(coef[1:5],compute_uv=False))
def poly(z,deg):return np.column_stack([np.ones(len(z))]+[z[:,cols].prod(1) for n in range(1,deg+1) for cols in it.combinations_with_replacement(range(z.shape[1]),n)])
folds=[*np.array_split(np.random.default_rng(202).permutation(len(x)),6),np.arange(56,72)]
for coords,Z in [('D',d),('raw',z),('DB',g),('D+raw',np.c_[d,z])]:
 for mode in ('GP','P2','P3','rat12','rat22','rat32','rat22soft'):
  if mode=='GP' and coords!='D':continue
  if mode=='P3' and Z.shape[1]>4:continue
  if mode.startswith('rat') and Z.shape[1]>4:continue
  errors=[]
  for h in folds:
   l=np.setdiff1d(np.arange(len(x)),h); amps,T=_mode_fit(time,t[l]); yy=np.c_[y[l],amps]; zz=(Z-Z[l].mean(0))/Z[l].std(0)
   if mode=='GP':p=np.column_stack([_predict(Z[l],Z[h],v) for v in yy.T])
   elif mode.startswith('P'):
    H=poly(zz,int(mode[1])); p=H[h]@_ls(H[l],yy)
   else:
    H=poly(zz,int(mode[3])); S=poly(zz,2)[:,1:]; p=[]
    for v in yy.T:
     A=H[l]; W=S[l]; X=np.c_[A,-W*v[:,None]]; reg=.1 if mode.endswith('soft') else .025; P=np.r_[np.full(A.shape[1],reg*.15),np.full(W.shape[1],reg)]; c=_ls(np.r_[X,np.diag(P)],np.r_[v,np.zeros(len(P))]); den=1+S[h]@c[A.shape[1]:]; p.append(H[h]@c[:A.shape[1]]/np.maximum(den,.35))
    p=np.array(p).T
   errors.append(np.mean(abs(np.c_[p[:,0],p[:,1:]@T]-obs[h]),0))
  E=np.asarray(errors)[:,[0,1,4,8]];print(coords,mode,'mean',E[:-1].mean(0).round(4),'worst',E[:-1].max(0).round(4),'tail',E[-1].round(4))
