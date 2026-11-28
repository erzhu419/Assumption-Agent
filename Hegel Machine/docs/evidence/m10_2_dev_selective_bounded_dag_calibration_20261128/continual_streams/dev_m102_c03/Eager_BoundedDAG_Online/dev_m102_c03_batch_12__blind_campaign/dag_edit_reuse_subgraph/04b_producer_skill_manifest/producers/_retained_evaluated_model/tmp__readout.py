import sys,pickle,numpy as np
from itertools import combinations_with_replacement
sys.path.insert(0,'/holdout_task'); from node_01 import _coordinates,_panels,_ls,_bias
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; n=52; z=_coordinates(s['panel_a'],t['x']); a,b=_panels(s['panel_c'],z); w=np.c_[a,b]; tar=np.c_[t['y'],t['traj']]; print('spectrum',np.linalg.svd(t['traj'],compute_uv=False)); print('z rms,range',z.std(0),z.min(0),z.max(0)); print('C errors',*[np.std(p-s['panel_c'][k]) for p,k in zip(_panels(s['panel_c'],s['panel_c']['u']),('v1','v2'))]);
F={'lin':w,'contract':np.c_[w,(a*b).sum(1),a[:,0]*b[:,1]],'ab':np.c_[w,(a[:,:,None]*b[:,None,:]).reshape(len(w),16)],'aa':np.c_[w,*[a[:,i]*a[:,j] for i,j in combinations_with_replacement(range(4),2)]],'bb':np.c_[w,*[b[:,i]*b[:,j] for i,j in combinations_with_replacement(range(4),2)]],'all2':np.c_[w,*[w[:,i]*w[:,j] for i,j in combinations_with_replacement(range(8),2)]]};F['ab_z']=np.c_[F['ab'],z];F['all2_z']=np.c_[F['all2'],z];
for k,f in F.items():
 f=_bias(f); c=_ls(f[:n],tar[:n]); r=f@c-tar; print(k,'train',np.mean(abs(r[:n,0])),np.mean(abs(r[:n,1:])),'CV',np.mean(abs(r[n:,0])),np.mean(abs(r[n:,1:])))
 if k in ('ab','all2'):
  idx=np.argsort(-abs(c[:,0]))[:15]; print('y coeff',list(zip(idx,np.round(c[idx,0],5)))); idx=np.argsort(-np.linalg.norm(c[:,1:],axis=1))[:15];print('t coeff',list(zip(idx,np.round(np.linalg.norm(c[idx,1:],axis=1),5))))
