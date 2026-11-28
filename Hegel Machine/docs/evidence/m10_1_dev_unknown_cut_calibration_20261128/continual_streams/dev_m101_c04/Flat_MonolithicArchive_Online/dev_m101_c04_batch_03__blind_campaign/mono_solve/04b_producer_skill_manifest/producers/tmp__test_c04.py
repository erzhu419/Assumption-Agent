import sys,pickle,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
from batch_solver import _features,_ls,_time_basis,solve_batch
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x,y,T=(tr[k] for k in ('x','y','traj')); f,c=_features(s,x); z=f[:,1:]; g=np.c_[f,c]; t=s['eval_times']; idx=np.arange(len(y))
splits=[idx]+[np.random.default_rng(i).permutation(len(y)) for i in (310,312,316,318,322,324)]+[np.argsort(np.linalg.norm(x,axis=1))]
Y={}; P={}
for ix in splits:
 a,b=ix[:56],ix[56:]; h=_time_basis(t,T[a]); Q=_ls(h,T[a].T).T
 def add(D,key,record,truth):record.setdefault(key,[]).append(np.mean(abs(D-truth),axis=0))
 for name,v in [('linear',f),('cross',g),('split_cross',np.c_[f,z[:,0]*z[:,2],z[:,1]*z[:,3]]),('cubic',np.c_[g,(f@_ls(f[a],y[a]))**3]),('quad',np.c_[f,*[z[:,i]*z[:,j] for i in range(4) for j in range(i,4)]])]:
  add(v[b]@_ls(v[a],y[a]),name,Y,y[b]); add((v[b]@_ls(v[a],Q))@h.T,name,P,T[b]); add(v[b]@_ls(v[a],T[a]),name+' free time',P,T[b]);
 for kind,fun in [('tanh',np.tanh),('sin',np.sin)]:
  for mode in ('cross_in','cross_out'):
   v=g if mode=='cross_in' else f
   def pred(p,i):return p[-2]*fun(v[i]@p[:v.shape[1]])+p[-1]+(c[i]*p[-3] if mode=='cross_out' else 0)
   q=least_squares(lambda p:pred(p,a)-y[a],np.r_[_ls(v[a],y[a]),0.,1.,0.],max_nfev=300).x
   add(pred(q,b),kind+mode,Y,y[b])
 stage={**s,'train':{**tr,'x':x[a],'y':y[a],'traj':T[a]},'test':{'x':x[b]}}; o=solve_batch(stage); add(o['y_hat'],'artifact',Y,y[b]); add(o['state_pred'],'artifact',P,T[b]);
for k,v in Y.items():print('y',k,np.round(v,6))
for k,v in P.items():print('t',k,'means',np.round(np.mean(v,axis=1),6),'worst terminal',np.max(np.array(v)[:,-1]))
w=_ls(g,np.c_[y,T]); print('fitted coeffs',w); print('fit RMS',np.std(g@w-np.c_[y,T],axis=0)); print('train/test vertex ranges', abs(x).max(0),abs(s['test']['x']).max(0))
