import importlib.util, pickle, numpy as np
spec=importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; all_=np.arange(len(tr['y']))
z=m._coordinates(s)[:72]; p,q=m._views(s,z)
def evaluate(hold, verbose=False):
 train=np.setdiff1d(all_,hold); study=dict(s,train={k:tr[k][train] for k in ('x','y','traj')},test={'x':tr['x'][hold]}); o=m.solve_batch(study)
 dy=abs(o['y_hat']-tr['y'][hold]); ds=abs(o['state_pred']-tr['traj'][hold]); print('mae',len(train),*np.round([dy.mean(),ds.mean(),*ds.mean(0),dy.max()],5))
 if verbose:
  h,j=m._temporal(s['eval_times'],tr['traj'][train]); xy=np.c_[m._affine(z),m._cyclic(z),z*z]; ro=m._fit(np.c_[xy[train],j],tr['y'][train]); j[:,1]=.5*(j[:,1]+(tr['y'][train]-np.c_[xy[train],j[:,0]]@ro[:-1])/ro[-1]); b=np.c_[xy,q]; pq=(p[:,:,None]*q[:,None,:]).reshape(72,16); w=m._fit(np.c_[b,pq][train],j); print('coefficients base',w[:-16].round(4).tolist(),'coupling slow',w[-16:,0].reshape(4,4).round(4),'coupling fast',w[-16:,1].reshape(4,4).round(4)); print('read',ro.round(4)); print('fit',np.mean(abs(np.c_[b,pq][train]@w-j),axis=0))
if __name__=='__main__':
 evaluate(all_[52:],True)
 for hold in np.array_split(all_,4): evaluate(hold)
 for hold in np.array_split(np.random.default_rng(202).permutation(all_),5): evaluate(hold)
 for hold in [np.argsort(np.linalg.norm(z,axis=1))[-20:], np.argsort(np.max(abs(z),axis=1))[-20:]]: evaluate(hold)
