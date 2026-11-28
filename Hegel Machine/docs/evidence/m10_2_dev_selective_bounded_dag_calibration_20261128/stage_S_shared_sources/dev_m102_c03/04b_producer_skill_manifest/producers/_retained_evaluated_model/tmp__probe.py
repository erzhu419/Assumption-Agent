import pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];np.set_printoptions(precision=5,suppress=True,linewidth=140)
def bias(x):return np.c_[np.ones(len(x)),x]
def fit(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
p=s['panel_a'];U=bias(p['u']);r=p['v']-U@fit(U,p['v']);i=np.linalg.norm(r,axis=1)<3;A=fit(U[i],p['v'][i]);print('A',A,'resid',np.linalg.norm(p['v']-U@A,axis=1)); print('inverse',np.linalg.inv(A[1:]));p=s['panel_b'];B=fit(bias(p['u']),p['v']);print('B gram',B[1:]@B[1:].T)
x=s['train']['x'];y=s['train']['y'];T=s['train']['traj'];t=s['eval_times'];print('t',t);print('y',y[:30]);print('x range',x.min(0),x.max(0));print('summary compare',np.mean(y),np.std(y),np.mean(T[:,-1]),np.std(T[:,-1]),np.mean(T,axis=0));_,sig,V=np.linalg.svd(T,full_matrices=False);print('sv',sig);print('V',V[:3]);print('time/PC',np.linalg.lstsq(np.array([np.ones(len(t)),t,t**2]).T,V[:2].T,rcond=None)[1]);print('y near PCs',np.c_[y,T@V[:2].T][:25]);print('corr x/PC/y',np.corrcoef(np.c_[x,T@V[:2].T,y].T));print('mean drift after rank 2',T.mean(0)@(np.eye(len(t))-V[:2].T@V[:2]));print('panel c ranges',[(k,np.min(v,0),np.max(v,0)) for k,v in s['panel_c'].items()]);
