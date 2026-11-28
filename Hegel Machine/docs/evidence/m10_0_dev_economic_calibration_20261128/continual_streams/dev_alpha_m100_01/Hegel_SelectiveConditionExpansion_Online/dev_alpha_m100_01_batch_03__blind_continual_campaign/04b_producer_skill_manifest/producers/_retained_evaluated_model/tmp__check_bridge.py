import pickle, numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import root
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=s['calibration_load_cells']; x,y=c['true_channel_loads'],c['bridge_raw']
terms=[i for d in range(1,4) for i in combinations_with_replacement(range(4),d)]
def design(x,i,kind):
 z=np.atleast_2d(x); m=terms if kind=='full' else [(j,) for j in range(4)] + ([(i,(i+1)%4),(i,i,i)] if kind=='ring' else [])
 return np.column_stack([np.ones(len(z))]+[np.prod(z[:,j],axis=1) for j in m])
def fit(x,y,kind):
 a=[design(x,i,kind) for i in range(4)]; w=np.ones(len(x))
 for _ in range(20):
  b=[np.linalg.lstsq(ai*w[:,None],yi*w,rcond=None)[0] for ai,yi in zip(a,y.T)]; r=np.linalg.norm(np.column_stack([ai@bi for ai,bi in zip(a,b)])-y,axis=1); w=1/np.hypot(1,r/max(1e-8,2*np.median(r)))
 keep=r<max(1e-8,8*np.median(r)); b=[np.linalg.lstsq(ai[keep],yi[keep],rcond=None)[0] for ai,yi in zip(a,y.T)]
 return b,keep
def forward(q,b,kind): return np.column_stack([design(q,i,kind)@bi for i,bi in enumerate(b)])
def inverse(v,b,kind):
 o=np.array([bi[0] for bi in b]); M=np.array([bi[1:5] for bi in b]); sol=root(lambda q:forward(q,b,kind)[0]-v,np.linalg.solve(M,v-o),tol=1e-10); return sol.x,np.max(np.abs(sol.fun))
b,good=fit(x,y,'ring'); print('ring coefficients\n',np.array(b)); print('ring flagged',np.flatnonzero(~good)); print('axis',np.flatnonzero(np.count_nonzero(x,axis=1)<=1))
for k in ['affine','full','ring']:
 e=[]; f=[]
 for i in range(len(x)):
  mask=np.arange(len(x))!=i; bi,_=fit(x[mask],y[mask],k)
  v,rr=inverse(y[i],bi,k); e.append(v-x[i]); f.append(forward(x[i],bi,k)[0]-y[i]);
 e=np.array(e); f=np.array(f); h=np.max(np.abs(x),axis=1)>1.55; ax=np.count_nonzero(x,axis=1)<=1
 for label,m in [('clean',good),('edges',good&h),('axis',good&ax),('mixed',good&~ax)]: print(k,label,m.sum(),'load mae/max',np.mean(np.abs(e[m])),np.max(np.abs(e[m])),'sum max',np.max(np.abs(e[m].sum(1))),'forward max',np.max(np.abs(f[m])))
for part in ['train','test']:
 for key in ['past_channel_raw','hold_channel_raw']:
  vals=s[part][key].reshape(-1,4); z=[inverse(v,b,'ring') for v in vals]; q=np.array([a for a,r in z]); print(part,key,'domain',q.min(0),q.max(0),'worst closure',max(r for a,r in z))
