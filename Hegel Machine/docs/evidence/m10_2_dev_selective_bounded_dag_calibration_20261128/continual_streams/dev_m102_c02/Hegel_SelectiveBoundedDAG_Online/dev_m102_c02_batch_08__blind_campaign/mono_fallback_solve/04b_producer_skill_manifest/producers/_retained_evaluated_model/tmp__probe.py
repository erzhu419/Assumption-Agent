import numpy as np,pickle
from itertools import combinations_with_replacement as cr
from scipy.linalg import lstsq
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True,linewidth=140)
def poly(x,d):
 return np.column_stack([np.ones(len(x))]+[np.prod(x[:,i],axis=1) for k in range(1,d+1) for i in cr(range(x.shape[1]),k)])
def learn(x,y):return lstsq(x,y,cond=1e-10)[0]
def mae(a):return np.mean(abs(a),axis=0)
for name in 'abc':
 p=s['panel_'+name]; u=p['u']; print('\nPANEL',name)
 for nm,v in p.items():
  if nm=='u':continue
  for xf,yf in [(lambda z:z,lambda z:z),(lambda z:z,np.arctanh),(np.tanh,lambda z:z)]:
   if yf==np.arctanh and np.max(abs(v))>=1:continue
   for d in (1,2):
    x=poly(xf(u),d); y=yf(v); b=learn(x,y); e=y-x@b; h=np.sum(np.linalg.qr(x)[0]**2,axis=1); print(nm,xf.__name__,yf.__name__,d,'mae',mae(e),'loo',mae(e/(1-h[:,None])))
    if name=='a' and d==1:print(b, '\nres rows',np.sort(np.linalg.norm(e,axis=1)))
    if name!='a' and d==2 and yf!=np.arctanh and xf!=np.tanh:print('coefs',b)
print('times',s['eval_times'])
t=s['train']; x=t['x']; y=np.c_[t['y'],t['traj']];
a=s['panel_a']; A=poly(a['u'],1); v=a['v']; keep=np.ones(len(A),bool)
for _ in range(3):
 b=learn(A[keep],v[keep]); e=np.linalg.norm(A@b-v,axis=1); keep=e<4*np.median(e)
c=(x-b[0])@np.linalg.pinv(b[1:]); print('robust A',b,'keep',keep.sum())
for nm,X in [('raw',x),('cal',c)]:
 for d in (1,2,3,4):
  D=poly(X,d); pr=D@learn(D[:52], y[:52]); print(nm,d,mae((pr-y)[:52]),mae((pr-y)[52:]))
