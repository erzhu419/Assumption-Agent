import numpy as n,pickle
from itertools import combinations_with_replacement as cwr
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; n.set_printoptions(precision=4,suppress=True)
def design(x,d):
 return n.column_stack([n.ones(len(x))]+[n.prod(x[:,ij],axis=1) for j in range(1,d+1) for ij in cwr(range(x.shape[1]),j)])
def score(A,Y):
 Q,R=n.linalg.qr(A,mode='reduced'); b=n.linalg.lstsq(R,Q.T@Y,rcond=None)[0]; r=Y-A@b; loo=r/n.maximum(1e-7,1-(Q*Q).sum(1))[:,None]; return n.mean(abs(r),axis=0),n.mean(abs(loo),axis=0), b
for k in ('panel_a','panel_b','panel_c'):
 p=s[k]; u=p['u']; print(k,'u range',u.min(0),u.max(0),'u rows',u[:3]);
 for j,v in p.items():
  if j=='u':continue
  print(j,'v range',v.min(0),v.max(0),'v rows',v[:3]);
  for d in range(1,4):
   D=design(u,d); print('poly',d,score(D,v)[:2] if D.shape[1]<len(u) else '-', 'inv',score(design(v,d),u)[:2] if D.shape[1]<len(u) else '-')
  if k=='panel_a':
   B=n.c_[design(u,1),u**2,u**3]; r,loo,b=score(B,v); print('diagonal cubic',r,loo,b)
   print('cubic full fit',score(design(u,3),v)[2][15:])
x=s['train']['x']; Y=n.c_[s['train']['y'],s['train']['traj']]; print('times',s['eval_times'],'x range',x.min(0),x.max(0));
for d in range(1,5):print('raw',d,score(design(x,d),Y)[:2])
