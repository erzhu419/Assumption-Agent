import pickle,numpy as np
from itertools import combinations_with_replacement as cwr
from scipy.optimize import least_squares
from scipy.linalg import qr
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
ls=lambda d,y:np.linalg.lstsq(d,y,rcond=1e-10)[0]
aug=lambda x:np.c_[np.ones(len(x)),x]
powr=lambda x,k:np.column_stack([np.prod(x[:,i],axis=1) for i in cwr(range(x.shape[1]),k)])
def cal():
 u,v=s['panel_a']['u'],s['panel_a']['v'];u=aug(u);take=np.ones(len(u),bool)
 for _ in range(10):
  a=ls(u[take],v[take]); r=np.linalg.norm(u@a-v,axis=1);take=r<max(.001,3*np.median(r))
 print('cal',take.sum(),np.sort(r),a,flush=True)
 return a
A=cal(); X=s['train']['x']; z=np.linalg.solve(A[1:].T,(X-A[0]).T).T
u,v=s['panel_b']['u'],s['panel_b']['v']; B=ls(aug(u),v);w=aug(z)@B
u,v1,v2=[s['panel_c'][i] for i in ('u','v1','v2')]; C=ls(aug(powr(u,2)),v1); print('C1',C);print('C2',ls(aug(np.c_[u,powr(u,2)]),v2));print('zcov',np.cov(z.T)); print('times',s['eval_times'])
y=s['train']['y'];t=s['train']['traj'];U,S,V=np.linalg.svd(t,full_matrices=False);print('S',S);t=(t@V[:2].T); yy=np.c_[y,t];
def score(F):
 p=F@ls(F[:52],yy[:52]);e=p-yy;return np.r_[np.abs(e[52:,0]).mean(),np.abs(e[52:,1:]@V[:2]).mean(),np.abs(e[:52,0]).mean(),np.abs(e[:52,1:]@V[:2]).mean()]
def c2fun(q):return aug(np.c_[q,powr(q,2)])@ls(aug(np.c_[u,powr(u,2)]),v2)
def run():
 out=[]
 for name,x in [('x',X),('z',z),('w',w),('Binv',np.linalg.solve(B[1:].T,(z-B[0]).T).T)]:
  c=aug(powr(x,2))@C;d=c2fun(x)
  groups={'x':x,'c':c,'d':d,'xc':x*c,'xd':x*d,'cd':c*d,'x3':powr(x,3),'c2':powr(c,2),'d2':powr(d,2),'c3':c**3,'d3':d**3,'tc':np.tanh(c),'td':np.tanh(d),'sc':np.sin(c),'sd':np.sin(d),'tx':np.tanh(2*x),'xcfull':(x[:,:,None]*c[:,None,:]).reshape(-1,16),'xdFull':(x[:,:,None]*d[:,None,:]).reshape(-1,16),'cdFull':(c[:,:,None]*d[:,None,:]).reshape(-1,16)}
  for k in [('x',),('c',),('d',),('x','c'),('x','d'),('c','d'),('x','c','d'),('x','c','x3'),('x','c','d','xc','xd','cd'),('x','c','d','c2','d2')]+[('x','c','d',j) for j in groups if j not in ('x','c','d')]+[('x','c','d',j,k) for j,k in cwr(list(groups)[3:17],2) if j!=k]:
   F=aug(np.column_stack([groups[i] for i in k])); e=score(F);out.append((e,name,k))
  for k in (2,3,4):
   F=aug(np.column_stack([powr(x,j) for j in range(1,k+1)]));out.append((score(F),name,('poly'+str(k),)))
 for i in (0,1):
  print('best',i)
  for e,n,k in sorted(out,key=lambda o:o[0][i])[:15]:print(np.round(e,5),n,k)
if __name__=='__main__':run()
