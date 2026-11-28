import sys,pickle,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task'); import node_00 as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; ls=m._ls; ones=m._ones
np.set_printoptions(precision=5,suppress=True,linewidth=140)
a=s['panel_a']; A=ones(a['u']); v=a['v']; off,scale=m._align(a); r=v-off-a['u']*scale; print('A residual sorted',np.sort(np.max(abs(r),axis=1))); good=np.max(abs(r),axis=1)<.15; print('A inlier fit',ls(A[good],v[good]),'max error',abs(A[good]@ls(A[good],v[good])-v[good]).max())
b=s['panel_b']; B=ls(ones(b['u']),b['v']); print('B',B,'sv',np.linalg.svd(B[1:])[1]);
c=s['panel_c']; u=c['u']; v=c['v1']; print('C',np.c_[u,v]); print('C1 sv',np.linalg.svd(v)[1]);
D={'sin':np.sin,'cos':np.cos,'tanh':np.tanh,'softsign':lambda t:t/(1+abs(t)), 'rat':lambda t:t/(1+t*t),'abs':abs,'gauss':lambda t:np.exp(-t*t),'rat2':lambda t:1/(1+t*t)}
for nam,z in [('raw',u),('B',ones(u)@B),('Binv',(u-B[0])@np.linalg.inv(B[1:]))]:
 for base in [z,u]:
  q=np.c_[ones(base),np.array([z[:,i]*z[:,j] for i in range(4) for j in range(i,4)]).T]; print('quadratic',nam,base is z,np.mean((q@ls(q,v)-v)**2)**.5)
  for k,f in D.items():
   def d(p): return np.c_[ones(base),f(p[0]*z+p[1:])]
   def err(p):
    w=d(p); return (w@ls(w,v)-v).ravel()
   res=[]
   for w in [.75,1.3,2.4,4.,6.,9.]:
    p=least_squares(err,[w,.1,-.1,.15,-.15],max_nfev=100).x; res.append((np.mean(err(p)**2)**.5,p))
   val,p=min(res,key=lambda x:x[0]); print(nam,base is z,k,round(val,7),p)
