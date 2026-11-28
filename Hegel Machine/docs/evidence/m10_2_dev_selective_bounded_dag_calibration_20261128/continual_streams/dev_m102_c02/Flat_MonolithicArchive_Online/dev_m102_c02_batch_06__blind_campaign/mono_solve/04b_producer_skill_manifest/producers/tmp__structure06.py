exec(open('/tmp/search06.py').read().split('from scipy.spatial.distance')[0]); import sys;sys.path.insert(0,'/holdout_task')
from batch_solver import _quad,_cross,_ridge,_warp
from sklearn.model_selection import KFold
folds=[(np.arange(52),np.arange(52,72)),*KFold(4,shuffle=True,random_state=251).split(Y)]; Y=Y[:72]
z,b,q,r,c,d=[h[:72] for h in (z,b,q,r,c,d)]; out=[]
def ck(label,H,regs=(.001,.02,.1)):
 for reg in regs:
  errs=np.array([abs(_ridge(H[np.r_[tr,te]],Y[tr],len(tr),reg)-Y[te]).mean(0) for tr,te in folds]); out.append((errs[0,0],errs[1:,0].mean(),errs[1:,0].max(),errs[0,1:].mean(),errs[1:,1:].mean(),label,reg))
for thr in [.34,.36,.375,.385,.395,.405,.415,.43]:
 for p in [1.15,1.25,1.35,1.5]:
  w=_warp(z,thr,p)
  ck((thr,p,'full'),np.c_[z,w,_quad(z),_quad(w),_cross(w,z)])
  ck((thr,p,'lower'),np.c_[z,w,_quad(w),_cross(w,z)])
  for j in [-1,0,1,2]:
   g=np.roll(z,j,axis=1); H=np.c_[w,z,b*g,_quad(w)]; ck((thr,p,j,'z'),H)
   ck((thr,p,j,'extended'),np.c_[H,b*g*g,_quad(z)])
for j in [-1,0,1,2]:
 for q0 in [z,q,r,b,c,d]:
  g=np.roll(q0,j,axis=1)
  for k in [.22,.54,1.25]:
   nm=next(n for n,t in [('z',z),('q',q),('r',r),('b',b),('c',c),('d',d)] if t is q0)
   for form,G in [('mul',b*(1+k*g)),('sat',b/(1+k*g*g)),('r',b*(1+k*g/(1+.22*g*g))),('conv',b*(1+k*g*g))]:
    ck((nm,j,k,form),np.c_[G,_quad(G)]); ck((nm,j,k,form,'plus'),np.c_[G,_quad(G),z])
for f in [lambda l:l[0]+l[1]+.5*l[3],lambda l:l[1]+l[2],lambda l:l[3]+l[4]]:
 print('TOP',*sorted(out,key=f)[:10],sep='\n')
