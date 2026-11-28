from contextlib import redirect_stdout
from io import StringIO
with redirect_stdout(StringIO()):
 from cvprobe import *
from scipy.optimize import minimize_scalar, least_squares
p=s['panel_c']; u=p['u'];
def fit2(z,k):return np.c_[aff(z),z*z/(1+k*z*z)]
f=lambda k: np.mean((fit2(u,k)@ls(fit2(u,k),p['v2'])-p['v2'])**2)
k=minimize_scalar(f,bounds=(.01,2),method='bounded').x
print('v2 rational',k,f(k)**.5)
B=ops['panel_bv']; A=ops['panel_av'].copy(); A[1:]=np.diag(np.diag(A[1:])); V2=ls(fit2(u,k),p['v2']);
for form in ('atan','atan+id','atan+sin','atan+qrat','atan+atan','soft','odd_rat','atan+odd_rat'):
 def feat(z,w):
  d=np.arctan(w[0]*z); out={'atan':lambda:d,'atan+id':lambda:np.c_[d,z],'atan+sin':lambda:np.c_[d,np.sin(w[1]*z)],'atan+qrat':lambda:np.c_[d,z*z/(1+w[1]*z*z)],'atan+atan':lambda:np.c_[d,np.arctan(w[1]*z)],'soft':lambda:z/np.sqrt(1+w[0]*z*z),'odd_rat':lambda:z/(1+w[0]*z*z),'atan+odd_rat':lambda:np.c_[d,z/(1+w[1]*z*z)]}[form]();return aff(out)
 def r(w):
  h=feat(u,w); return (h@ls(h,p['v1'])-p['v1']).ravel()
 opts=[least_squares(r,start,bounds=(.02,10.),max_nfev=350) for start in ([2.2,.7],[2.2,2],[2.2,4.5])]; opt=min(opts,key=lambda a:a.cost)
 print(form,opt.x,np.sqrt(np.mean(opt.fun**2)))
 if form=='atan+id':
  a1=opt.x; H1=lambda z:aff(np.c_[np.arctan(a1[0]*z),z]); V1=ls(H1(u),p['v1'])
fun={'A':lambda z:aff(z)@A,'a':lambda z:(z-A[0])/np.diag(A[1:]),'B':lambda z:aff(z)@B,'b':lambda z:(z-B[0])@np.linalg.inv(B[1:]),'C':lambda z:H1(z)@V1,'D':lambda z:fit2(z,k)@V2}
from itertools import product
res=[]
for pre in ('','A','a','B','b','AB','aB','BA','Ba','Ab','ab','bA','ba'):
 z=x
 for v in pre:z=fun[v](z)
 c,d=fun['C'](z),fun['D'](z)
 feats={'Q':poly(z,2),'Q+CD':np.c_[poly(z,2),c,d],'Q+zC':np.c_[poly(z,2),c,z*c],'Q+zD':np.c_[poly(z,2),d,z*d],'CD':aff(np.c_[c,d]),'CDQ':poly(np.c_[c,d],2),'Dq':poly(d,2),'Cq':poly(c,2),'Q+rat':np.c_[poly(z,2),z**3/(1+k*z*z),z*z/(1+k*z*z)],'Q+cross':np.c_[poly(z,2), c,d,c*d,z*c,z*d]}
 for name,F in feats.items():
  for p1 in (0.,.02,.2,1.):
   if p1==0 and F.shape[1]>=35:continue
   w=np.repeat(p1,F.shape[1]);w[0]=0;w[1:5]*=.01
   a=cv(F,w);res.append((pre,name,p1,a))
for i in range(3):
 print('metric',i)
 for pre,name,p1,a in sorted(res,key=lambda q:q[3][:,i].max())[:8]: print(pre,name,p1,np.round(a[:,i],5))
