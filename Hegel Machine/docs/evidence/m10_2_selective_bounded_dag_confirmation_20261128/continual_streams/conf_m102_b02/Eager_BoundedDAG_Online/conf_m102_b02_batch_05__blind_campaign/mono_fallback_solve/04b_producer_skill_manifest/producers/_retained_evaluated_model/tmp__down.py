from discover import *
from scipy.optimize import least_squares
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.model_selection import KFold
tr=S['train']; z=coords(S,tr['x']); p=S['panel_b']; u,v=p['u'],p['v'];
fs={'erf':erf,'sin':np.sin,'tanh':np.tanh,'rat':lambda x:x/(1+x*x),'asinh':np.arcsinh,'atan':np.arctan,'log':lambda x:x*np.log1p(x*x),'soft':lambda x:x/np.sqrt(1+x*x)}
B={}
for name,f in fs.items():
 for mode in [0,1,2]:
  def F(q,a):
   h=f(a*q[0]); return D(np.c_[a,h] if mode==0 else a+q[1]*h if mode==1 else a+q[1:5]*f(a*q[0]))
  def r(q):
   G=F(q,u);return (G@fit(G,v)-v).ravel()
  opts=[least_squares(r,np.r_[k,np.full([0,1,4][mode],-1/k)],bounds=(np.r_[.03,np.full([0,1,4][mode],-100)],np.r_[7,np.full([0,1,4][mode],100)]),max_nfev=130) for k in [.45,.9,1.5,2.9,4.5]]
  o=min(opts,key=lambda o:o.cost); print('B',name,mode,o.x,np.std(o.fun));B[name+str(mode)]=(np.std(o.fun),F(o.x,z)@fit(F(o.x,u),v))
bs=sorted(B,key=lambda n:B[n][0]); print('best',bs[:8]);
t=S['eval_times']; T=tr['traj']; a,b=1.7936,.4178; F=np.array([-np.expm1(-a*t),-np.expm1(-b*t)]).T; Y=np.c_[tr['y'],(np.linalg.pinv(F)@T.T).T]; print('Y',np.std(Y,0),Y[:10]);
cv=[(np.arange(52),np.arange(52,72))]+list(KFold(4,shuffle=True,random_state=921).split(z));
for bn in bs[:4]:
 b=B[bn][1]; X=np.c_[z,b]; w=X/X.std(0);
 groups=[('base',w),('quad',P(w,2)[:,9:]),('triple',P(w,3)[:,45:]),('rat',w*w/(1+w*w)),('tanh',np.tanh(w)),('sin',np.sin(w)),('square',w*w),('abs',abs(w)),('roll',np.c_[z*np.roll(b,1,axis=1),z*b,z*np.roll(b,-1,axis=1)])];
 for label,keys in [('base',[0]),('quadratic',[0,1]),('sq',[0,6]),('cyclic',[0,8]),('nonlin',[0,3,4,5,6,7,8]),('full',[0,1,2,3,4,5,7])]:
  X1=D(np.column_stack([groups[k][1] for k in keys]));out=[]
  for lam in [1e-7,.0001,.003,.03,.15,.5,2,8]:
   es=[]
   for i,j in cv:
    sc=np.maximum(X1[i].std(0),.05);sc[0]=1; A=X1/sc; K=A@A[i].T; pred=K[j]@np.linalg.solve(K[i]+lam*np.eye(len(i)),Y[i]); es.append(np.r_[np.mean(abs(pred-Y[j]),0),np.mean(abs(pred[:,1:]@F.T-T[j]))]);
   es=np.array(es);out.append((es[0,[0,3]],es[1:,[0,3]].mean(0),lam))
  print('down',bn,label,*sorted(out,key=lambda x:sum(x[0])+sum(x[1]))[:2]);
 print('lin residual rank/coef',np.linalg.lstsq(D(w),Y,rcond=None)[0],np.linalg.svd(Y-D(w)@fit(D(w),Y))[1]);
