exec(open('/tmp/ablate.py').read().split('res=[]')[0]);from scipy.optimize import least_squares
from itertools import combinations
c=np.tanh(1.25*C); d=np.arctan(1.25*D)

def ridge(F,Y,tr,te,r):
 a=F[tr].mean(0);std=np.maximum(F[tr].std(0),1e-6); f=(F[tr]-a)/std; g=(F[te]-a)/std; ym=Y[tr].mean(0); q=np.linalg.solve(f.T@f+r*np.eye(f.shape[1]),f.T@(Y[tr]-ym));return g@q+ym,q/std[:,None]

def cv(f,r, target=Y):
 vals=[]
 for tr,te in inds:
  pred,_=ridge(f,target,tr,te,r);e=abs(pred-Y[te]); vals.append([e[:,0].mean(),e[:,1:].mean(),e[:,-1].mean()]);
 return np.array(vals)

simple=np.c_[c,d,F(c,d)]; _,coef=ridge(simple,Y,np.arange(N),np.arange(N),.5);print('linear matrices\n',coef[:,[0,1,-1]].round(3));print('C x D scalar',coef[8:,0].reshape(4,4)); print('C x D state',coef[8:,-1].reshape(4,4));print('C x D rank',np.linalg.svd(coef[8:,0].reshape(4,4))[1]); print('obs y affine from states',np.std(Y[:,0]-m._aff(Y[:,1:])@m._ls(m._aff(Y[:,1:]),Y[:,0]))); print('smallest temporal singular values',np.linalg.svd(Y[:,1:],compute_uv=False))
res=[]
for cname,cc in [('C',C),('tanhC',c),('C0',c1(B-B.mean(0))),('pre',c1(np.tanh(1.25*B)))]:
 for dname,dd in [('D',D),('tanhD',np.tanh(1.25*D)),('atanD',d)]:
  prod=F(cc,dd); cs=np.repeat(cc,4,axis=1); ds=np.tile(dd,(1,4));
  ff={'bilinear':prod,'sumdiff':np.c_[np.tanh(cs+ds),np.tanh(cs-ds)],'square':np.c_[F(cc,cc),F(dd,dd),prod],'tanhprod':np.tanh(1.25*prod),'rat':prod/(1+.22*(cs+ds)**2),'satC':prod/(1+1.25*cs**2),'satD':prod/(1+1.25*ds**2),'trig':np.sin(prod*1.25),'triple':np.c_[prod,prod*cs,prod*ds],'ratdiag':np.c_[cc*dd,cc/(1+1.25*dd**2),dd/(1+1.25*cc**2)],'diag':np.c_[cc*dd,cc**2*dd,cc*dd**2], 'expdiag':np.c_[cc*np.exp(dd),dd*np.exp(cc),np.tanh(cc+dd)]}
  for name,f in ff.items():
   for base,g in [('core',np.c_[cc,dd]),('plusz',np.c_[cc,dd,z]),('plusb',np.c_[cc,dd,B]),('corecross',np.c_[cc,dd,prod])]:
    if base=='corecross' and name=='bilinear':continue
    for r in (.04,.25,1.,3.):
     e=cv(np.c_[g,f],r);res.append((e.mean(0)+.5*e.max(0),(cname,dname,name,base,r),e))
for j in range(3):
 print('algebraic results',j)
 for l,desc,e in sorted(res,key=lambda a:a[0][j])[:8]:print(desc,'gate',e[0].round(4),'mean',e.mean(0).round(4),'worst',e.max(0).round(4))
