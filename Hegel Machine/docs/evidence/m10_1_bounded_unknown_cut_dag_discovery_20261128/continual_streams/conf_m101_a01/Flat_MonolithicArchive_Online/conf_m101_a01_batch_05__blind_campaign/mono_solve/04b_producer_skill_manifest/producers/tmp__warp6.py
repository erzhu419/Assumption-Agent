from explore import *
from scipy.optimize import minimize_scalar
np.set_printoptions(precision=4,suppress=True,linewidth=150)
U=s['panel_c']['u']; wc=dict(zip(w.__code__.co_freevars,[c.cell_contents for c in w.__closure__]))['coeff']; zc=dict(zip(z.__code__.co_freevars,[c.cell_contents for c in z.__closure__]))['coeff']; print('w linear vs B',wc[1:5]@Bi,'quadratic vs B',wc[5:]@Bi, 'z leading vs B',zc[1:5]@Bi,sep='\n')
folds=[np.arange(52,72),np.arange(56,72)]+[np.random.default_rng(i).permutation(72)[:20] for i in range(8)]
def polys(h,k=2):return aug(old._poly(h,range(1,k+1)))
warps={'w(d)@Bi':(w(d)-w(np.zeros((1,4))))@Bi,'w(d)':w(d)-w(np.zeros((1,4))), 'w(b)@Bi':(w(b)-w(B[:1]))@Bi,'w(b)':w(b)-w(B[:1]), 'z(d)@Bi':(z(d)-z(np.zeros((1,4))))@Bi,'z(d)':z(d)-z(np.zeros((1,4))), 'z(b)@Bi':(z(b)-z(B[:1]))@Bi,'z(b)':z(b)-z(B[:1]), 'd²':d*d,'d²rat':d*d/(1+.16*d*d),'b²@Bi':b*b@Bi}
def run(ar,delta,target,k):
 vals=[]; es=[]
 for te in folds:
  tr=np.setdiff1d(np.arange(len(y)),te)
  def err(v, out=False):
   f=polys(ar+v*delta,k); m=ls(f[tr],target[tr]); return np.mean(abs(f[te]@m-target[te])) if out else np.mean((f[tr]@m-target[tr])**2)
  grid=np.linspace(-2,3,31); i=np.argmin([err(v) for v in grid]); p=minimize_scalar(err,bounds=(grid[max(0,i-1)],grid[min(len(grid)-1,i+1)]),method='bounded').x
  vals.append(p); es.append(err(p,True))
 return np.round([np.mean(vals),np.std(vals),*es[:2],np.mean(es[2:]),max(es)],4)
for nm,delta in warps.items():print(nm,[run(d,delta,target,k) for target in (y,T) for k in [2,3]])
# Nonlinear warped-coordinate coefficients fitted by variable projection; fit both endpoints independently and jointly.
for form in ['dquad','dpoly','bquad','wfree','wplus','zplus']:
 for tar in [y,T,np.c_[y,T]]:
  es=[]; co=[]
  for te in folds:
   tr=np.setdiff1d(np.arange(72),te)
   def feature(p):
    if form in ['dquad','bquad','dpoly']:
     a=b if form=='bquad' else d
     h=a+p[:4]*a*a/(1+np.exp(p[4:8])*a*a)
     if form=='dpoly': h+=p[8:]*a**3
    elif form=='wfree':h=d+(w(d)-w(np.zeros((1,4))))@p.reshape(4,4)
    elif form=='wplus':h=d+p*warps['w(d)@Bi']
    else:h=d+p*warps['z(d)@Bi']
    return polys(h)
   def err(p):
    f=feature(p)[tr]; return np.r_[(f@ls(f,tar[tr])-tar[tr]).ravel(), .008*p]
   p0=np.zeros({'dquad':8,'dpoly':12,'bquad':8,'wfree':16,'wplus':4,'zplus':4}[form]);
   if form in ['dquad','dpoly','bquad']:p0[4:8]=-1.8
   p=least_squares(err,p0,max_nfev=110).x; f=feature(p); pred=f[te]@ls(f[tr],tar[tr]); es.append(np.mean(abs(pred-tar[te]),axis=0));co.append(p)
  es=np.array(es); print(form,tar.shape,es[:2].mean(axis=tuple(range(1,es.ndim))),np.mean(es[2:]), np.max(es.mean(axis=tuple(range(1,es.ndim)))),np.mean(co,0))
