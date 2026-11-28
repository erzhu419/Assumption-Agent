exec(open('/tmp/readouts05.py').read().split('sol=[]')[0])
from scipy.linalg import qr
np.set_printoptions(precision=5,suppress=True)
times0=times
p0=np.array([.57257,2.265422,-.091598,.008623,9.917495,.003997,13.274296])

def trial(kind, nonlinear, joint, ii, jj):
 def feats(x,p):
  d,powr,l,a,f,c,h=p[:7]; z=x**3/(1+d*np.abs(x)**powr)+l*x+a*np.sin(f*x)+c*np.sin(h*x)
  if kind=='third': z+=p[7]*np.sin(p[8]*x)
  if kind=='even': z+=p[7]*(np.cos(f*x)-1)+p[8]*(np.cos(h*x)-1)
  if kind=='fifth': z+=p[7]*x**5
  if kind=='cosphase': z+=p[7]*(np.cos(p[8]*x)-1)
  return aff(z)
 def get(p):
  a=feats(U,p); op=ls(a,V); zz=feats(X,p)@op
  return a@op-V,zz
 def link(d,p):
  h=d@p
  return np.c_[np.ones(len(h)),h, {'atan':np.arctan,'tanh':np.tanh,'asinh':np.arcsinh}[nonlinear](h)]
 def tb(p):return np.c_[np.log1p(times*p[0]),-np.expm1(-times*p[1])]
 def residual(p,outputs=False):
  pr,z=get(p); a=aff(z); d=scalar(z,'paired'); b=tb(p[-2:]); sy=link(d,p[-8:-2]); cy=ls(sy[ii],y[ii]); ct=ls(a[ii],T[ii])@b@np.linalg.inv(b.T@b)
  yp=sy@cy; tp=a@ct@b.T
  if outputs:return np.sqrt(np.mean(pr**2)),np.mean(abs(yp[jj]-y[jj])),np.mean(abs(tp[jj]-T[jj])),np.std((yp-y)[ii]),np.std((tp-T)[ii]),p,cy
  return np.r_[pr.ravel(),joint*(yp-y)[ii],joint*(tp-T)[ii].ravel()]
 extra={'core':[],'third':[0,19.],'even':[0,0],'fifth':[0],'cosphase':[0,16.]}[kind]
 elo={'core':[],'third':[-.15,2],'even':[-.15,-.15],'fifth':[-.5],'cosphase':[-.15,2]}[kind]
 ehi={'core':[],'third':[.15,32],'even':[.15,.15],'fifth':[.5],'cosphase':[.15,32]}[kind]
 _,z=get(np.r_[p0,extra]); d=scalar(z,'paired'); ini=ls(d[ii],y[ii]); lo=np.r_[[.01,.7,-2,-.2,6,-.2,11],elo,[-100]*6,[.01,.01]]; hi=np.r_[[4,4,2,.2,12,.2,17],ehi,[100]*6,[10,10]]
 opts=[least_squares(residual,np.r_[p0,extra,scale*ini,[1.4,.7]],bounds=(lo,hi),max_nfev=130) for scale in (.7,1.5,3)]
 return residual(min(opts,key=lambda q:q.cost).x,True)
rng=np.random.default_rng(53); folds=[np.arange(72),rng.permutation(72),rng.permutation(72),rng.permutation(72)]
for kind,linkf,joint in [(k,'atan',w) for k in ('core','third','even','fifth','cosphase') for w in (.25,.7)] + [('core','tanh',.25),('core','asinh',.25)]:
 results=[trial(kind,linkf,joint,p[:56],p[56:]) for p in folds]
 print(kind,linkf,joint, np.array([r[:5] for r in results]),'params',results[0][-2], 'link',results[0][-1],flush=True)
