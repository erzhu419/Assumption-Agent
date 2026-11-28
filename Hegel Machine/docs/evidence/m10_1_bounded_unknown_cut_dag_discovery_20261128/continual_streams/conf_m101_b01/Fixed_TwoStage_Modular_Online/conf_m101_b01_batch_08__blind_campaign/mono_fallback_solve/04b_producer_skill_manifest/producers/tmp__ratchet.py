from compose import *
from scipy.spatial.distance import cdist
from itertools import combinations
u,v=[s['panel_b'][k] for k in ('u','v')]
# Separate channel-specific warps from a single shared warp.
def full(p,x,kind):
 if kind=='channel': return aff(x+np.sin(x*p[:4])*p[4:])
 f,a=p[:2]; h=x+a*np.sin(f*x)
 if kind=='tanh': h=x+a*np.tanh(f*x)
 if kind=='warp': h=x+a*np.sin(f*x+p[2]*x**3)
 if kind=='out': return aff(h)
 if kind=='cross': h=h+p[2]*R(x)*R(R(x))
 if kind=='bilinear': h=h+p[2]*cycle(x)
 if kind=='inputshift': h=x+a*np.sin(f*(x+p[2]*R(x)))
 if kind=='two':h=h+p[2]*np.tanh(p[3]*x)
 return aff(h)
def fitB(tr,kind,p0):
 def parts(p):
  y=v if kind!='out' else v+p[2]*v**3
  d=full(p,u[tr],kind);return d,ls(d,y[tr]),y
 def residual(p):
  d,w,y=parts(p);return (d@w-y[tr]).ravel()
 p=least_squares(residual,p0,max_nfev=150).x;_,w,_=parts(p)
 def pred(x):
  a=full(p,x,kind)@w
  if kind=='out':
   for _ in range(5):a-=(a+p[2]*a**3-full(p,x,kind)@w)/(1+3*p[2]*a*a)
  return a
 return pred,p
bv=[]
for k,p0 in [('base',[2.8,-.4]),('channel',[2.8]*4+[-.4]*4),('warp',[2.8,-.4,0]),('tanh',[2.8,-.4]),('out',[2.8,-.4,0]),('cross',[2.8,-.4,0]),('bilinear',[2.8,-.4,0]),('inputshift',[2.8,-.4,0]),('two',[3.,-.3,-.05,2.])]:
 es=[]
 for i in range(4):
  te=np.arange(i*6,(i+1)*6); tr=np.setdiff1d(np.arange(24),te); f,p=fitB(tr,k,p0);es.append(np.mean(abs(f(u[te])-v[te])))
 f,p=fitB(np.arange(24),k,p0); bv.append((np.mean(es),k,f)); print('B',k,np.round(es,5),np.mean(abs(f(u)-v)),p,flush=True)
a=afit(s)['diagsecond']; z=inverse(a,s['train']['x']); op=operators(s); op['B']=min(bv,key=lambda r:r[0])[2]; B,C,D=[op[k] for k in 'BCD']; b=B(z);c=C(b);d=D(b); dc=D(c)
Y=np.c_[s['train']['y'],s['train']['traj']]; folds=[np.arange(52,72)]+[np.random.default_rng(i).permutation(72)[:20] for i in [42,43,44,45,46]]
sets={'b':b,'c':c,'d':d,'dc':dc,'z':z,'bcd':np.c_[b,c,d],'cd':np.c_[c,d],'zb':np.c_[z,b],'zcd':np.c_[z,c,d]}
# Test a compact additive kernel, second-order interactions, and monotone link on the common branch signals.
def do(tr,te,groups,style,length,lam,link):
 mats=[]
 for g in groups:
  q=sets[g]; q=(q-q[tr].mean(0))/q[tr].std(0).clip(1e-7); q=q.clip(-8,8); r=cdist(q,q,'sqeuclidean')/q.shape[1]; dot=q@q.T/q.shape[1];
  if style=='rbf': K=np.exp(-r/length)
  elif style=='matern': R=np.sqrt(r/length);K=(1+R)*np.exp(-R)
  elif style=='poly': K=(1+dot/length)**2
  elif style=='poly3': K=(1+dot/length)**3
  elif style=='hybrid': K=np.exp(-r/length)+.35*dot+.1*dot**2
  elif style=='add': K=sum(np.exp(-(p[:,None]-p[None,:])**2/length) for p in q.T)/q.shape[1]+.25*dot**2
  mats.append(K)
 K=np.mean(mats,axis=0); yt=Y[tr].copy(); ctr=yt.mean(0); sc=yt.std(0); yt=(yt-ctr)/sc
 if link: yt=np.arcsinh(link*yt)/link
 w=np.linalg.solve(K[np.ix_(tr,tr)]+lam*np.eye(len(tr)),yt); pred=K[np.ix_(te,tr)]@w
 if link: pred=np.sinh(np.clip(link*pred,-3,3))/link
 e=abs(pred*sc+ctr-Y[te]); return e[:,0].mean(),e[:,1:].mean()
res=[]
for groups in [('bcd',),('cd',),('b',),('c',),('zb',),('zcd',),('b','c','d'),('b','c','d','dc'),('z','c','d')]:
 for style in ['rbf','matern','poly','poly3','hybrid','add']:
  for l in [.5,1.5,5,16]:
   for lam in [.005,.04,.2]:
    for link in [0,1]:
     es=np.array([do(np.setdiff1d(np.arange(72),te),te,groups,style,l,lam,link) for te in folds]);res.append((groups,style,l,lam,link,es))
for i in range(2):
 for m in ['gate','mean','worst']:
  def key(r):
   x=r[-1][:,i]; return x[0] if m=='gate' else x.mean() if m=='mean' else x.max()
  print(i,m,[(r[:-1],r[-1][0].round(4),r[-1].mean(0).round(4),r[-1].max(0).round(4)) for r in sorted(res,key=key)[:5]])
