from panel04 import *
from scipy.optimize import minimize_scalar

def calibrate(a,x):
 out=[]
 for u,v,z in zip(a['u'].T,a['v'].T,x.T):
  d=aff(u); q=ls(d,v)
  for _ in range(10):
   i=np.argsort(abs(d@q-v))[:40];q=ls(d[i],v[i])
  i=abs(d@q-v)<.002;q=ls(d[i],v[i]);out.append((z-q[0])/q[1])
 return np.array(out).T

def c1d(x):return np.c_[np.ones(len(x)),x[:,0]*x[:,1],x[:,0]*x[:,3],x[:,1]*x[:,2],x[:,2]*x[:,3]]

def operators(s):
 b=s['panel_b'];c=s['panel_c'];u=c['u']
 _,_,_,B=fit(b['u'],b['v'],*funcs['dead'])
 C=lambda x:c1d(x)@ls(c1d(u),c['v1'])
 specs={
 'r2':(lambda x,p:np.c_[np.ones(len(x)),x/(1+p[0]*x*x),x*x/(1+p[0]*x*x)],[.02],([-.4],[1.])),
 'r4':(lambda x,p:np.c_[np.ones(len(x)),x/(1+p*x*x),x*x/(1+p*x*x)],[.02]*4,([-.4]*4,[1.]*4)),
 'r6':(lambda x,p:np.c_[np.ones(len(x)),x/(1+p[0]*x*x+p[1]*x**4),x*x/(1+p[0]*x*x+p[1]*x**4)],[.02,0],([-.2,-.1],[1.,1.])),
 'root':(lambda x,p:np.c_[np.ones(len(x)),x/(1+p[0]*x*x)**p[1],x*x/(1+p[0]*x*x)**p[1]],[.04,.5],([0,.05],[2.,6.])),
 'ratabs':(lambda x,p:np.c_[np.ones(len(x)),x/(1+p[0]*abs(x)+p[1]*x*x),x*x/(1+p[0]*abs(x)+p[1]*x*x)],[.03,.03],([-.4,-.4],[2.,2.])),
 'trig':(lambda x,p:np.c_[np.ones(len(x)),np.sin(p[0]*x)/p[0],(1-np.cos(p[0]*x))/p[0]**2],[.2],([.001],[4.])),
 'cubic':(lambda x,p:np.c_[np.ones(len(x)),x+p[0]*x**3,x*x+p[1]*x**4],[-.02,0],([-1,-1],[1,1])),
 'quartic':(lambda x,p:np.c_[np.ones(len(x)),x,x*x,x**3,x**4],[0],([-1],[1.])),
 'ratioMix':(lambda x,p:np.c_[np.ones(len(x)),x,x*x]/(1+p[0]*np.sum(x*x,axis=1,keepdims=True)),[.02],([-.1],[1.])),
 }
 best=(np.inf,None)
 for name,spec in specs.items():
  r,l,p,f=fit(u,c['v2'],*spec)
  if __name__=='__main__':print('C2',name,round(r,7),round(l,7),p)
  if l<best[0]:best=(l,f)
 return {'B':B,'C':C,'D':best[1]}

if __name__=='__main__':
 tr=s['train'];t=s['eval_times']; y=tr['y'];T=tr['traj']; z=calibrate(s['panel_a'],tr['x']); ops=operators(s)
 paths={'':z}
 for k in range(1,4):
  for seq in itertools.permutations(ops,k):
   name=''.join(seq);paths[name]=ops[name[-1]](paths[name[:-1]])
 paths.update({a+'+'+b:np.c_[paths[a],paths[b]] for a,b in itertools.combinations(list(paths),2) if (a[:-1]==b[:-1] or a=='' or b=='' or (a[1:]==b[1:] and a[1:]))})
 for target in [y,T]:
  results=[]
  for name,w in paths.items():
   ds={'lin':aff(w),'quad':poly(w),'pair':np.c_[aff(w),w[:,0]*w[:,2],w[:,1]*w[:,3]],'abs':np.c_[aff(w),abs(w)],'tanh':aff(np.tanh(2*w))}
   for kind,d in ds.items():
    if d.shape[1]>45:continue
    q=ls(d[:52],target[:52]);err=np.mean(abs(d[52:]@q-target[52:]));trerr=np.mean(abs(d[:52]@q-target[:52]));results.append((err,trerr,name or 'A',kind))
  print('target',target.shape); print(*sorted(results)[:22],sep='\n')
 _,sing,v=np.linalg.svd(T,full_matrices=False);print('trajectory svd',sing);print('modes',v[:3]);print('mean',T.mean(0));print('basis fits')
 for name,g in [('t',t),('sqrt',np.sqrt(t)),('log',np.log1p(t)),('ones',np.ones_like(t))]:
  def loss(k):
   a=np.c_[1-np.exp(-k*t),g];return np.mean((ls(a,T.T).T@a.T-T)**2)
  r=minimize_scalar(loss,bounds=(.05,8),method='bounded');print(name,r.x,np.sqrt(r.fun))
