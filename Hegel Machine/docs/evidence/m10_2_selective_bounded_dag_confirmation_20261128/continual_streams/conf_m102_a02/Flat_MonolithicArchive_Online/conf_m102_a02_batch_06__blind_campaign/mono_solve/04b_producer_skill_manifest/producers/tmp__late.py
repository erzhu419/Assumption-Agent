import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task'); import batch_solver as m
from scipy.optimize import least_squares
from scipy.linalg import qr
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=m._inverse(s['train']['x'],m._calibration(s['panel_a'])); b=m._operator(s['panel_b'])(x); y=s['train']['y']; S=s['train']['traj']; splits=[np.arange(len(x))]+[np.random.default_rng(k).permutation(len(x)) for k in range(4)]
A=m._features(x,b); U=np.c_[x,b]; D=np.c_[np.ones(len(x)),U,U**2]; R=[]
for bn,F in [('a',A),('full',m._quad(U)),('bare',m._quad(b)),('add',A[:,:29]),('sparse',np.c_[m._quad(b),x,x*b,x*np.roll(b,-1,axis=1),b*np.roll(x,-1,axis=1)])]:
 for dn,G in [('x',x),('b',b),('xb',U),('qxb',D[:,1:]),('mix',np.c_[x,b,x*b]),('sumdiff',np.c_[x,b,b+np.roll(b,-1,axis=1),b-np.roll(b,-1,axis=1),x+np.roll(x,-1,axis=1),x-np.roll(x,-1,axis=1)])]:
  for lam in [.0001,.001,.005]:
   for mode in ['none','grid','proj','lin']:
    out=[]; details=[]
    for ids in splits:
     tr,te=ids[:52],ids[52:]; f=F/np.maximum(np.linalg.norm(F[tr],axis=0),1e-10); g=G/np.maximum(G[tr].std(0),.1); ft=f[tr]; gt=g[tr]; yt=y[tr]; E=np.eye(f.shape[1]);
     def fit(d):
      a=ft/d[tr,None]; w=np.linalg.solve(a.T@a+lam*E,a.T@yt); return a@w-yt,w
     if mode=='none':d=np.ones(len(x))
     if mode=='grid':
      opts=[1+c*g[:,j]**2 for j in range(g.shape[1]) for c in [.05,.15,.35,.8]]; d=min(opts,key=lambda z:np.mean(fit(z)[0]**2)+.0003*np.mean((z[tr]-1)**2))
     if mode=='proj':
      def fun(p):return np.r_[fit(1+(g@p)**2)[0],.10*p]
      P=[]
      for j in np.argsort(abs(gt.T@fit(np.ones(len(x)))[0]))[-2:]:
       start=np.zeros(g.shape[1]); start[j]=.25; P.append(least_squares(fun,start,max_nfev=40))
      p=min(P,key=lambda z:z.cost).x; d=1+(g@p)**2; details.append(p.round(2))
     if mode=='lin':
      dt=g*g; a=np.c_[ft,-yt[:,None]*dt[tr]]; p=np.linalg.solve(a.T@a+np.diag(np.r_[np.repeat(lam,f.shape[1]),np.repeat(.3,g.shape[1])]),a.T@yt)[f.shape[1]:]; d=np.maximum(.4,1+dt@p)
     e,w=fit(d); out.append(np.mean(abs(f[te]@w/d[te]-y[te])))
    R.append((np.array(out),bn,dn,lam,mode,details[:1]))
for k in [lambda r:r[0][0],lambda r:r[0].mean(),lambda r:max(r[0])]:
 for r in sorted(R,key=k)[:7]:print('R',r)
