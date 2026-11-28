import numpy as np,pickle,itertools
from scipy.optimize import least_squares
from scipy.special import erf
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def D(x):return np.c_[np.ones(len(x)),x]
def fit(F,y):return np.linalg.lstsq(F,y,rcond=1e-10)[0]
def P(u,d):return np.stack([np.prod(u[:,ii],axis=1) for j in range(d+1) for ii in itertools.combinations_with_replacement(range(u.shape[1]),j)],1)
def cal(s,x):
 p=s['panel_a']; F=D(p['u']); y=p['v']; keep=np.ones(len(F),bool)
 for _ in range(8):
  c=fit(F[keep],y[keep]); e=np.linalg.norm(F@c-y,axis=1);keep=e<max(.001,3*np.median(e))
 return np.linalg.solve(c[1:].T,(x-c[0]).T).T,c,e
if __name__=='__main__':
 np.set_printoptions(precision=4,suppress=True,linewidth=140)
 z,c,e=cal(S,np.r_[S['train']['x'],S['test']['x']]);print('cal',c,np.sort(e),'z min max',z.min(0),z.max(0))
 pc=S['panel_c'];u=pc['u']; G=P(u,2); v1=pc['v1'];v2=pc['v2']; v1c=fit(G,v1);print('v1 coefficients',v1c); print('v2 coefficients',fit(G,v2));
 for p,n in [(pc,'v2'),(S['panel_b'],'v')]:
  u,y=p['u'],p[n];print(n)
  for name,func in [('tanh',np.tanh),('sin',np.sin),('erf',erf),('asinh',np.arcsinh),('atan',np.arctan)]:
   for k in [1,2]:
    def calc(q):return P(func(q[0]*u),k)
    def res(q):
     F=calc(q);return (F@fit(F,y)-y).ravel()
    opt=min([least_squares(res,[a],bounds=(.05,7)) for a in [.45,1.2,2.5,4.5]],key=lambda o:o.cost);print(name,k,opt.x,np.std(opt.fun))
  if n=='v2':
   for F in [D(np.c_[u,v1]),D(np.c_[u,v1,u*u]),D(np.c_[u,v1,u*v1]),D(np.c_[u,v1,np.tanh(v1)]),D(np.c_[u,np.tanh(v1)])]: print('joint',F.shape,np.std(F@fit(F,y)-y))
 for lab,a in [('z',z),('c1',P(z,2)@v1c),('c2',P(z,2)@fit(G,v2))]:
  print(lab,np.std(a,0),np.quantile(a,[0,.05,.95,1],axis=0));
  for d in [1,2,3]:
   F=P(a,d);y=np.c_[S['train']['y'],S['train']['traj']]; c=fit(F[:52],y[:52]); print(d,'train',np.mean(abs(F[:52]@c-y[:52]),0)[[0,1,8]],'CV',np.mean(abs(F[52:72]@c-y[52:]),0)[[0,1,8]])
 t=S['eval_times'];y=S['train']['traj'];
 for typ in range(5):
  def get(q):
   a=-np.expm1(-q[0]*t)
   b=[t,np.sin(q[-1]*t),t*np.exp(-q[0]*t),-np.expm1(-q[-1]*t),np.tanh(q[-1]*t)][typ];return np.array([a,b]).T
  def r(q):
   F=get(q);return (F@fit(F,y.T)-y.T).ravel()
  opt=min([least_squares(r,[a,b],bounds=(.04,6),max_nfev=90) for a in [.3,.8,1.4] for b in [.5,1.3,2.5]],key=lambda o:o.cost);print('time model',typ,opt.x,np.std(opt.fun))
