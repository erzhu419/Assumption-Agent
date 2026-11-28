import pickle,numpy as np,glob,importlib.util
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; Y=np.c_[tr['y'],tr['traj']]; t=s['eval_times']; np.set_printoptions(precision=5,suppress=True)
sp=importlib.util.spec_from_file_location('m',glob.glob('/tmp/hegel_continual_archive/*02_solver.py')[0]); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
z=m._decode(x,m._calibration(s['panel_a'])); ops=m._panel_operators(s['panel_c']); a,b=[f(z) for f in ops]; u=s['panel_b']['u']; v=s['panel_b']['v']
def fit(d,y): return np.linalg.lstsq(d,y,rcond=None)[0]
def aff(x): return np.c_[np.ones(len(x)),x]
def poly(x,n): return np.column_stack([np.ones(len(x))]+[x[:,i].prod(axis=1) for k in range(1,n+1) for i in combinations_with_replacement(range(x.shape[1]),k)])
def project(fun,start):
 def res(p):
  d=fun(u,p); return (d@fit(d,v)-v).ravel()
 p=least_squares(res,start,max_nfev=250).x; d=fun(u,p); c=fit(d,v); print('b panel',p,np.sqrt(np.mean(res(p)**2))); return fun(z,p)@c
U,D,V=np.linalg.svd(tr['traj'],full_matrices=False); print('time',t,'modes',V[:2]); print('calib',m._calibration(s['panel_a'])); print('raw and decoded ranges',x.min(0),x.max(0),z.min(0),z.max(0))
fs=[lambda q,p:aff(q),lambda q,p:poly(q,2),lambda q,p:np.c_[aff(q),np.arctan(p[0]*q),np.sin(p[1]*q)],lambda q,p:aff(q+p[1]*np.cos(p[0]*q)),lambda q,p:aff(np.tanh(p[0]*q)),lambda q,p:np.c_[aff(q),np.sin(p[0]*q),q*q],lambda q,p:np.c_[aff(q),np.tanh(p[0]*q),q*q],lambda q,p:np.c_[aff(q),np.sin(p[0]*q),np.cos(p[1]*q)]]
B=[project(f,[1.2,1.9]) for f in fs]
def report(k,d):
 p=d[52:]@fit(d[:52],Y[:52]); e=np.abs(p-Y[52:]); q=d@fit(d,Y); print(k,d.shape[1],e[:,0].mean(),e[:,1:].mean(),np.sqrt(np.mean((q-Y)**2,axis=0))[[0,1,-1]])
for k,d in [('x',x),('z',z),('a',a),('b',b),('a+b',a+b),('a-b',a-b),('a*b',a*b),('ab',np.c_[a,b])]+[(f'B{i}',j) for i,j in enumerate(B)]:
 for deg in [1,2,3] if d.shape[1]==4 else [1]: report(f'{k} {deg}',poly(d,deg))
for k,d in [('zab',np.c_[z,a,b]),('zab++',np.c_[z,a,b,a*b,a*a,b*b]),('zabB',np.c_[z,a,b,*B]),('ab-quadratic',poly(np.c_[a,b],2)),('z-poly+a+b',np.c_[poly(z,3),a,b,a*b])]: report(k,aff(d))
