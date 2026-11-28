import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement,combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def fit(a,b): return np.linalg.lstsq(a,b,rcond=None)[0]
def lin(x): return np.c_[np.ones(len(x)),x]
def cal(p):
 u,v=p['u'],p['v']; a=lin(u); rng=np.random.default_rng(42); w=min((fit(a[i],v[i]) for i in (rng.choice(len(u),6,replace=False) for _ in range(200))),key=lambda w: np.median(np.linalg.norm(a@w-v,axis=1))); k=np.linalg.norm(a@w-v,axis=1)<2
 return np.array([fit(des(u,j)[k],v[k,j]) for j in range(4)])
def des(u,j):
 x=u[:,j]; q=u[:,(j+1)%4]; return np.c_[np.ones(len(u)),x,q,x*q,x**3]
def forward(u,w): return np.column_stack([des(u,j)@w[j] for j in range(4)])
def invert(x,w):
 z=x.copy()
 for _ in range(15):
  q=np.roll(z,-1,axis=1); j=np.arange(4); J=np.zeros((len(z),4,4)); J[:,j,j]=w[:,1]+w[:,3]*q+3*w[:,4]*z*z; J[:,j,(j+1)%4]=w[:,2]+w[:,3]*z; z-=np.linalg.solve(J,(forward(z,w)-x)[...,None])[...,0]
 return z
if __name__=='__main__':
 tr=s['train']; x=tr['x']; y=tr['y']; T=tr['traj']; t=s['eval_times']; w=cal(s['panel_a']); z=invert(x,w); print('calibration',w, 'inversion max',np.max(abs(forward(z,w)-x)), 'latent range',z.min(0),z.max(0)); b=s['panel_b']; b=lin(z)@fit(lin(b['u']),b['v']);
 for name,h in [('raw',x),('calibrated',z),('panel-b',b)]:
  for d in (1,2,3):
   ids=[i for k in range(d+1) for i in combinations_with_replacement(range(4),k)]; f=np.column_stack([h[:,i].prod(1) for i in ids]); Y=np.c_[y,T]; c=fit(f[:52],Y[:52]); e=f@c-Y; print(name,d,'train/val MAE',[(np.mean(abs(j[:,0])),np.mean(abs(j[:,1:]))) for j in (e[:52],e[52:])]);
   if d==2: print('terms y,tlast',[(i,*a.round(4)) for i,a in zip(ids,c[:,[0,-1]])]);
