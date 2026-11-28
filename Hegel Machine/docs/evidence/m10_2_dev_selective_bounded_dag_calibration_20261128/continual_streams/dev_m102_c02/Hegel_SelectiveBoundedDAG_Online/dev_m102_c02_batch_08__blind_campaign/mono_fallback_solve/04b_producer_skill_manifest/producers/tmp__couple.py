import sys,pickle,numpy as np,itertools
sys.path.insert(0,'/holdout_task')
from batch_solver import _cal,_op,_cross,_lin,_ridge,_kernel
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; z=_cal(s['panel_a'],t['x']); y=t['y']; T=t['traj']; ts=s['eval_times']; b=_op(s['panel_b'],'v',z,'b'); c=_op(s['panel_c'],'v1',z,'c');
u,v=[s['panel_c'][k] for k in ('u','v2')]; pp=lambda x:np.c_[_lin(x),x*x,x**3]; d=pp(z)@np.linalg.lstsq(pp(u),v,rcond=None)[0];
g=np.random.default_rng(550); splits=[np.arange(72)]+[g.permutation(72) for _ in range(8)]; sf=_lin(np.c_[z,b,_cross(z,b)]); f=_lin(np.c_[b,c,_cross(b,c)]);
def loss(p):
 e=np.array([np.mean(abs(q-y[s[52:]])) for q,s in zip(p,splits)]);return e[0],e.mean(),e.max()
def fit(F,target,r):return [_ridge(F[s[:52]],target[s[:52]],F[s[52:]],r) for s in splits]
base=fit(sf,T,.08); old=[]
for s0 in splits:
 old.append((.75*_kernel(f[s0],b[s0],y[s0[:52],None],52,2,.001)+.25*_kernel(f[s0],b[s0],y[s0[:52],None],52,1,.03)).ravel())
print('old',loss(old))
D={'b':b,'c':c,'z':z,'d':d,'cube':z**3,'tanh':np.tanh(1.25*z),'rat':z/(1+1.25*z*z),'tanb':np.tanh(b),'tanc':np.tanh(c),'tand':np.tanh(d),'z5':z**5}
F={'just_'+k:_lin(a) for k,a in D.items()}; F.update({'bz':sf,'bc':f}); F.update({k+'_'+j:_lin(np.c_[a,h]) for (k,a),(j,h) in itertools.combinations(D.items(),2)}); F.update({k+'cross'+j:_lin(np.c_[a,h,_cross(a,h)]) for (k,a),(j,h) in itertools.combinations(D.items(),2) if not (k in ['cube','z5'] and j in ['cube','z5'])}); F.update({k+'self':_lin(np.c_[a,a*a,a**3]) for k,a in D.items()});
L=[]
for nm,A in F.items():
 for r in [.07,.4,1.]:
  for co in [0,.8,1.,1.15,1.3]:
   p=[a+co*b0[:,-1] for a,b0 in zip(fit(A,y-co*T[:,-1],r),base)]; L.append((loss(p),nm,r,co))
for idx in range(3):print('best',idx,sorted(L,key=lambda t:t[0][idx])[:9]);
# Refine panel B as linear mixture of a common nonlinearity plus a weak channel-dependent residual; compare raw residuals from the readout.
P=np.linalg.svd(T-T.mean(0),full_matrices=False)[2];a=np.c_[_lin(b),T@P[:3].T]; e=y-a@np.linalg.lstsq(a,y,rcond=None)[0];print('y explained with privileged state',np.mean(abs(e)),np.linalg.lstsq(a,y,rcond=None)[0]);print('res correlations',np.round(np.corrcoef(np.c_[e,z,z*z,d,b].T)[0],3));print('res ordered',np.round(e,3));
