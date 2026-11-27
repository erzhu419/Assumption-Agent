import importlib.util as iu, pickle, numpy as np
from pathlib import Path
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
mods=[]
for p in sorted(Path('/tmp/hegel_continual_archive').glob('*solver.py')):
 spec=iu.spec_from_file_location(p.stem,p); m=iu.module_from_spec(spec); spec.loader.exec_module(m); mods.append(m)
reflect, periodic, robin=mods[:3]
y=robin._calibrated_observations(s); Z=s['Z_nuisance']; nx,ny=s['nx'],s['ny']; i,j=s['sensor_ij'].T; idx=i+nx*j
rhs=np.column_stack([v.ravel(order='F') for v in s['source_modes']]); ident=sparse.eye(nx*ny,format='csc'); diag=sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc')
axes=[]
for a in 'xy':
 k,v,L,r=[s[key] for key in ('kappa_'+a,'vel_'+a,'L'+a,'robin_'+a)]; h=L/(len(k)-1)
 R=reflect._axis_operator(k,v,L); P=periodic._axis_operator(k,v,L); O=robin._axis_operator(k,v,r,L)
 axes.append((R,P,O,h)); print('axis',a,'reflect vs zero Robin',np.max(abs((R-robin._axis_operator(k,v,0,L)).toarray())), 'wrap edges',P[0,-1],P[-1,0]);
def forward(ax,ay):
 L=sparse.kron(sparse.eye(ny),ax,format='csc')+sparse.kron(ay,sparse.eye(nx),format='csc')+diag
 return np.vstack([splu(L+u*ident).solve(rhs)[idx] for u in s['mu_shifts']])
full=np.arange(len(y)); sensors=np.tile(np.arange(len(idx)),len(s['mu_shifts'])); shifts=np.repeat(np.arange(len(s['mu_shifts'])),len(idx))
def fit(S,tr):
 A=np.c_[S[tr,:-1]-S[tr,-1:],Z[tr]]; b=y[tr]-s['total_budget']*S[tr,-1]; q=np.linalg.lstsq(A,b,rcond=None)[0]; return np.r_[q[:7],s['total_budget']-q[:7].sum()],q[7:]
def err(S,tr,te):
 w,g=fit(S,tr); return S[te]@w+Z[te]@g-y[te]
for a in range(3):
 for b in range(3):
  S=forward(axes[0][a],axes[1][b]); w,g=fit(S,full)
  cv=[np.sqrt(np.mean(err(S,full[sensors%4!=f],full[sensors%4==f])**2)) for f in range(4)]
  cvshift=[np.sqrt(np.mean(err(S,full[shifts!=f],full[shifts==f])**2)) for f in range(3)]
  print('wall', 'RPO'[a]+'RPO'[b], 'fit rms', np.sqrt(np.mean(err(S,full,full)**2)), 'worst sensor/shift',max(cv),max(cvshift),'weight range',w.min(),w.max())
def parameterized(p):
 mats=[]
 for (R,P,O,h),(leak0,leak1,wrap) in zip(axes,np.reshape(p,(2,3))):
  d=np.zeros(R.shape[0]); d[0]=leak0/h; d[-1]=leak1/h
  mats.append(R+wrap*(P-R)+sparse.diags(d,format='csc'))
 return forward(*mats)
for fold in [None,0,1,2,3]:
 tr=full if fold is None else full[sensors%4!=fold]; te=full if fold is None else full[sensors%4==fold]
 sol=least_squares(lambda p:err(parameterized(p),tr,tr),np.zeros(6),bounds=([-0.2,-0.2,0]*2,[3,3,1]*2)); w,g=fit(parameterized(sol.x),tr)
 print('profile fold',fold,'wall parameters',sol.x.round(5),'test rms',np.sqrt(np.mean(err(parameterized(sol.x),tr,te)**2)), 'w range',w.min(),w.max(),'nuisance',g.round(5))
