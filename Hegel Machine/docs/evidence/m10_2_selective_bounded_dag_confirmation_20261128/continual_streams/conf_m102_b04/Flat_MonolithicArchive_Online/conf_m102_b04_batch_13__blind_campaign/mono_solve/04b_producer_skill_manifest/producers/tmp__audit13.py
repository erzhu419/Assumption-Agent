import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares,minimize_scalar
from itertools import combinations_with_replacement
p='/holdout_task/'
spec=importlib.util.spec_from_file_location('solver',p+'batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; y=tr['y']; T=tr['traj']; t=s['eval_times']; print('observed fields:',sorted(s),sorted(tr)); print('telemetry has one study-level value per component:',s['telemetry_vec'].shape)
u,v=[s['panel_a'][k] for k in ('u','v')]; gain_dist=np.linalg.norm(v-u,axis=1); med=np.median(gain_dist); retained=np.flatnonzero(gain_dist<med+8*np.median(abs(gain_dist-med)))
for kind in ('sine_only','tanh','rational'):
 def design(z,p):
  eye=np.broadcast_to(np.eye(4),(len(z),4,4)); base=[eye,eye*z[...,None],np.sin(p[-1]*np.roll(z,-2,1))[...,None]]
  if kind=='tanh': base.append(np.tanh(p[0]*z)[...,None])
  if kind=='rational': base.append((z/(1+p[0]*z*z))[...,None])
  return np.concatenate(base,axis=-1)
 errors=[]
 for j in retained:
  i=retained[retained!=j]; target=v[i].ravel()
  def fit(p):
   a=design(u[i],p).reshape(4*len(i),-1); w=m._ls(a,target); return a@w-target,w
  ini={'sine_only':[2.4],'tanh':[1.8,2.4],'rational':[1.4,2.4]}[kind]; opts={'sine_only':([1.4],[3.5]),'tanh':([.5,1.4],[4,3.5]),'rational':([.05,1.4],[6,3.5])}[kind]
  q=least_squares(lambda q:fit(q)[0],ini,bounds=opts,gtol=1e-12).x
  errors.extend((design(u[j:j+1],q)@fit(q)[1]-v[j]).ravel())
 print('leave-one-inlier-out calibration',kind,'MAE',np.mean(np.abs(errors)),'max',np.max(np.abs(errors)))
cal=m._calibration(s['panel_a']); h=m._operator(s['panel_b'])(m._inverse(x,cal)); B=m._affine(h); S=m._scalar_design(h); a,b,c,d=h.T; q0=a*c+b*d; q1=a*c-b*d
Q=np.column_stack([B]+[h[:,i]*h[:,j] for i,j in combinations_with_replacement(range(4),2)])
# Explicit pairwise sum/difference squares and three-way products augment the sparse symmetric readout.
D=np.c_[S,(a-c)**2+(b-d)**2,(a+c)**2-(b+d)**2,(a-c)**2-(b-d)**2,q1]
C=np.c_[D,a*b*c,a*b*d,a*c*d,b*c*d]
assert np.allclose(4*q0,(a+c)**2+(b+d)**2-(a-c)**2-(b-d)**2)
rng=np.random.default_rng(928470); splits=[np.arange(52,72),np.argsort(np.linalg.norm(h-h.mean(0),axis=1))[-20:],np.argsort(abs(y))[-20:],np.argsort(abs(T[:,-1]))[-20:]]+[rng.permutation(len(y))[:20] for _ in range(24)]
rs={k:[] for k in ('affine','symmetric','pair differences','pair plus triples','full quadratic','rational','one rate','two rates','timewise affine')}
for j in splits:
 i=np.setdiff1d(np.arange(len(y)),j)
 for k,M in [('affine',B),('symmetric',S),('pair differences',D),('pair plus triples',C),('full quadratic',Q)]:
  rs[k].append(np.mean(abs(M[j]@m._ls(M[i],y[i])-y[j])))
 def rational(theta):
  n,z=theta; return S/(1+n*(q0 if z==0 else q1)[:,None]**2)
 def loss(theta):
  r=rational(theta)[i]; return np.mean((r@m._ls(r,y[i])-y[i])**2)
 theta=min([(0.,0)]+[(minimize_scalar(lambda n:loss((n,z)),bounds=(0.,1.),method='bounded').x,z) for z in (0,1)],key=loss)
 R=rational(theta); rs['rational'].append(np.mean(abs(R[j]@m._ls(R[i],y[i])-y[j])))
 beta=m._ls(B[i],T[i]); fitted=B[i]@beta
 def one(r):
  F=(-np.expm1(-r*t))[:,None]; return m._ls(F,fitted.T).T@F.T
 r=minimize_scalar(lambda r:np.mean((one(r)-fitted)**2),bounds=(.01,8),method='bounded').x
 for k,P in [('one rate',B[j]@m._ls(B[i],one(r))),('two rates',m._trajectory(h[i],h[j],t,T[i])),('timewise affine',B[j]@beta)]:
  e=abs(P-T[j]); rs[k].append((e.mean(),e.mean(0).max()))
for k,E in rs.items(): print(k,'prescribed',np.round(E[0],7),'mean',np.round(np.mean(E,axis=0),7),'worst',np.round(np.max(E,axis=0),7))
assert np.max(rs['symmetric'])<=.038 and np.max(rs['two rates'])<=.038
