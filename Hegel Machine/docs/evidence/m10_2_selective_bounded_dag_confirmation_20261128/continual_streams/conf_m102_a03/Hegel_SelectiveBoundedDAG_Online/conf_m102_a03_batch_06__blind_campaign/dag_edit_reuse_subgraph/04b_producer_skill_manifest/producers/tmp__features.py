import numpy as np,pickle,itertools,glob,runpy
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a=runpy.run_path(glob.glob('/tmp/hegel_continual_archive/*01_solver.py')[0]); z=a['_inverse'](s['train']['x'],a['_calibration'](s['panel_a'])); y=s['train']['y']; Y=s['train']['traj'];
terms=[()]+[ix for k in range(1,4) for ix in itertools.combinations_with_replacement(range(4),k) if k<3 or len(set(ix))==1];
def design(z,terms=terms):return np.column_stack([np.prod(z[:,ix],axis=1) for ix in terms])
def fit(A,b):return np.linalg.lstsq(A,b,rcond=1e-10)[0]
def cv(A,b):
 h=np.diag(A@np.linalg.pinv(A)); return np.mean(abs((A@fit(A,b)-b)/(1-h).reshape((-1,)+(1,)*(b.ndim-1))))
def panel(u,v):
 A=design(u); B=fit(A,v)
 for j in range(4):
  print('col',j,' '.join(f'{ix}:{coef:.3f}' for ix,coef in zip(terms,B[:,j]) if abs(coef)>.007))
 print('max error',abs(A@B-v).max(), 'full loo',cv(A,v));return B
maps=[]
for p,k in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 print(p,k); maps.append(design(z)@panel(s[p]['u'],s[p][k]))
L=design(z)[:,:5]; V=np.column_stack(maps); 
for mask in range(1,16):
 A=np.column_stack([f for j,f in enumerate([L]+maps) if mask>>j&1]); print('mask',mask,cv(A,y),cv(A,Y))
print('combined nonlinear');
for f in [*maps,z,np.column_stack(maps)]:
 A=np.column_stack([np.ones(len(z)),f]); print('each',cv(A,y),cv(A,Y));
 if f.shape[1]==4:
  for d in (2,3):
   A=np.column_stack([np.prod(f[:,ix],axis=1) for k in range(d+1) for ix in itertools.combinations_with_replacement(range(4),k)]); print(d,cv(A,y),cv(A,Y))
