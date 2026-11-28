exec(open('/tmp/probe_walls.py').read().split('for walls in itertools.product')[0])
from scipy.linalg import lstsq
one = np.ones((1,8)); center = np.full(8,s['total_budget']/8); N = np.linalg.qr(one.T,mode='complete')[0][:,1:]
bank = s['row_bank_id']; xy = s['sensor_ij']; print('sensor boundary counts', ((xy == 0) | (xy == np.array([s['nx']-1,s['ny']-1]))).sum(0)); print('nuisance variation',np.linalg.svd(z,compute_uv=False)); print('endpoint velocities',s['vel_x'][[0,-1]],s['vel_y'][[0,-1]])

def scores(S, g, ridge):
 vals=[]
 for j in np.unique(g):
  train=g!=j
  D=np.column_stack((S@N,z)); target=y-S@center
  D2=np.vstack((D[train], np.diag(np.sqrt([ridge]*7+[s['nuisance_ridge']]*z.shape[1]))))
  co=lstsq(D2,np.r_[target[train],np.zeros(D.shape[1])])[0]; w=center+N@co[:7]
  if w.min() < 0 or w.max() > s['u_box']:
   X=S[train]; Z=z[train]; Y=y[train]; gram=np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(z.shape[1]),Z.T); T=X-Z@(gram@X); R=Y-Z@(gram@Y); w,_=prior._box_simplex(T.T@T+ridge*np.eye(S.shape[1]),-T.T@R,s['u_box'],s['total_budget']); b=gram@(Y-X@w)
  else: b=co[7:]
  vals.append(np.sqrt(np.mean((y[~train]-S[~train]@w-z[~train]@b)**2)))
 return np.r_[np.sqrt(np.mean(np.square(vals))),max(vals)]
for walls in itertools.product(('sealed','periodic'),repeat=2):
 S,_=prior._forward(s,walls)
 for ridge in (0.0,0.001,0.01,0.04):
  print(walls,ridge,'rosette',scores(S,groups,ridge).round(6),'shift',scores(S,shifts,ridge).round(6),'bank',scores(S,bank,ridge).round(6))
