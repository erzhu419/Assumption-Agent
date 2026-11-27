exec(open('/tmp/compare_chamber.py').read().split('for x,b in itertools.product')[0])
logs=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))
for rec in logs:
 a=np.array(rec['facility_telemetry_trace']); f=np.fft.rfft(a-a.mean(axis=0),axis=0); rows=[]
 for i in range(a.shape[1]):
  k=np.argmax(abs(f[:,i])); rows.append((int(k),round(float(abs(f[k,i])),2),round(float(np.angle(f[k,i])),2)))
 print(rec['completed_campaign_index'],rows)
aux=st['aux_manifold_telemetry']; print('production aux min/mean/max/std',*[np.round(z,4) for z in (aux.min(0),aux.mean(0),aux.max(0),aux.std(0))]); print('production aux trace first/last',aux[[0,-1]])

def prediction(S,tr,rho):
 A=S[tr]; B=Z[tr]; yy=y[tr]; C=np.linalg.lstsq(B,np.column_stack((A,yy)),rcond=None)[0]; Q=A-B@C[:,:-1]; t=yy-B@C[:,-1]; w,_=old._box_simplex(Q.T@Q+rho*np.eye(A.shape[1]),-Q.T@t,cap,total); return S@w+Z@(C[:,-1]-C[:,:-1]@w)

def build(x,b):
 L=(kron(eye(ny),axis('x',nx,x))+kron(axis('y',ny,b),eye(nx))+diags(s['V_reaction_2d'].ravel(order='F'))).tocsc(); return np.vstack([splu(L+mu*I).solve(rhs)[pos] for mu in s['mu_shifts']])
models=['SS','SP','PS','PP']; matrices={m:build(*m) for m in models}
for rho in (0.002,0.01,0.04):
 for groups in (np.arange(nr)//ns,np.tile(np.arange(ns)//4,nr//ns),s['row_bank_id']):
  errors={}
  for m,S in matrices.items():
   errors[m]=np.array([rmse((prediction(S,groups!=g,rho)-y)[groups==g]) for g in np.unique(groups)])
  print('ridge/group',rho,len(np.unique(groups)),{m:np.round(err,5).tolist() for m,err in errors.items()})
