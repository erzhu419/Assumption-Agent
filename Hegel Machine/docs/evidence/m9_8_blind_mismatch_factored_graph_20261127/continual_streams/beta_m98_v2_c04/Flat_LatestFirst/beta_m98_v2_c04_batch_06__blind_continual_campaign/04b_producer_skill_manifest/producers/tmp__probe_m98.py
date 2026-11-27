import pickle, itertools, numpy as np, json, runpy
from scipy.sparse import eye, kron, diags
from scipy.sparse.linalg import splu
it=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; st=it['study']; sw=st['sensor_calibration_sweeps']; ref=sw['ref_rosette']; raw=sw['raw_rosette']
root='/tmp/hegel_continual_archive/beta_m98_v2_c04_'; openax=runpy.run_path(root+'batch_04_solver.py')['_axis']; perax=runpy.run_path(root+'batch_03_solver.py')['_axis']
def basis(x,j,kind):
 a=x[:,j]; l=x[:,(j-1)%4]; r=x[:,(j+1)%4]; o=x[:,(j+2)%4]; P=[np.ones(len(a)),l,a,r]
 if kind=='linear': return np.column_stack(P)
 if kind=='full_linear': return np.column_stack(P+[o])
 if kind=='archive04': return np.column_stack(P+[a**3,a*r])
 if kind=='full_quad': return np.column_stack(P+[o]+[x[:,i]*x[:,k] for i,k in itertools.combinations_with_replacement(range(4),2)])
 if kind=='full_cubic': return np.column_stack([np.ones(len(a))]+[np.prod(x[:,inds],axis=1) for d in (1,2,3) for inds in itertools.combinations_with_replacement(range(4),d)])
 if kind=='own_poly': return np.column_stack(P+[o,a*a,a**3,a*l,a*r,a*o,a**5])
def fit(ref,raw,kind):
 C=[]
 for j in range(4):
  X=basis(ref,j,kind); y=raw[:,j]; b=np.linalg.lstsq(X,y,rcond=None)[0]
  for _ in range(25):
   e=y-X@b; m=max(1e-9,1.4826*np.median(abs(e-np.median(e)))); w=np.sqrt(np.minimum(1,1.5*m/np.maximum(abs(e),1e-15))); b=np.linalg.lstsq(X*w[:,None],y*w,rcond=None)[0]
  C.append(b)
 return C
def pred(x,c,k): return np.column_stack([basis(x,j,k)@c[j] for j in range(4)])
def forward(x,y):
 def ax(q,k):
  args=[st['kappa_'+q],st['vel_'+q],st['L'+q]]
  return perax(*args) if k=='periodic' else openax(*args,st['robin_'+q] if k=='robin' else 0.)
 nx,ny=st['nx'],st['ny']; op=kron(eye(ny),ax('x',x),format='csc')+kron(ax('y',y),eye(nx),format='csc')+diags(st['V_reaction_2d'].ravel(order='F'),format='csc'); a,b=st['sensor_ij'].T; i=a+nx*b; f=st['source_modes'].transpose(0,2,1).reshape(-1,nx*ny).T
 return np.vstack([splu(op+v*eye(nx*ny,format='csc')).solve(f)[i] for v in st['mu_shifts']])
if __name__=='__main__':
 print('sweep ranges',ref.min(0),ref.max(0),np.quantile(abs(raw),[.5,.9,.95,.98,1],axis=0)); print('single', {k:np.round(v[:8],4) for k,v in sw['single_bank_sweeps'].items()})
 for k in ['linear','full_linear','archive04','full_quad','own_poly','full_cubic']:
  c=fit(ref,raw,k); out=np.zeros_like(raw)
  for t in range(5):
   take=np.arange(len(ref))%5==t; out[take]=pred(ref[take],fit(ref[~take],raw[~take],k),k)
  print(k,'CV quantiles',np.round(np.quantile(abs(out-raw),[.5,.8,.9,.95]),7),'coeff',np.round(c,5) if k in ('archive04','own_poly','full_linear') else '')
  score=[]
  for x,y in itertools.product(['robin','sealed','periodic'],repeat=2):
   S=forward(x,y); errs=[]
   for p in it['campaign_diagnostics']:
    e=(pred((S@p['w_pulse']).reshape(-1,4),c,k).ravel()-p['y_pulse_raw']); errs.append(np.linalg.norm(e)/np.linalg.norm(p['y_pulse_raw']))
   score.append((max(errs),x,y,errs))
  print('pulse fits',sorted(score))
 for z in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
  a=np.asarray(z['facility_telemetry_trace']); print(z['completed_batch_index'],a.shape,np.round(a.mean(0),3),np.round(a.std(0),3))
