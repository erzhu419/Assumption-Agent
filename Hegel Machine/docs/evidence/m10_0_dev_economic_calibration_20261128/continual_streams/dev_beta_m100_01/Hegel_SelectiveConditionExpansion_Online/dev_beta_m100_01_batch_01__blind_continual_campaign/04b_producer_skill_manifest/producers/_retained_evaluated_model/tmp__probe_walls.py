import pickle, importlib.util, numpy as np
from scipy import sparse
from scipy.optimize import least_squares
p='/tmp/hegel_continual_archive/dev_beta_m100_01_s00_solver.py'; spec=importlib.util.spec_from_file_location('prior',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=i['study']; ds=i['campaign_diagnostics']; w=np.stack([d['w_pulse'] for d in ds],axis=1)
c=s['sensor_calibration_sweeps']; x=c['ref_rosette']; y=c['raw_rosette']; gain=((x-x.mean(0))*(y-y.mean(0))).sum(0)/((x-x.mean(0))**2).sum(0); off=y.mean(0)-gain*x.mean(0)
obs=np.stack([((d['y_pulse_raw'].reshape(-1,4)-off)/gain).ravel() for d in ds],axis=1)
nx,ny=s['nx'],s['ny']; ix,iy=sparse.eye(nx),sparse.eye(ny)
axes=[]
for axis,n in [('x',nx),('y',ny)]:
 h=s['L'+axis]/(n-1); k=s['kappa_'+axis]; axes.append((m._axis(k,s['vel_'+axis],h,0),s['robin_'+axis]/h,k[[0,-1]]/h**2))
V=sparse.diags(s['V_reaction_2d'].ravel(order='F'))
def operator(q,d):
 out=[]
 for j,(a,r,k) in enumerate(axes):
  a=a.tolil(); n=a.shape[0]
  for v,b,ne in [(0,0,1),(1,n-1,n-2)]:
   row=2*j+v; a[b,ne]=-d[row]*k[v]; a[b,b]=d[row]*k[v]+q[row]*r
  out.append(a.tocsc())
 return (sparse.kron(iy,out[0])+sparse.kron(out[1],ix)+V).tocsc()
def pred(q,d=(1,1,1,1)):
 return m._forward(s,operator(q,d))[0]@w

def report(tag,e):
 z=e.reshape(3,-1,3); print(tag,'shift RMS',np.sqrt(np.mean(z*z,axis=(1,2))),'pulse RMS',np.sqrt(np.mean(e*e,axis=0)),'worst',np.max(np.abs(e)))
for b in [(1,1,1,1),(0,0,0,0),(1,1,0,0),(0,0,1,1)]: report(str(b),pred(b)-obs)
for mode in ['shared','axis','wall','diffusion','all']:
 def pars(p):
  if mode=='shared': return np.repeat(p,4),np.ones(4)
  if mode=='axis': return np.repeat(p,2),np.ones(4)
  if mode=='wall': return p,np.ones(4)
  if mode=='diffusion': return np.ones(4),p
  return p[:4],p[4:]
 start=np.ones(dict(shared=1,axis=2,wall=4,diffusion=4,all=8)[mode]); es=[]
 for j in [-1,0,1,2]:
  keep=np.arange(3)!=j
  fit=least_squares(lambda a:(pred(*pars(a))[:,keep]-obs[:,keep]).ravel(),start,bounds=(0,10),gtol=1e-13,ftol=1e-12,xtol=1e-12)
  e=pred(*pars(fit.x))-obs
  if j==-1: print('fit',mode,fit.x); report('full',e)
  else: es.append(e[:,j])
 report('LOPO '+mode,np.array(es).T)
