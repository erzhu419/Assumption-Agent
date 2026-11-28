import numpy as np, pickle, json, importlib.util
from scipy import sparse
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('old','/tmp/hegel_continual_archive/dev_beta_m100_01_batch_01_solver.py'); old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
it=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0];s=it['study'];p=it['campaign_diagnostics'];W=np.stack([e['w_pulse'] for e in p],axis=1);Y=np.stack([old._calibrate(dict(s,y_obs_raw=e['y_pulse_raw'])) for e in p],axis=1)

def op(q):
 out=[]
 for k,ab in enumerate('xy'):
  kap=s['kappa_'+ab];n=len(kap);h=s['L'+ab]/(n-1);A=old._axis(kap,s['vel_'+ab],h).tolil();a,b,c,d=q[4*k:4*k+4];u=a*kap[0]/h**2;v=b*kap[-1]/h**2
  A[0,0]=u+c*s['robin_'+ab]/h;A[0,1]=-u;A[-1,-1]=v+d*s['robin_'+ab]/h;A[-1,-2]=-v;out.append(A.tocsc())
 ax,ay=out;return sparse.kron(sparse.eye(s['ny']),ax,format='csc')+sparse.kron(ay,sparse.eye(s['nx']),format='csc')+sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc')

def resid(q):return old._forward(s,op(q))[0]@W-Y
for name,q in [('open',[1,1,1,1]*2),('reflect',[1,1,0,0]*2),('double reflect',[2,2,0,0]*2)]:print(name,np.sqrt(np.mean(resid(q)**2,axis=0)))
for omit in [-1,0,1,2]:
 keep=np.arange(W.shape[1])!=omit
 opt=least_squares(lambda q:resid(q)[:,keep].ravel(),[1,1,.3,.3]*2,bounds=(0,10),gtol=1e-13,ftol=1e-13,xtol=1e-13);print('omit',omit,'q',opt.x,'rmse',np.sqrt(np.mean(resid(opt.x)**2,axis=0)))
for e in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 a=np.asarray(e['facility_telemetry_trace']);print(e['completed_campaign_short_name'], 'mean',a.mean(0),'std',a.std(0))
print('aux first rows',s['aux_manifold_telemetry'][:5]);c=s['sensor_calibration_sweeps'];r=c['ref_rosette'];y=c['raw_rosette'];b=c['single_bank_sweeps'];
for i in range(4):
 m=b['bank_id']==i;t=np.r_[r[:,i],b['ref_signal'][m]];z=np.r_[y[:,i],b['raw_readout'][m]];print('bank',i,np.polyfit(t,z,1),np.std(z-np.polyval(np.polyfit(t,z,1),t)))
