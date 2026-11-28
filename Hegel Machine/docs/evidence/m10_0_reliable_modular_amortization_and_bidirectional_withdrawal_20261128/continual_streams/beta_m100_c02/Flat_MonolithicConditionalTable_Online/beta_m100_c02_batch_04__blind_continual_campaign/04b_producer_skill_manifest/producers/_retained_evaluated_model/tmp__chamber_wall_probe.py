import sys,pickle,numpy as np
from scipy.optimize import least_squares
from scipy import sparse
sys.path.insert(0,'/tmp/hegel_continual_archive')
import beta_m100_c02_batch_03_solver as m
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; d=item['campaign_diagnostics']; W=np.stack([p['w_pulse'] for p in d],axis=1); Y=np.stack([m.calibrate(dict(s,y_obs_raw=p['y_pulse_raw'])) for p in d],axis=1)
base=np.repeat([s['robin_x'],s['robin_y']],2); orig=m.axis_operator
c=s['sensor_calibration_sweeps']; ref=c['ref_rosette']; off,gain,cub,l,r,cross=m.calibration_map(c); pred=off+gain*ref+cub*ref**3+l*np.roll(ref,1,axis=1)+r*np.roll(ref,-1,axis=1)+cross*ref*np.roll(ref,-1,axis=1); print('sweep coeff',np.round(np.array([off,gain,cub,l,r,cross]),4)); print('absolute sweep residual percentiles',np.quantile(np.abs(pred-c['raw_rosette']),[.5,.8,.9,.95,.99])); print('obs layout banks',s['row_bank_id'][:32]); print('Z pulse nuisance RMS',np.sqrt(np.mean((s['Z_nuisance']@np.linalg.lstsq(s['Z_nuisance'],Y,rcond=None)[0])**2,axis=0)))
def evaluate(loss, inward=(1,1,1,1), wrap=(0,0), periodic=False):
 def axis(k,v,h,losses):
  i=0 if len(k)==s['nx'] else 1; a=orig(k,v,h,losses).tolil(); n=len(k)
  for j,row,col,other in ((0,0,1,n-1),(1,n-1,n-2,0)):
   ii=2*i+j; f=inward[ii]*k[row]/h**2; q=wrap[i]*(k[0]+k[-1])/(2*h*h)
   if periodic:
    f=0.5*(k[row]+k[col])/h**2+max(v[row]*(1 if j else -1),0)/h
    q+=max(v[row]*(-1 if j else 1),0)/h
   a[row,col]=-f; a[row,other]=-q; a[row,row]=f+q+losses[j]/h
  return a.tocsc()
 m.axis_operator=axis
 return m.chamber_response(s,loss)[0]@W-Y
for label,loss,ins,wr,per in [('sealed',0*base,np.ones(4),[0,0],False),('periodic',0*base,np.ones(4),[1,1],True),('x wrap',0*base,np.ones(4),[1,0],False),('y wrap',0*base,np.ones(4),[0,1],False),('reflecting doubled',0*base,2*np.ones(4),[0,0],False)]:
 e=evaluate(loss,ins,wr,per).reshape(3,-1,3); print(label,'means',np.round(e.mean(1),4),'rms',np.round(np.sqrt((e**2).mean(1)),4))
for kind,n in [('loss',4),('wrap',2),('inward and wrap',6),('all',10)]:
 def parse(x):
  if kind=='loss': return (x,)
  if kind=='wrap': return (0*base,np.ones(4),x)
  if kind=='inward and wrap': return (0*base,1+x[:4],x[4:])
  return (x[:4],1+x[4:8],x[8:])
 for h in [-1,0,1,2]:
  tr=[i for i in range(3) if i!=h]; lo=([-0.12]*4 if kind in ('loss','all') else [])+([-0.95]*4 if kind in ('inward and wrap','all') else [])+([0,0] if kind!='loss' else []); x0=np.maximum(0,np.array(lo)+1e-4)
  res=least_squares(lambda x:evaluate(*parse(x))[:,tr].ravel(),x0,bounds=(lo,np.full(n,8.)),max_nfev=120); e=evaluate(*parse(res.x)).reshape(3,-1,3); print(kind,h,'params',np.round(res.x,5),'max shift rms by pulse',np.round(np.sqrt((e**2).mean(1)).max(0),6))
