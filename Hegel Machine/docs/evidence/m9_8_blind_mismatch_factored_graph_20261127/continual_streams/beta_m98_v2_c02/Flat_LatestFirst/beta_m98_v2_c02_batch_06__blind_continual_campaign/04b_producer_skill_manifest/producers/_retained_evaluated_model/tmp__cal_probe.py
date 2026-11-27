import glob, importlib.util, pickle, itertools
import numpy as np
from scipy.linalg import lstsq
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
s=item['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; solo=c['single_bank_sweeps']
def load(tag):
 p=glob.glob('/tmp/hegel_continual_archive/*'+tag+'_solver.py')[0]; sp=importlib.util.spec_from_file_location('a'+tag,p); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m

def robust(A,z):
 w=np.ones(len(A)); q=np.linalg.lstsq(A,z,rcond=None)[0]
 for _ in range(45):
  e=z-A@q; scale=max(3e-5, .8*np.median(abs(e))); w=1/np.hypot(1,e/scale)**2
  q=lstsq(A*w[:,None]**.5,z*w**.5,lapack_driver='gelsy')[0]
 return q,z-A@q

def bases(v,j):
 a,b,d,e=v[:,[(j+i)%4 for i in (0,-1,1,2)]].T
 B={'1':np.ones(len(a)),'a':a,'b':b,'c':d,'d':e,'ac':a*d,'a3':a**3}
 for k,h in list(B.items())[1:5]:
  for n in (2,3,4,5): B[k+str(n)]=h**n
  for l,g in list(B.items())[1:5]:
   if k<l: B[k+l]=h*g
   if k!=l: B[k+'2'+l]=h*h*g
 for i,j,k in itertools.combinations('abcd',3): B[i+j+k]=B[i]*B[j]*B[k]
 return B

if __name__=='__main__':
 m=load('batch_04')
 for j in range(4):
  mask=solo['bank_id']==j; X=np.vstack([x,np.eye(4)[j]*solo['ref_signal'][mask,None]]); z=np.r_[y[:,j],solo['raw_readout'][mask]]; B=bases(X,j)
  for name,terms in [('affine','1 a b c d'.split()),('prior','1 a b c ac a3'.split()),('quadratic',['1']+[k for k in B if len(k)==1 or len(k)==2 and not k.endswith(tuple('345'))]),('cubic',[k for k in B if not k.endswith(tuple('45'))]),('all',list(B))]:
   q,r=robust(np.column_stack([B[k] for k in terms]),z); print(j,name,np.quantile(abs(r),[.5,.8,.9,.95,1]).round(6),[(k,round(v,5)) for k,v in zip(terms,q) if abs(v)>0.005]);
  print('solo',np.c_[solo['ref_signal'][mask],z[-6:]].round(5))
 for tag in ['s00','batch_01','batch_02','batch_03','batch_04']:
  S,_=load(tag)._forward(s); print('wall',tag,'prediction norm by pulse and shift',[[round(np.linalg.norm(t),5) for t in np.split(S@p['w_pulse'],3)] for p in item['campaign_diagnostics']])
