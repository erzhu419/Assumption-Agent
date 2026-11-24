import pickle, sys
import numpy as np
sys.path.insert(0,'/holdout_task')
from thermoacoustic_recon import reconstruct_two_absorbers_mm as run
public=pickle.load(open('/holdout_task/public_tat.pkl','rb'))
rng=np.random.default_rng(615)

def check(s,c,q):
 out=run(s)
 z=np.asarray(out['absorbers_xy_mm']); d=np.linalg.norm(z[:,None]-q,axis=-1)
 assert set(out)=={'sound_speed_m_per_s','absorbers_xy_mm'} and type(out['sound_speed_m_per_s']) is float
 assert z.shape==(2,2) and np.isfinite(z).all() and z[0,0]<=z[1,0]
 return np.array([abs(out['sound_speed_m_per_s']-c), max(d.min(0).max(),d.min(1).max())]),z

for e in public:
 g=e['gt']; err,_=check(e['scan'],g['sound_speed_m_per_s'],np.array(g['absorbers_xy_mm'])); print('public',err)
s0=public[0]['scan']; xy=s0['sensor_xy_mm']; pin=s0['calib_pin_xy_mm']; fs=s0['fs_hz']; t=np.arange(len(s0['signals']))[:,None,None]; print('pin',pin)
results={}
for name in ['general','boundary','pin_overlap','close_pair','weaker_second','width_shift']:
 for i in range(12):
  c=[1435.,1545.,1490.,rng.uniform(1435,1545)][i%4]
  th=rng.uniform(0,2*np.pi,2); r=31.9*np.sqrt(rng.random(2)); q=r[:,None]*np.c_[np.cos(th),np.sin(th)]
  u=np.array([np.cos(th[0]),np.sin(th[0])]); width=4.4; weak=.75
  if name=='general' and np.linalg.norm(q[0]-q[1])<4: q[1]=-q[0]
  if name=='boundary': q[0]=31.95*u
  if name=='pin_overlap': q[0]=pin+[.15,.35,.7,1.5][i%4]*u; q[1]=-pin/2
  if name=='close_pair': q[0]=q[0]*.7; q[1]=q[0]+[.6,1.,1.8,3.][i%4]*u
  if name=='weaker_second': weak=.25
  if name=='width_shift': width=[3.4,5.5][i%2]
  d=np.linalg.norm(xy[:,None]-np.vstack((q,pin)),axis=-1); center=d*fs/1000/c
  gain=rng.uniform(.70,1.35,len(xy)); gain[rng.choice(len(xy),2,False)]*=-rng.uniform(.75,1.65,2)
  amp=gain[:,None]*np.array([4.6,4.6*weak,2.1])[None,:]*(45/d)**.7
  u0=(t-center)/width; sig=(-u0*np.exp(-.5*u0*u0)*amp).sum(axis=-1)+rng.normal(0,.08,t.shape[:1]+(len(xy),))
  err,z=check({**s0,'signals':sig},c,q); results.setdefault(name,[]).append(err)
  if np.any(err>[2.5,.85]): print('miss',name,i,'error',err,'true',q.round(2).tolist(),'fitted',z.round(2).tolist())
for name,errs in results.items(): print(name,'worst',np.max(errs,axis=0),'fails',np.sum(np.any(np.array(errs)>[2.5,.85],axis=1)))
