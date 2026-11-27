import pickle,numpy as np
from itertools import combinations_with_replacement
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=D['calibration_load_cells']; x,y=c['bridge_raw'],c['true_channel_loads']; print('bounds', np.round([x.min(0),x.max(0),y.min(0),y.max(0)],3)); print('first raw/true',np.round(np.c_[x,y][:10],3)); print('aff forward',np.round(np.linalg.lstsq(np.c_[np.ones(len(y)),y],x,rcond=None)[0],3));
for a,b in [(x,y),(y,x)]:
 for mode in ['linear','diag2','diag3','diag5','full2','full3']:
  z=[np.ones(len(a)),*a.T]; deg=1 if mode=='linear' else int(mode[-1]);
  for j in range(2,deg+1):
   z+=list((a**j).T) if mode.startswith('diag') else [a[:,I].prod(1) for I in combinations_with_replacement(range(4),j)]
  z=np.array(z).T; pred=[]; rng=np.random.default_rng(123)
  for k in np.array_split(rng.permutation(len(z)),8):
   m=~np.isin(np.arange(len(z)),k); pred.extend(z[k]@np.linalg.lstsq(z[m],b[m],rcond=None)[0]-b[k]);
  e=np.array(pred); print('inverse' if a is x else 'forward',mode,'channel',np.round(np.sqrt(np.mean(e**2,0)),5),'total',round(np.sqrt(np.mean(e.sum(1)**2)),5));
for k,v in c['single_channel_sweeps'].items(): print(k,v.shape,np.round(v[:14],3));
for b in ('train','test','chamber_verification_trials'):
 a=np.concatenate([D[b]['past_channel_raw'].reshape(-1,4),D[b]['hold_channel_raw']]);print(b,'raw min/max',np.round([a.min(0),a.max(0)],3));
