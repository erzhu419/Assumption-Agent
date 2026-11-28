from readout_probe import *
from scipy.spatial.distance import cdist
p,pr,y,tr=next(batches()); x=s['train']['x'][:52]; slope,offset=node_00._calibration(s['panel_a']); print('calibration:',slope, offset)
for a,b in [(p['st'][:,:4],x),(p['sy'],x),(p['scores'],x)]:
 print('cross-correlations:\n',np.round(np.corrcoef(a.T,b.T)[:len(a.T),len(a.T):],3))
f=np.c_[np.ones(len(x)),node_00._quadratic(x)]; y=s['train']['y'][:52]; print('oracle polynomial quadratic rms',np.std(y-f@np.linalg.lstsq(f,y,rcond=None)[0])); print('scalar quantiles',np.quantile(y,[0,.1,.5,.9,1])); print('variance in state modes',np.var(p['scores'],axis=0))
