import pickle, numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; print('top',s.keys()); s=s.get('study',s); np.set_printoptions(precision=4,suppress=True,linewidth=140)
def basis(x,d):return np.column_stack([np.ones(len(x))]+[x[:,ix].prod(1) for j in range(1,d+1) for ix in combinations_with_replacement(range(x.shape[1]),j)])
for k in s:
 if k.startswith('panel'):
  p=s[k]; X=p['u']; print(k)
  for v,Y in p.items():
   if v=='u':continue
   for Z in [X]+([p['v1']] if v=='v2' else []):
    for d in (1,2):
     A=basis(Z,d); w=np.linalg.lstsq(A,Y,rcond=None)[0];
     if k=='panel_a':w=least_squares(lambda W:(A@W.reshape(w.shape)-Y).ravel(),w.ravel(),loss='cauchy',f_scale=.03).x.reshape(w.shape)
     E=Y-A@w; print(v,'from', 'u' if Z is X else 'v1',d,'med/mx',np.median(abs(E),axis=0),np.max(abs(E),axis=0)); print(w if d==1 or k!='panel_a' else '')
tr=s['train']; x,y,T=tr['x'],tr['y'],tr['traj']; print('traj singular',np.linalg.svd(T,compute_uv=False)); print('metadata',[(k,v) for k,v in s.items() if k not in ('train','test','panel_a','panel_b','panel_c')]); print('correlation',np.corrcoef(np.c_[x,y,T].T).round(2)[:5]);
