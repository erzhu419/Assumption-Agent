from scalar_probe import *
from scipy.spatial.distance import cdist
from scipy.optimize import brentq
from probe_metric import fit
p,state,y,v,t,x,q=fold(np.arange(52),np.arange(52,72)); sl,off=node_00._calibration(s['panel_a']); print('calibration',sl,off); K,Kq=p['K'],p['Kq']; print('kernel diagonal range',K.diagonal().min(),K.diagonal().max());
D=cdist(x/sl,x/sl,'sqeuclidean'); E=cdist(x,x,'sqeuclidean'); print('range',D.max(),E.max());
for label,A in [('K',K),('logK',-np.log(K/K.diagonal()[0])),('exp residual',K-1.5*np.exp(-D/6)),('raw dot',x@x.T)]:
 print(label,A.min(),A.max(),np.linalg.svd(A,compute_uv=False)[:9]);
