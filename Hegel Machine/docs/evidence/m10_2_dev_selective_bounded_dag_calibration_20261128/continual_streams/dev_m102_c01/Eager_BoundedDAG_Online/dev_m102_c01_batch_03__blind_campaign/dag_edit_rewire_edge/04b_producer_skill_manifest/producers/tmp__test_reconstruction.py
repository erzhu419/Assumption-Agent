from scalar_probe import *
from scipy.linalg import orthogonal_procrustes
import importlib.util
sp=importlib.util.spec_from_file_location('candidate','/holdout_task/node_01.py'); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
for va in [np.arange(52,72),np.arange(20),*[np.arange(k,72,3) for k in range(3)]]:
 tr=np.setdiff1d(np.arange(72),va); p,state,y,v,t,x,q=fold(tr,va); xh,qh=m._coordinates(p['K'],p['Kq'],4); calc=orthogonal_procrustes(xh,x)[0]; pred=m._predict(xh,qh,p['scores'],state@p['modes'].T,y); print(va[:3],np.max(abs(xh@calc-x)),np.max(abs(qh@calc-q)),np.mean(abs(pred-v)),np.mean(abs(t-state)),flush=True)
