from chamber_probe import *
from chamber_probe import _chamber
from scipy.optimize import least_squares
mem,memrate=_chamber(tau,th,ha,drift*0,t)
E=cy-mem
print('drift',np.round(drift,3)); print('rate minus memory',np.round(ry-memrate,3)); print('by horizon [effective drift integral slope, median error, RMSE]:')
for j,z in enumerate(t):
 a=np.column_stack((drift,ha, np.ones(len(ha)))); coef=np.linalg.lstsq(a,E[:,j],rcond=None)[0]; err=E[:,j]-a@coef
 print(round(z,3),np.round(coef,5),round(np.median(abs(err)),5),round(np.sqrt(np.mean(err**2)),5))
for name,g in [('log',np.log1p(t)),('linear',t),('log+quad',np.log1p(t)+.1*t*t),('linear+quad',t+.1*t*t),('log+linear',.5*(np.log1p(t)+t))]:
 metrics(name,(mem+drift[:,None]*g,memrate+drift),q)
print('residual snapshots [t1, t6, t12] for archived log drift:'); print(np.round(np.column_stack((q['lab_id'],ha,drift,(E-drift[:,None]*np.log1p(t))[:,[0,5,-1]])),4))
