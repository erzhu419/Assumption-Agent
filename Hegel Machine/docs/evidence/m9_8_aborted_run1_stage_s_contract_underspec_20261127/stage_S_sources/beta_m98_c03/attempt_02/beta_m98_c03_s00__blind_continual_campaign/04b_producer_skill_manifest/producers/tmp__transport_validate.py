from transport_probe import *
from scipy.optimize import least_squares
c=s['sensor_calibration_sweeps']; x=c['ref_rosette']; raw=c['raw_rosette']; n=len(x)
lib=[np.ones((n,1)),x]
for degree in (2,3):
    lib.append(np.column_stack([np.prod(x[:,j],axis=1) for j in itertools.combinations_with_replacement(range(4),degree)]))
fold=np.tile(np.arange(4),n//4)
for a in (0,2,3,4):
    P=np.column_stack(lib[:a]) if a else None
    e=raw-x
    if a:
        for k in range(4):
            tr=fold!=k
            e[~tr]-=P[~tr]@np.linalg.lstsq(P[tr],e[tr],rcond=None)[0]
    print('cal degree',a-1,'rms',np.sqrt(np.mean(e**2)),'worst bank',np.sqrt(np.mean(e**2,axis=0)).max(),'single / mixed',*[np.sqrt(np.mean(v**2)) for v in (e[:16],e[16:])])
B,E,hx,hy=init(1,'ar','face','velocity'); d0=np.repeat([rx/hx,ry/hy],2)
rope=[('nominal',np.zeros((4,0))),('axis scale',np.repeat(np.eye(2),2,axis=0)),('independent walls',np.eye(4))]
for name,W in rope:
    fitlist=[]; held=[]
    for train,test in [(0,1),(1,0)]:
        a=least_squares(lambda a:(predict(B,E@(d0+W@a))[:,train]-y[:,train]),np.zeros(W.shape[1]),gtol=1e-12).x if W.shape[1] else np.empty(0)
        dd=d0+W@a; fitlist.append(dd)
        e=(predict(B,E@dd)-y).reshape(3,-1,2)[:,:,test]
        held.append(list(np.sqrt(np.mean(e**2,axis=1))))
    print(name,'wall coefficients',np.array(fitlist),'held pulse-shift RMSE',np.array(held),'worst',np.max(held))
for k in ('robin_x','robin_y'):
 print(k,s[k])
print('source design',np.linalg.svd(np.concatenate([solve(B+np.diag(E@d0+mu),s['source_modes'].reshape(M,-1,order='F').T,check_finite=False)[ids] for mu in shifts]),compute_uv=False))
