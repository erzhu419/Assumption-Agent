import sys, pickle
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, '/holdout_task')
from upstream_transducer import _fit_bridge, _forward, _inverse

def shared(cal, rational=False):
    b = _fit_bridge(cal)
    x, y = cal['true_channel_loads'], cal['bridge_raw']
    s = cal['single_channel_sweeps']; j = np.arange(4); k = (j+1)%4; p = (j-1)%4
    ls = s['lab_id']; xs = np.eye(4)[ls]*s['true_stress'][:,None]; ys = s['cmd_stress']
    r = np.max(abs(_forward(x,b)-y),axis=1); rs = abs(_forward(xs,b)[np.arange(len(ls)),ls]-ys)
    scale = max(np.median(np.r_[r,rs]), 1e-10)
    a = r < 8*scale; t = rs < 8*scale
    def coeff(v):
        c = np.zeros((4,7)); g = v[4:8]; c[:,0] = v[:4]
        c[j,j+1] = g; c[j,k+1] = g*v[8]; c[j,p+1] = g*v[9]; c[:,5:] = g[:,None]*v[10:12]
        return c
    def predict(z,v):
        c = coeff(v); q = _forward(z,c)
        if rational:
            term = c[:,6]*z*np.roll(z,-1,axis=1)
            denom = 1+v[12]*(z-np.roll(z,-1,axis=1))**2
            q += term*(1/denom-1)
        return q
    g = b[j,j+1]; v = np.r_[b[:,0],g,np.median(b[j,k+1]/g),np.median(b[j,p+1]/g),np.median(b[:,5:]/g[:,None],axis=0)]
    if rational: v = np.r_[v,0.]
    def residual(v):
        return np.r_[(predict(x[a],v)-y[a]).ravel(),predict(xs[t],v)[np.arange(t.sum()),ls[t]]-ys[t]]
    lim = (np.r_[np.full(12,-np.inf),-0.02], np.full(13,np.inf)) if rational else (-np.inf,np.inf)
    v = least_squares(residual,v,bounds=lim).x
    return coeff(v), lambda z: predict(z,v), v[8:]

for item in pickle.load(open('/holdout_task/public_batch.pkl','rb')):
    d=item['dataset']; c=d['calibration_load_cells']; x,y=c['true_channel_loads'],c['bridge_raw']; b=_fit_bridge(c)
    clean=np.max(abs(_forward(x,b)-y),axis=1)<1; axis=np.count_nonzero(x,axis=1)<=1
    boundary=(~axis)&(np.max(abs(x),axis=1)>1.4)
    print('section keys',sorted(d), 'shared ratios',shared(c)[2], 'rational ratios and curvature',shared(c,True)[2])
    errors={}
    for i in np.flatnonzero(clean):
        keep=np.arange(len(x))!=i; cal=dict(c,true_channel_loads=x[keep],bridge_raw=y[keep]); a=_fit_bridge(cal); h, f, _=shared(cal); _, g, _=shared(cal,True)
        errors[i]=[np.max(abs(_forward(x[i:i+1],a)-y[i])),np.max(abs(f(x[i:i+1])-y[i])),np.max(abs(g(x[i:i+1])-y[i])),np.max(abs(_inverse(y[i],a)-x[i])),np.max(abs(_inverse(y[i],h)-x[i]))]
    for name,m in [('axis',axis),('mixed',~axis),('boundary',boundary)]:
        e=np.array([v for i,v in errors.items() if m[i]])
        print(name,'n',len(e),'RMSE',np.sqrt(np.mean(e**2,axis=0)),'MAX',e.max(axis=0),'columns: forward sparse/shared/rational; inverse sparse/shared')
