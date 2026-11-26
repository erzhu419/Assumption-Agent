exec(open('/tmp/alpha_cv.py').read().split('for layout')[0])
from scipy.optimize import least_squares
for layout in (False, True):
    x=m._align(d,layout)[:n]; inp=m._inputs(tr,x,cal); law=m._Law(inp,dt,t); p=m._fit(law,y,rate)
    c,r=law.evaluate(p); print('layout',layout,'train abs median',np.median(abs(c-y)),np.median(abs(r-rate)))
    raw=tr['coupon_features_raw']; labs=tr['lab_id']; keep=np.isfinite(raw).all(1)
    p[4:]=0; c,r=law.evaluate(p)
    for lab in range(4):
        sel=(labs==lab)&keep
        X=inp[2][sel]; yy=y[sel]-c[sel]; rr=rate[sel]-r[sel]
        dc=np.vstack([(X[:,None,:]*t[None,:,None]).reshape(-1,5),X*3]); dz=np.r_[yy.ravel(),rr*3]
        q=least_squares(lambda w:dc@w-dz,np.zeros(5),loss='cauchy',f_scale=.025).x
        print(lab,'local drift',np.round(q,5),'local residual median',np.median(abs(dc@q-dz)))
