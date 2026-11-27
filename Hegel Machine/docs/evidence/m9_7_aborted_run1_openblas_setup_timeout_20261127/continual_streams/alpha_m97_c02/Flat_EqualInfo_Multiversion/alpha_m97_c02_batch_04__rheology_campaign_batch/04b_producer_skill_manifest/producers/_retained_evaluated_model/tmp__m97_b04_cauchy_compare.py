exec(open('/tmp/m97_b04_cv.py').read().split('v, w =')[0])
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('robust','/tmp/m97_b04_robust.py')
b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
missing=np.isnan(tr['coupon_features_raw']).any(axis=1)
def refit(f, initial, loss, omit_imputed=False):
    if omit_imputed: f=f & ~missing
    def residual(pv):
        cy,cr=m._design(pv[:2],p[f],h[f],x[f],dt,t,a[f])
        return np.r_[(cy@pv[2:]-y[f]).ravel(),cr@pv[2:]-r[f]]
    e=residual(initial); cut=f.sum()*len(t)
    def mad(v): return max(1.e-7,1.4826*np.median(np.abs(v-np.median(v))))
    scale=np.r_[np.full(cut,mad(e[:cut])),np.full(f.sum(),mad(e[cut:]))]
    fit=least_squares(lambda pv: residual(pv)/scale,initial,loss=loss,f_scale=2.5,x_scale='jac',max_nfev=100,bounds=(np.r_[np.log([.02,.04]),np.full(9,-np.inf)],np.r_[np.log([40.,100.]),np.full(9,np.inf)]))
    return fit.x
names=['soft_l1','cauchy','cauchy_observed_features']
res={j:[np.zeros_like(y),np.zeros_like(r)] for j in names}
for lab in np.unique(tr['lab_id']):
    q=tr['lab_id']==lab; f=~q
    lt,w=b._fit(p[f],h[f],x[f],dt,t,a[f],y[f],r[f]); start=np.r_[lt,w]
    for j,pv in zip(names,(start,refit(f,start,'cauchy'),refit(f,start,'cauchy',True))):
        cy,cr=m._design(pv[:2],p[q],h[q],x[q],dt,t,a[q]);res[j][0][q]=cy@pv[2:]-y[q];res[j][1][q]=cr@pv[2:]-r[q]
    e,z=res['cauchy']; print('held lab',lab,'curve errors > 1 per horizon',np.sum(np.abs(e[q])>1,axis=0),'rate errors > .1',np.sum(np.abs(z[q])>.1))
for j,(e,z) in res.items():
    print(j)
    for name,q in [(f'lab {k}',tr['lab_id']==k) for k in range(4)]+[('imputed',missing),('observed features',~missing),('zero damage',a==0),('positive damage',a>0)]:
        print(name,'raw rmse',report(e[q],z[q]),'median ae',np.round([np.median(np.abs(e[q])),np.median(np.abs(z[q]))],6),'90% ae',np.round([np.quantile(np.abs(e[q]),.9),np.quantile(np.abs(z[q]),.9)],6))
    print('maximum horizon median ae',np.max(np.median(np.abs(e),axis=0)))
