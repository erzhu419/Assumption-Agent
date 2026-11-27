exec(open('/tmp/m97_b04_cv.py').read().split('v, w =')[0])
spec = importlib.util.spec_from_file_location('robust','/tmp/m97_b04_robust.py')
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
fits = [m._fit,b._fit]
res = [[np.zeros_like(y),np.zeros_like(r)] for _ in fits]
core_y = np.zeros_like(y,dtype=bool); core_r = np.zeros_like(r,dtype=bool)
for lab in np.unique(tr['lab_id']):
    q = tr['lab_id']==lab; f = ~q
    for i,fit in enumerate(fits):
        lt,w=fit(p[f],h[f],x[f],dt,t,a[f],y[f],r[f]); cy,cr=m._design(lt,p[q],h[q],x[q],dt,t,a[q]); res[i][0][q]=cy@w-y[q];res[i][1][q]=cr@w-r[q]
    cy,cr=m._design(lt,p[f],h[f],x[f],dt,t,a[f]); ey,er=cy@w-y[f],cr@w-r[f]
    def fence(e): return 5*1.4826*np.median(np.abs(e-np.median(e)))
    core_y[q]=np.abs(res[1][0][q]) < fence(ey); core_r[q]=np.abs(res[1][1][q]) < fence(er)
    print('fold',lab,'robust taus',np.exp(lt),'fences',fence(ey),fence(er),'curve outside',sum(~core_y[q].ravel()))
for label,q in [(f'lab {u}',tr['lab_id']==u) for u in np.unique(tr['lab_id'])]+[(f'stratum {u}',tr['stratum']==u) for u in np.unique(tr['stratum'])]+[('zero damage',a==0),('positive damage',a>0),('imputed',np.isnan(tr['coupon_features_raw']).any(axis=1)),('all',np.ones(n,dtype=bool))]:
    print(label,'baseline -> robust: raw RMSE',[report(v[q],w[q]) for v,w in res], 'median AE',[[round(np.median(np.abs(v[q])),6),round(np.median(np.abs(w[q])),6)] for v,w in res], 'same core RMSE',[report(v[q][core_y[q]],w[q][core_r[q]]) for v,w in res])
for fit in fits:
    lt,w=fit(p,h,x,dt,t,a,y,r); print('full',np.exp(lt),w)
print('curve residual abs percentiles', [np.percentile(np.abs(v),[50,90,95,98,99,100]).round(5) for v,w in res]); print('robust core horizon RMSE',[rms(res[1][0][:,i][core_y[:,i]]) for i in range(len(t))])
