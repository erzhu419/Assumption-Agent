import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
import alpha_w03_batch_08_solver as m
s=pickle.load(open('/holdout_task/public_alpha_w03_batch_08.pkl','rb'))[0]; d=s['dataset']; tr=d['train']; n=len(tr['lab_id']); x,h,p,l=m._prep(d); x,h,p,l=x[:n],h[:n],p[:n],l[:n]; dt,t,sy=(d[k] for k in ('past_dt','eval_times','yield_stress')); curves=tr['strain_curves_obs']-tr['strain_0'][:,None]; rates=tr['instant_rate_obs']; good=np.isfinite(tr['coupon_features_raw']).all(axis=1); cal,fixed=m._calibration_start(d['calibration_load_cells']); folds=np.zeros(n,int); rng=np.random.default_rng(13)
for lab in range(4):
    for valid in (False,True):
        ii=rng.permutation(np.flatnonzero((l==lab)&(good==valid))); folds[ii]=np.arange(len(ii))%5
pred=np.zeros((5,n,len(t))); rate=np.zeros((5,n)); target=np.zeros(n); diagnostics=[]
for fold in range(5):
    fit=(folds!=fold)&good; val=folds==fold
    tau,b,cf=m._fit(x[fit],h[fit],p[fit],l[fit],dt,t,sy,curves[fit],rates[fit],cal,fixed)
    hp,pp=m._physical(h,p,l,cf); c,r=m._design(tau,x,hp,pp,dt,t,sy); y=c@b; yr=r@b; resid=curves-y; late=t>=np.median(t); dy=np.median(resid[:,late]/t[late],axis=1); dr=rates-yr; target[val]=dy[val]; corrections=np.zeros((5,n)); corrections[1:,~good]=dr[~good]
    for lab in range(4):
        a=(l==lab)&fit; z=(l==lab)&good
        # Test whether reference scaling, rather than a constant lab drift,
        # explains coupon-to-coupon differences in the linear rate term.
        signal=x[:,1:]@b[3:7]
        mat=np.c_[np.ones(n),signal]
        rich=np.c_[x,hp,r[:,0],r[:,1],r[:,7],r[:,8]]
        def robust(A):
            w=np.ones(a.sum()); D=A[a]; v=dy[a]
            for _ in range(5):
                beta=np.linalg.lstsq(D*w[:,None],v*w,rcond=None)[0]; e=v-D@beta; scale=max(.001,1.4826*np.median(abs(e-np.median(e)))); w=1/np.hypot(1.,e/(3*scale))
            return A@beta
        scale_fit=robust(mat); corrections[2,z]=scale_fit[z]; corrections[3,z]=robust(rich)[z]
        e=dy[a]-scale_fit[a]; v=dr[a]-scale_fit[a]; w=(abs(v-np.median(v))<.07)&(abs(e)<.08); lam=np.clip(e[w]@v[w]/max(1e-12,v[w]@v[w]),0,1); corrections[4,z]=scale_fit[z]+lam*np.clip(dr[z]-scale_fit[z],-.07,.07)
        if fold==0:
            diagnostics.append((lab,np.linalg.lstsq(mat[a],dy[a],rcond=None)[0].round(5),np.linalg.lstsq(rich[a],dy[a],rcond=None)[0].round(5),round(lam,3)))
    corrections[0,~good]=dr[~good]
    pred[:,val]=y[val]+corrections[:,val,None]*t; rate[:,val]=yr[val]+corrections[:,val]
print('lab, intercept and covariate-signal slope, expanded effects, innovation weight:',diagnostics)
# Median projected late slope resists impulsive observation errors; also
# report pointwise robust strain and onset errors as independent checks.
for j,name in enumerate(('base','repeat base','reference scale','expanded','scale+innovation')):
    e=pred[j]-curves; er=rate[j]-rates; late=t>=np.median(t); el=np.median(e[:,late]/t[late],axis=1)
    print(name,[(label,int(z.sum()),*[round(v,6) for v in (np.sqrt(np.mean(el[z]**2)),np.median(abs(e[z])),np.median(abs(er[z])))]) for label,z in [(i,(l==i)&good) for i in range(4)]+[('missing',~good)]])
