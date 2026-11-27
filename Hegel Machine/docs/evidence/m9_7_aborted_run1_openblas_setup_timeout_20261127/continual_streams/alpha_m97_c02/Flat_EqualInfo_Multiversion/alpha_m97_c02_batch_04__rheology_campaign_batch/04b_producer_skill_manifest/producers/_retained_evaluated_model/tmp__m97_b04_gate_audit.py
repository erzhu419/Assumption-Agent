exec(open('/tmp/m97_b04_cv.py').read().split('v, w =')[0])
from scipy.optimize import least_squares
print('gate settings',d['shear_banding_index'],d['crit_stress'])
print('in-batch calibration',d['calibration_load_cells'])
cp,ch,_,_=m._prepare(d)
excess=np.max(np.abs(cp),axis=1)-d['crit_stress']
print('minimum absolute history distance to damage boundary',np.min(np.abs(excess)))
actual=m._design

def alternative(kind):
    def design(lt,p,h,x,dt,t,a):
        c,r=actual(lt,p,h,x,dt,t,a)
        if kind=='linear':
            c[:,:,2:7]=x[:,None,:]*t[None,:,None]
        elif kind=='linear_zero_damage':
            zero=a==0
            c[zero,:,2:7]=x[zero,None,:]*t[None,:,None]
        elif kind=='no_crystallization':
            c[:,:,7:]=0; r[:,7:]=0
        return c,r
    return design

try:
    for kind in ('specified','linear','linear_zero_damage','no_crystallization'):
        m._design=actual if kind=='specified' else alternative(kind)
        stats=[]
        for lab in np.unique(tr['lab_id']):
            q=tr['lab_id']==lab; f=~q
            lt,b=m._fit(p[f],h[f],x[f],dt,t,a[f],y[f],r[f]); c,g=m._design(lt,p[q],h[q],x[q],dt,t,a[q])
            for sub in (a[q]==0,a[q]>0):
                if sub.any(): stats.append((np.median(np.abs((c@b-y[q])[sub])),np.median(np.abs((g@b-r[q])[sub]))))
        print(kind,'worst held-lab/damage median AE strain/rate',np.round(np.max(stats,axis=0),6))
finally:
    m._design=actual

out=m.solve_alpha_m97_c02_batch_04(d)
lt=np.log(out['estimated_taus']); b=np.r_[out['hereditary_weights'],out['regime_mode_weights']]; tau=np.exp(lt)
for shear in (.50,np.nextafter(.50,1.)):
    a0=m._activation(dict(d,shear_banding_index=shear),cp,dt)
    z=np.zeros((len(cp),2))
    for s,step in enumerate(dt):
        alpha=np.exp(-step/tau);z=alpha*z+(1-alpha)*cp[:,s,None]
    _,_,xx,_=m._prepare(d)
    gap=ch[:,None]-z
    drift=xx@b[2:7]+b[8]*a0*z[:,0]
    prony=gap*b[:2]; prony[:,0]+=b[7]*a0*ch
    expected=prony@(-np.expm1(-t[:,None]/tau)).T+drift[:,None]*np.log1p(t)
    c,g=m._design(lt,cp,ch,xx,dt,t,a0)
    assert np.allclose(c@b,expected,rtol=1e-12,atol=1e-12)
    assert np.allclose(g@b,prony@(1/tau)+drift,rtol=1e-12,atol=1e-12)
    assert bool(np.any(a0)) == (shear<=.50)
    print('shear gate',repr(shear),'active rows',np.count_nonzero(a0),'recurrence and initial-rate identity passed')
