exec(open('/tmp/check_rheology01.py').read().split("print('active'")[0])
lab=d['train']['lab_id']; miss=np.isnan(d['train']['coupon_features_raw']).any(1)
# Fit on all observations or on complete coupons, leaving each lab out in turn.
for complete_only in (False, True):
    scores=[]
    for held in range(4):
        train=(lab!=held) & (~miss if complete_only else True)
        test=lab==held
        tau,w=m._fit((*[v[:n][train] for v in args],dt,t),yy[train].ravel(),'all')
        err=m._basis(np.log(tau),*[v[:n][test] for v in args],dt,t,'all')[0]@w-yy[test]
        scores.append([np.median(abs(err[:,:-1])), np.median(abs(err[:,-1])), np.median(abs(err[~miss[test],:-1])), np.median(abs(err[~miss[test],-1]))])
    print('complete_only',complete_only,'LOLO per lab: curve, rate, complete curve, complete rate',np.round(scores,5).tolist())
    print('worst per metric',np.round(np.max(scores,axis=0),5))
# Exercise the specified archive branch and in-batch cleaning independently.
cal=d['calibration_load_cells']; baseline=m._calibration(cal)
for held in range(4):
    rows=np.flatnonzero(cal['lab_id']==held)
    mask=(cal['lab_id']!=held); mask[rows[0]]=True
    assert np.allclose(m._calibration({k:v[mask] for k,v in cal.items()})[held],m._prior_calibration(held))
    for ix in rows:
        for sign in (-1,1):
            bad={k:v.copy() for k,v in cal.items()}; bad['true_stress'][ix]+=sign*3.1
            clean=rows[rows!=ix]; expected=np.polyfit(cal['cmd_stress'][clean],cal['true_stress'][clean],1)
            assert np.allclose(m._calibration(bad)[held],expected), (held,ix,sign)
print('Calibration: four sparse-channel archive paths and 80 single-point EMI injections passed')
