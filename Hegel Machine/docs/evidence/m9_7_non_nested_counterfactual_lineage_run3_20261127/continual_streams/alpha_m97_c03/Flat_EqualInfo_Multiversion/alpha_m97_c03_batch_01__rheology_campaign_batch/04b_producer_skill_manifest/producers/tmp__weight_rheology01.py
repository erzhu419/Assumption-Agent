exec(open('/tmp/check_rheology01.py').read().split("print('active'")[0])
from pathlib import Path
source=Path(base+'alpha_m97_c03_batch_01_solver.py').read_text()
lab=d['train']['lab_id']
complete=~np.isnan(d['train']['coupon_features_raw']).any(1)
for rate_weight in (0.5,1.,2.):
    ns={}
    s=source.replace("    def model(p, jac=False):", f"    obs_weight = np.tile(np.r_[np.ones(len(args[-1])), {rate_weight!r}], len(target)//(len(args[-1])+1))\n    def model(p, jac=False):")
    s=s.replace("return np.einsum('ij,j->i', a, p[2:]) - target", "return obs_weight * (np.einsum('ij,j->i', a, p[2:]) - target)")
    s=s.replace('return np.column_stack((h, a))','return obs_weight[:, None] * np.column_stack((h, a))')
    exec(compile(s,'<weight-probe>','exec'),ns)
    folds=[]; spectra=[]
    for held in range(4):
        keep=lab!=held; test=~keep; loc=complete[test]
        tau,w=ns['_fit']((*[v[:n][keep] for v in args],dt,t),yy[keep].ravel(),'all')
        spectra.append(tau)
        e=np.abs(m._basis(np.log(tau),*[v[:n][test] for v in args],dt,t,'all')[0]@w-yy[test])
        folds.append([np.median(e[:,:-1]),np.median(e[:,-1]),np.median(e[loc,:-1]),np.median(e[loc,-1]),np.quantile(e[loc,:-1],.8),np.quantile(e[loc,-1],.8)])
    print('rate residual weight',rate_weight,'worst fold medians all and complete, then complete p80 strain/rate:',np.round(np.max(folds,axis=0),6),'fold slow taus:',np.round(np.asarray(spectra)[:,1],5))
