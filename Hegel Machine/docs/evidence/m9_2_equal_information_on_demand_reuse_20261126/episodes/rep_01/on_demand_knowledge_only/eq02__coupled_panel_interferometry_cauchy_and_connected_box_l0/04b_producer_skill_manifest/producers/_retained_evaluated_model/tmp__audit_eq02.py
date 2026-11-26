from pathlib import Path
exec(Path('/tmp/verify_eq02.py').read_text().split('for seed, kind in enumerate')[0])

# Targets near inclusion boundaries and reduced-rank blocks can reveal weak leaf bounds.
for seed, kind in enumerate(('paired','ar','common','rankdef')*3, 90):
    p = 18 + seed % 2
    rng = np.random.default_rng(seed)
    x = design(rng, 48, p, kind)
    box = (.20, .55, 1.2)[(seed-90)//4]
    l2, l0 = .005, .50*box**2
    w = box*rng.choice([-1.,1.],p)*rng.uniform(.7,1.3,p)
    if kind == 'paired':
        w[1::2] = -w[:2*(p//2):2]
    y = x @ w + rng.normal(0,.15*box,x.shape[0])
    start = perf_counter()
    try:
        cert = mod._BoxSearch(x,y,l0,l2,box).solve()
    except RuntimeError as e:
        print(f'{seed} {kind}: FAIL {perf_counter()-start:.3f}s {e}', flush=True)
        continue
    dt = perf_counter()-start
    gap, leaves, slack = replay(x,y,l0,l2,box,cert)
    print(f'{seed} {kind}: {dt:.3f}s gap={gap:.3g} leaves={leaves}', flush=True)

# Check whether a single bad reference center dominates held-out calibration prediction.
for seed in (31, 35, 38):
    study, theta, _ = study_for(seed, 19, 'paired')
    strata = study['panel']['stratum']
    groups = study['panel']['center_id']
    phase = np.angle(study['interferometry']['signal_phasor']*study['interferometry']['ref_phasor'].conj())
    h = np.column_stack((np.ones(len(strata)),strata==1,strata==2,mod._align(study['panel'],strata),phase))
    errors = []
    for c in np.unique(groups):
        cal = study['calibration_indices']
        train = cal[groups[cal]!=c]
        test = cal[groups[cal]==c]
        y = study['y_assay_raw']
        robust = mod._cauchy(h[train],y[train])
        ols = np.linalg.lstsq(h[train],y[train],rcond=None)[0]
        errors.append((np.median(abs(y[test]-h[test]@robust)),np.median(abs(y[test]-h[test]@ols)),np.linalg.norm(robust-theta)/np.linalg.norm(theta)))
    print('held-out center',seed,'worst [Cauchy MedAE, OLS MedAE, Cauchy theta rel L2]',np.max(errors,axis=0).round(5).tolist())
