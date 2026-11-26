exec(open('/tmp/alpha_sparse_verify.py').read().split("check('both wings, two clean checks'")[0])
worst = {2: (0., None), 3: (0., None)}
for wing in ((0,1),(2,3)):
    b, target = batch(wing,10)
    checks = b['calibration_load_cells']; ids = checks['lab_id']
    x, _, m = s._align(b)
    for n in worst:
        for mode, ranks in [('wide',[0,5,9]), ('mid',[2,5,7]), ('narrow',[3,4,5])]:
            keep = ~np.isin(ids,wing); picks = []
            for l in wing:
                j = np.flatnonzero(ids==l); j = j[np.argsort(checks['cmd_stress'][j])]
                selected = j[ranks if n==3 else ranks[::2]]
                keep[selected] = True; picks.append(selected)
            for ix in range(n):
                for sign in (-1,1):
                    c = {k:v[keep].copy() for k,v in checks.items()}
                    for j in picks: c['true_stress'][np.sum(keep[:j[ix]])] += sign*.15
                    q = {**b,'calibration_load_cells':c}
                    cal = s._sparse_calibration(q,s._calibration(q),x,m)
                    err = float(np.max(np.abs(cal-target) @ [2.5,1.]))
                    if err>worst[n][0]: worst[n]=(err,(wing,mode,ix,sign))
print('Worst sparse EMI errors and witnesses:', worst)
assert worst[3][0] < .025 and worst[2][0] < .095
