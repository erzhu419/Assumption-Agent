from pathlib import Path
exec(Path('/tmp/rheology03_cv.py').read_text().split('for weight in')[0])

# Check the rational terminal flow at t=0, where its slope contributes to the reported rate.
eps = 1e-5
tau, _ = s._fit_robust(model, y, r, dt, t)
c, rate = s._design(train, x, params, dt, np.array([0., eps, 2*eps]), True)(tau)
np.testing.assert_allclose(c[:, 0], 0., atol=1e-14)
np.testing.assert_allclose((4*c[:, 1]-c[:, 2])/(2*eps), rate, rtol=1e-7, atol=1e-8)

# Fit each ablation on fold-training labels only; compare response transfer, not pooled classification accuracy.
for name, removed in [('full', ()), ('no_gap_difference', (8,)), ('no_slip', (7,)), ('linear_flow', ())]:
    base = s._design(train, x, params, dt, t, name != 'linear_flow')
    def design(taus):
        a, b = base(taus)
        if removed:
            a, b = a.copy(), b.copy()
            a[..., removed] = 0
            b[..., removed] = 0
        return a, b
    ec, er = np.empty_like(y), np.empty_like(r)
    for lab in range(4):
        val = labs == lab
        ta, w = s._fit_robust(design, y, r, dt, t, row_mask=~val)
        c, rate = design(ta)
        ec[val], er[val] = (c @ w - y)[val], (rate @ w - r)[val]
    print(name, 'complete-coupon worst-lab MedAE: curve, rate', [round(max(np.median(np.abs(a[(labs==lab)&~missing])) for lab in range(4)), 6) for a in (ec, er)])
    if name == 'full':
        for lab in range(4):
            mask = (labs == lab) & ~missing
            print('lab', lab, 'complete-coupon signed median: curve, rate', np.round([np.median(ec[mask]), np.median(er[mask])], 6).tolist())
        print('full missing-coupon counts by lab', [int(np.sum((labs==lab)&missing)) for lab in range(4)])
