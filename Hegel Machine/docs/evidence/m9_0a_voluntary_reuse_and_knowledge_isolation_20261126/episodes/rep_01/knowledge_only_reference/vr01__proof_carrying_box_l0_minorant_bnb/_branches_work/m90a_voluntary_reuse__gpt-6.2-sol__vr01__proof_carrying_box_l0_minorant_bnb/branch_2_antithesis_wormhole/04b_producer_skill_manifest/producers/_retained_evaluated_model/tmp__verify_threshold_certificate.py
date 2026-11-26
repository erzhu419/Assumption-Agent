import runpy
import numpy as np
ns = runpy.run_path('/tmp/check_proof_box.py')
solve = ns['m'].solve_proof_carrying_box_l0
check = ns['check']
for box in (7e-9, .25, 1.1):
    scales = np.r_[np.ones(8), np.full(7, 1e5), np.full(7, 1e8)]
    y = np.linspace(-1.3, 1.3, scales.size)
    q = dict(X=np.diag(scales), y=y, lam0=.22, lam2=.14, M=box)
    tau = min(1e-8, box)
    mag = np.abs(scales*y)/(scales**2 + q['lam2'])
    small = np.minimum(mag, tau)
    large = np.clip(mag, tau, box)
    def cost(b):
        return .5*(scales*b-np.abs(y))**2+.5*q['lam2']*b*b
    optimum = np.minimum(cost(small), cost(large)+q['lam0']).sum()
    result = solve(q)
    stats = check(q, result)
    assert result['certified_lower_bound'] <= optimum + 1e-11 <= result['objective'] + 1e-10
    assert result['objective'] - optimum <= 1e-5
    # Independently evaluate the literal-threshold minorant on every leaf.
    x = q['X'].astype(np.longdouble)
    for leaf in result['subcube_leaves']:
        d = leaf['minorant_diagonal'].astype(np.longdouble)
        c = leaf['minorant_center'].astype(np.longdouble)
        z, a = leaf['fixed_zero'], leaf['fixed_active']
        r = y-x@c
        coefficient = np.abs(x.T@r+d*c)
        h = d+q['lam2']
        def gain(cap):
            v = np.minimum(coefficient/h,cap)
            return v*coefficient-.5*h*v*v
        off, on = -gain(tau), q['lam0']-gain(box)
        adjusted = .5*(r@r)+r@(x@c)+.5*((d*c)@c)+off[z].sum()+on[a].sum()+np.minimum(off,on)[~(z|a)].sum()
        assert leaf['lb'] <= adjusted + 1e-12
    print('mixed-scale orthogonal box', box, 'optimum', optimum, 'objective error', result['objective']-optimum, 'gap and leaves', stats[:2], flush=True)
