from verify_multicenter import solver, check, np
import time
rng = np.random.default_rng(825)
worst = {}
for family in ('shuffled band', 'dense', 'box saturation', 'near activation'):
    rows = []
    for i in range(24):
        p = 19+i%4
        n = 44
        if family == 'dense':
            x = rng.normal(size=(n,p)); x /= np.linalg.norm(x,axis=0)
        else:
            g = np.eye(p); j=np.arange(p-1)
            g[j,j+1]=g[j+1,j]=rng.uniform(-.29,.29,p-1)
            x = np.linalg.qr(rng.normal(size=(n,p)))[0]@np.linalg.cholesky(g).T
            x = x[:,rng.permutation(p)]
        l0 = rng.uniform(.12,.32); l2 = rng.uniform(.07,.25)
        box = rng.uniform(.25,.50) if family=='box saturation' else rng.uniform(.75,1.4)
        if family == 'near activation':
            target = rng.choice([-1,1],p)*np.sqrt(2*l0*(1+l2))*rng.uniform(.94,1.06,p)
            y = x @ np.linalg.solve(x.T@x,target)
        else:
            y = x@rng.uniform(-1.35,1.35,p)+rng.normal(0,.11,n)
        s = dict(X=x,y=y,M=box,lam0=l0,lam2=l2)
        start = time.monotonic()
        try:
            leaves,gap = check(s,solver._certify(s))
            rows.append((leaves,gap,time.monotonic()-start,i))
        except RuntimeError as e:
            print('FAILED',family,i,e,flush=True)
            np.savez(f'/tmp/multi_failure_{family.replace(" ","_")}_{i}.npz',**s)
    if rows:
        print(family, 'passed', len(rows), 'max leaves',max(rows),'max gap',max(r[1] for r in rows),flush=True)
# For duplicate columns any positive diagonal entry violates PSD, by testing
# the difference vector. If a selected column is free, its box-envelope value
# at the exact one-active coefficient is below its charged value by this much.
l0,l2,M,y=.22,.14,1.1,.84
v=y/(1+l2)
slope=l0/M+.5*l2*M
print('duplicate optimum free-active envelope deficit',l0+.5*l2*v*v-slope*v,'optimal supports',2**11,'minimum leaves including all-zero',2**11+1)
